"""
段階28: 減る動き(N)の情報を3つの方向から確かめる(EXP-037)。

段階27(EXP-036)の「負の差分 N(min(Δθ, 0))に情報がありそう」という結果の確認。入力は data/processed/stage27/ の
N と、D(1次差分、段階13と同じ123件の窓から計算)。作り直さない。変換器・分類器・置換の仕方は段階27と同じ
(MiniRocket(10000)→ StandardScaler(with_mean=False)→ RidgeClassifierCV(logspace(−3, 3, 10) × n_train)、
置換は市町村×年の中でラベルを並べ替える)。W・B は段階12・13と同じ定義。
  部分A(Q1、回し直しなし)  N − P の差。段階27の bootstrap_auc.csv の同じリサンプルどうしで全体の AUC の差の95%区間。
                          W・B の N − P は partial_*.pkl のスコアから、同じリサンプル(段階27と同じ乱数の列)で計算する。
                          判定: 区間が0より上 → 減る動きのほうが情報が多い / 0を含む → 検出できない。事後の比較(探索的)
  部分B(Q2)  段階20と同じ方法での再確認: カーネルの乱数10〜19、分割の乱数20〜39(make_splits(..., seed0=20))、
             置換 i の乱数は [1, i]・200回。判定(事前に固定、1つ): N の W が帰無分布の95%点を上回るか。
             記述: N の全体・B、同じ乱数・分割での D の全体・W・B(置換なし)、W の N − D(市町村のブートストラップ2000回)
  部分C(Q3)  段階21と同じ leave-one-year-out: テストの年 2016・2017、学習はほかの年の123件の中の全件(2018年は学習だけ)。
             カーネルの乱数0〜9、置換 i の乱数は [2, i]・200回。判定: N の W(2年合わせ)が帰無分布の95%点を上回るか。
             記述: 年ごとの W・B・全体 AUC、D(置換なし)、段階21の R_all_LOYO・M_diff_LOYO との比較

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage28\\run_stage28.py --part A
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage28\\run_stage28.py --part B C
  試運転: ... --part B C --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage28/。入力 × 部分 × 乱数ごとの途中結果を partial/ に保存し、同じ引数なら読み込んで再開する。
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
for st in ['stage13', 'stage16', 'stage27']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', st))
import run_stage13 as s13  # noqa: E402
import run_stage16 as s16  # noqa: E402
import run_stage27 as s27  # noqa: E402
from src.evaluation import make_splits, permute_within  # noqa: E402
from src.windows import first_diff  # noqa: E402

s6, s12, fast = s13.s6, s13.s12, s13.fast
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage28')
PART_DIR = os.path.join(OUT_DIR, 'partial')
S27 = os.path.join(ROCKET_DIR, 'outputs', 'stage27')
S21 = os.path.join(ROCKET_DIR, 'outputs', 'stage21')
D27 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage27')
D13 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13', 'label_muniyear_median')
N_EXPECT = 123
B_SEEDS, B_SPLIT0, B_PERM = list(range(10, 20)), 20, 1
C_SEEDS, C_PERM, C_YEARS = list(range(10)), 2, [2016, 2017]
TYPES = ['W', 'B', 'X']
INK, MUTED = s6.INK, s6.MUTED
COL = {'D': '#52514e', 'N': '#eb6834', 'P': '#2a78d6'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def load(log):
    X = np.load(D13 + '_X.npy')
    meta = pd.read_csv(D13 + '_meta.csv', encoding='utf-8-sig', dtype={'muniCd': str})
    m27 = pd.read_csv(os.path.join(D27, 'label_muniyear_median_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    N = np.load(os.path.join(D27, 'label_muniyear_median_N_X.npy'))
    Xd = first_diff(X)
    ok = len(meta) == N_EXPECT and meta[['field_id', 'year']].equals(m27[['field_id', 'year']]) and \
        np.allclose(N, np.minimum(Xd, 0)) and N.shape == Xd.shape
    log(f'確認 入力: 123件、段階27の N(data/processed/stage27)と meta の並びが段階13と同じ、N = min(Δθ, 0): {ok}')
    assert ok
    return {'N': N, 'D': Xd}, meta


def cached(name, tag, fn):
    """partial/<name>.pkl があり tag が同じなら読み込み、無ければ fn() を計算して保存する。"""
    os.makedirs(PART_DIR, exist_ok=True)
    path = os.path.join(PART_DIR, f'{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            saved = pickle.load(fh)
        if saved['tag'] == tag:
            return saved['out'], True
    out = fn()
    with open(path, 'wb') as fh:
        pickle.dump({'tag': tag, 'out': out}, fh)
    return out, False


# ---------------------------------------------------------------- 部分A
def part_a(log):
    log('\n==== 部分A: N − P(段階27のブートストラップのリサンプルを使う、回し直しなし) ====')
    boot = pd.read_csv(os.path.join(S27, 'bootstrap_auc.csv'), encoding='utf-8-sig')
    ok_cols = {'boot', 'N', 'P', 'D'} <= set(boot.columns) and boot.boot.is_unique and len(boot) == 2000
    log(f'確認A1 bootstrap_auc.csv に入力ごとの値がリサンプルの番号(boot)で対応づけて保存されている: {ok_cols}'
        f'(列 {list(boot.columns)}、{len(boot)} 行)')
    assert ok_cols
    meta = pd.read_csv(os.path.join(D27, 'label_muniyear_median_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    S = {}
    for nm in ['N', 'P']:
        with open(os.path.join(S27, f'partial_{nm}.pkl'), 'rb') as fh:
            S[nm] = pickle.load(fh)['out']['S']
    C, Nn = {}, None
    for nm in S:
        C[nm], Nk = s27.pair_sums(S[nm], meta, sp)
        Nn = Nk if Nn is None else Nn
        assert np.array_equal(Nk, Nn)
    # 段階27の boot_diff と同じ乱数の列でリサンプルし、全体・W・B を出す
    cd = meta.muniCd.to_numpy(dtype=str)
    my = meta.muniyear.to_numpy(dtype=str)
    yr = meta.year.values
    units, cidx = np.unique(cd, return_inverse=True)
    same = cd[:, None] == cd[None, :]
    mW = my[:, None] == my[None, :]
    mB = (yr[:, None] == yr[None, :]) & ~mW
    rng = np.random.default_rng(0)
    rows = []
    for b in range(len(boot)):
        k = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = k[cidx].astype(float)
        r = {'boot': b}
        nn_all = w @ (Nn * same).sum(axis=1) + w @ (Nn * ~same) @ w
        nW = w @ (Nn * mW).sum(axis=1)
        nB = w @ (Nn * mB) @ w
        for nm in S:
            r[f'all_{nm}'] = (w @ (C[nm] * same).sum(axis=1) + w @ (C[nm] * ~same) @ w) / nn_all
            r[f'W_{nm}'] = (w @ (C[nm] * mW).sum(axis=1)) / nW if nW > 0 else np.nan
            r[f'B_{nm}'] = (w @ (C[nm] * mB) @ w) / nB if nB > 0 else np.nan
        rows.append(r)
    bt = pd.DataFrame(rows)
    ok_same = np.allclose(bt.all_N, boot.N, rtol=0, atol=1e-12) and np.allclose(bt.all_P, boot.P, rtol=0, atol=1e-12)
    log(f'確認A2 計算し直した全体の AUC(N・P、2000リサンプル)が段階27の bootstrap_auc.csv と一致(同じリサンプル): {ok_same}')
    assert ok_same
    pt = {}
    for nm in S:
        pt[f'all_{nm}'] = C[nm].sum() / Nn.sum()
        pt[f'W_{nm}'] = (C[nm] * mW).sum() / (Nn * mW).sum()
        pt[f'B_{nm}'] = (C[nm] * mB).sum() / (Nn * mB).sum()
    folds27 = pd.read_csv(os.path.join(S27, 'folds.csv'), encoding='utf-8-sig')
    ps27 = pd.read_csv(os.path.join(S27, 'per_seed.csv'), encoding='utf-8-sig')
    fm = {nm: folds27[folds27.config == nm].auc.mean() for nm in ['N', 'P']}
    out = []
    for met in ['all', 'W', 'B']:
        d = bt[f'{met}_N'] - bt[f'{met}_P']
        if met == 'all':
            point, point_note = fm['N'] - fm['P'], 'fold の AUC の平均の差(段階27の summary と同じ)'
        else:
            point = ps27[ps27.config == 'N'][met].mean() - ps27[ps27.config == 'P'][met].mean()
            point_note = '段階27の W・B(繰り返しごとに合算・乱数で平均)の差'
        out.append({'metric': met, 'diff': 'N − P', 'point': point, 'point_note': point_note,
                    'point_pooled_pairs': pt[f'{met}_N'] - pt[f'{met}_P'],
                    'boot_q025': d.quantile(.025), 'boot_q975': d.quantile(.975), 'boot_frac_pos': (d > 0).mean(),
                    'n_boot': len(d), 'role': '判定' if met == 'all' else '記述'})
    res = pd.DataFrame(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, 'partA_diff.csv'), index=False, encoding='utf-8-sig')
    bt.to_csv(os.path.join(OUT_DIR, 'partA_bootstrap.csv'), index=False, encoding='utf-8-sig')
    log(res.round(4).to_string(index=False))
    r = res.iloc[0]
    log(f'[判定 部分A] 全体の AUC の N − P {r.point:+.3f} [{r.boot_q025:+.3f}, {r.boot_q975:+.3f}] → '
        + ('区間が0より上: 減る動きのほうが増える動きより情報が多い(事後の比較、探索的)' if r.boot_q025 > 0
           else '区間が0を含む: 差は検出できない' if r.boot_q975 >= 0 else '区間が0より下'))
    return res


# ---------------------------------------------------------------- 学習(fold ごと)
def _fold(seed, X, tr, te, Ytr):
    """行0 = 観測ラベル(sklearn と一致を確かめる)、行1.. = 置換(まとめて解き、あいまいな行は学習し直す)。"""
    Ftr, Fte = s6._one_fold('minirocket', seed, X, tr, te)
    alpha, dec, unsure = fast._batched_ridge(Ftr, Fte, Ytr)
    ref = s6.RidgeNTrain().fit(Ftr, Ytr[0])
    rd = ref.decision_function(Fte)
    ok = bool(np.isclose(alpha[0], ref.alpha_) and np.allclose(dec[0], rd, rtol=1e-7, atol=1e-10))
    alpha[0], dec[0] = ref.alpha_, rd
    redo = np.flatnonzero(unsure[1:]) + 1
    for i in redo:
        dec[i] = s6.RidgeNTrain().fit(Ftr, Ytr[i]).decision_function(Fte)
    return alpha[0] / len(tr), dec, ok, len(redo)


def _fold_check(seed, X, tr, te, Ytr):
    """確認用: 置換ラベルを sklearn で1回ずつ学習した決定関数。"""
    Ftr, Fte = s6._one_fold('minirocket', seed, X, tr, te)
    return np.array([s6.RidgeNTrain().fit(Ftr, yy).decision_function(Fte) for yy in Ytr])


def perms(y, meta, seed0, P):
    my = meta.muniyear.to_numpy(dtype=str)
    return np.array([permute_within(y, my, np.random.default_rng([seed0, i])) for i in range(P)]).reshape(P, len(y))


# ---------------------------------------------------------------- 部分B
def part_b(inputs, meta, seeds, P, n_boot, log):
    log(f'\n==== 部分B: 同じデータの中での再確認(カーネル {seeds[0]}〜{seeds[-1]}、分割 {B_SPLIT0}〜{B_SPLIT0 + 19}、'
        f'置換 [{B_PERM}, i]・{P}回) ====')
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, B_SPLIT0)
    sp0 = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    n_diff = sum(not np.array_equal(np.sort(a[3]), np.sort(b[3])) for a, b in zip(sp, sp0))
    sp_p = make_splits(perms(y, meta, B_PERM, 3)[0], meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, B_SPLIT0)
    same_p = all(np.array_equal(a[3], b[3]) for a, b in zip(sp, sp_p))
    log(f'確認B1 分割(乱数 {B_SPLIT0}〜)は段階27(0〜19)と同じ位置の fold でテスト件が違う {n_diff}/{len(sp)}、'
        f'カーネルの乱数 {seeds[0]}〜{seeds[-1]} は段階27(0〜9)と重ならない {not set(seeds) & set(range(10))}、'
        f'置換しても分割が変わらない {same_p}')
    assert n_diff >= 1 and not set(seeds) & set(range(10)) and same_p
    Yp = perms(y, meta, B_PERM, P)
    ps = (Yp, Yp, Yp, np.arange(1, P + 1), np.arange(1, P + 1), 'muniyear')
    # 確認: まとめて解いた置換 = sklearn(N、最初の乱数、置換20回)
    nq = min(20, P)
    Ytr_all = np.vstack([y, Yp[:nq]])
    a = Parallel(n_jobs=s13.N_JOBS)(delayed(_fold)(seeds[0], inputs['N'], tr, te, Ytr_all[:, tr]) for r, f, tr, te in sp)
    b = Parallel(n_jobs=s13.N_JOBS)(delayed(_fold_check)(seeds[0], inputs['N'], tr, te, Ytr_all[:, tr]) for r, f, tr, te in sp)
    ok = all(x[2] for x in a) and all(np.allclose(x[1], z, rtol=1e-6, atol=1e-9) for x, z in zip(a, b))
    log(f'確認B2 置換{nq}回 × {len(sp)} fold(N、乱数{seeds[0]}): まとめて解いた決定関数 = sklearn で1回ずつ: {ok}')
    assert ok
    res = {}
    for name, P_use in [('N', P), ('D', 0)]:
        S = np.full((len(seeds), s6.N_REPEATS, len(y)), np.nan)
        rows, nulls = [], {'all': [], 'W': [], 'B': []}
        for si, seed in enumerate(seeds):
            Ytr_all = np.vstack([y, Yp[:P_use]])

            def run():
                return Parallel(n_jobs=s13.N_JOBS)(delayed(_fold)(seed, inputs[name], tr, te, Ytr_all[:, tr])
                                                   for r, f, tr, te in sp)
            rr, hit = cached(f'B_{name}_k{seed}', {'P': P_use, 'split0': B_SPLIT0, 'perm': B_PERM}, run)
            assert all(x[2] for x in rr), f'{name} 乱数{seed}: 観測ラベルで sklearn と一致しない fold がある'
            decs = [x[1] for x in rr]
            for (r, f, tr, te), d in zip(sp, decs):
                S[si, r, te] = d[0]
            ref = pd.DataFrame([{'repeat': r, 'fold': f, 'auc': roc_auc_score(y[te], d[0]) if len(set(y[te])) == 2 else np.nan,
                                 'bacc': balanced_accuracy_score(y[te], (d[0] > 0).astype(int))}
                                for (r, f, tr, te), d in zip(sp, decs)])
            if P_use:
                obs, null = s13.analyse(decs, meta, sp, ps)
                for k in nulls:
                    nulls[k].append(null[k])
            else:
                obs, _ = s13.analyse([np.vstack([d[:1], d[:1]]) for d in decs], meta, sp,
                                     (y[None, :], y[None, :], y[None, :], np.array([1]), np.array([1]), 'muniyear'))
            s13.check_obs(f'B {name} 乱数{seed}', obs, ref, log)
            row = {'part': 'B', 'config': name, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
                   'alpha_at_max': np.mean([np.isclose(x[0], s6.BASE_ALPHAS[-1]) for x in rr]),
                   'n_redo': sum(x[3] for x in rr), 'cached': hit}
            for t in TYPES:
                row[t] = np.nanmean(obs[t][0])
            row['n_pairs_W'] = obs['W'][1].mean()
            rows.append(row)
            log(f'    B {name} 乱数{seed}: 全体 {row["all_auc"]:.3f}  W {row["W"]:.3f}  B {row["B"]:.3f}  '
                f'alpha 上限 {row["alpha_at_max"]:.2f}' + ('(途中結果を読み込み)' if hit else ''))
        res[name] = {'per_seed': pd.DataFrame(rows), 'S': S,
                     'null': {k: np.mean(np.stack(v), axis=0) for k, v in nulls.items()} if P_use else None}
    # まとめ
    srows = []
    for name in ['N', 'D']:
        p = res[name]['per_seed']
        nl = res[name]['null']
        for met in ['all_auc', 'all_bacc', 'W', 'B', 'X']:
            r = {'part': 'B', 'config': name, 'metric': met, 'mean': p[met].mean(), 'seed_min': p[met].min(),
                 'seed_max': p[met].max(), 'n_seeds': len(p), 'n_perm': P if nl else 0,
                 'alpha_at_max': p.alpha_at_max.mean(), 'role': '判定' if (name == 'N' and met == 'W') else '記述'}
            if nl is not None and met in ('all_auc', 'all_bacc', 'W', 'B'):
                v = nl['all'][:, 1 if met == 'all_auc' else 0] if met.startswith('all') else nl[met]
                np.save(os.path.join(OUT_DIR, f'null_partB_N_{met}.npy'), v)
                r.update({'null_mean': np.nanmean(v), 'null_q95': np.nanquantile(v, .95),
                          'p': (1 + np.sum(v >= p[met].mean())) / (1 + P)})
            if met == 'W':
                r['n_pairs_per_repeat'] = p.n_pairs_W.mean()
            srows.append(r)
    summ = pd.DataFrame(srows)
    boot, pt = s16.w_diff_bootstrap(res['N']['S'], res['D']['S'], meta, sp, n_boot)
    boot = boot.rename(columns={'W_GC_date': 'W_N', 'W_M_diff': 'W_D'})
    boot.to_csv(os.path.join(OUT_DIR, 'partB_w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    wd = {'part': 'B', 'config': 'N − D', 'metric': 'W_diff', 'mean': res['N']['per_seed'].W.mean() - res['D']['per_seed'].W.mean(),
          'point_pooled_pairs': pt['a'] - pt['b'], 'boot_q025': boot.W_diff.quantile(.025),
          'boot_q975': boot.W_diff.quantile(.975), 'boot_frac_pos': (boot.W_diff > 0).mean(), 'n_boot': n_boot, 'role': '記述'}
    summ = pd.concat([summ, pd.DataFrame([wd])], ignore_index=True)
    summ.to_csv(os.path.join(OUT_DIR, 'partB_summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat([res[n]['per_seed'] for n in res]).to_csv(os.path.join(OUT_DIR, 'partB_per_seed.csv'), index=False,
                                                       encoding='utf-8-sig')
    return summ, res


# ---------------------------------------------------------------- 部分C
def part_c(inputs, meta, seeds, P, log):
    log(f'\n==== 部分C: 年をまたいだ確認(LOYO、テストの年 {C_YEARS}、カーネル {seeds[0]}〜{seeds[-1]}、'
        f'置換 [{C_PERM}, i]・{P}回) ====')
    y, yr = meta.y.values, meta.year.values
    fl = []
    for Y in C_YEARS:
        tr, te = np.flatnonzero(yr != Y), np.flatnonzero(yr == Y)
        my = meta.muniyear.to_numpy(dtype=str)[te]
        w = sum(int((y[te][my == g] == 1).sum() * (y[te][my == g] == 0).sum()) for g in np.unique(my))
        log(f'  {Y}年: テスト {len(te)} 件(市町村 {meta.muniCd.values[te].astype(str).tolist().__len__() and len(set(meta.muniCd.values[te]))})、'
            f'W のペア {w}、学習(ほかの年){len(tr)} 件')
        assert not (yr[tr] == Y).any()
        fl.append((Y, tr, te))
    Yp = perms(y, meta, C_PERM, P)
    res = {}
    for name, P_use in [('N', P), ('D', 0)]:
        rows, nullW, nullB, nullYear = [], [], [], []
        for seed in seeds:
            Ytr_all = np.vstack([y, Yp[:P_use]])

            def run():
                return Parallel(n_jobs=min(s13.N_JOBS, len(fl)))(delayed(_fold)(seed, inputs[name], tr, te, Ytr_all[:, tr])
                                                                 for Y, tr, te in fl)
            rr, hit = cached(f'C_{name}_k{seed}', {'P': P_use, 'perm': C_PERM, 'years': C_YEARS}, run)
            assert all(x[2] for x in rr)
            Lall = np.vstack([y, Yp[:P_use]])
            cnt = {}
            row = {'part': 'C', 'config': name, 'seed': seed, 'cached': hit,
                   'alpha_at_max': np.mean([np.isclose(x[0], s6.BASE_ALPHAS[-1]) for x in rr])}
            for (Y, tr, te), x in zip(fl, rr):
                m = s12.pair_masks(meta, te)
                m['ALL'] = np.ones((len(te), len(te)), bool)
                c = s12.pair_counts(x[1], Lall[:, te], m)
                cnt[Y] = c
                with np.errstate(invalid='ignore', divide='ignore'):
                    for t in ['W', 'B', 'ALL']:
                        row[f'{t if t != "ALL" else "all_auc"}_{Y}'] = c[t][0][0] / c[t][1][0] if c[t][1][0] else np.nan
                row[f'all_bacc_{Y}'] = balanced_accuracy_score(y[te], (x[1][0] > 0).astype(int))
                ok = np.isclose(row[f'all_auc_{Y}'], roc_auc_score(y[te], x[1][0]), rtol=0, atol=1e-12) and \
                    c['X'][1][0] == 0 and np.isclose(sum(c[t][0][0] for t in TYPES) / sum(c[t][1][0] for t in TYPES),
                                                     row[f'all_auc_{Y}'], rtol=0, atol=1e-12)
                assert ok, f'C {name} 乱数{seed} {Y}年: AUC が sklearn と一致しない'
            with np.errstate(invalid='ignore', divide='ignore'):
                CW = sum(cnt[Y]['W'][0] for Y in C_YEARS)
                NW = sum(cnt[Y]['W'][1] for Y in C_YEARS)
                CB = sum(cnt[Y]['B'][0] for Y in C_YEARS)
                NB = sum(cnt[Y]['B'][1] for Y in C_YEARS)
                Wv, Bv = CW / NW, CB / NB
                allv = np.mean([cnt[Y]['ALL'][0] / cnt[Y]['ALL'][1] for Y in C_YEARS], axis=0)
            row.update({'W': Wv[0], 'B': Bv[0], 'all_auc': allv[0], 'n_pairs_W': NW[0], 'n_pairs_B': NB[0]})
            if P_use:
                nullW.append(Wv[1:])
                nullB.append(Bv[1:])
                nullYear.append(allv[1:])
            rows.append(row)
            log(f'    C {name} 乱数{seed}: W(2年){row["W"]:.3f}  B {row["B"]:.3f}  全体(2年の平均){row["all_auc"]:.3f}  '
                + '  '.join(f'{Y}: W {row[f"W_{Y}"]:.3f} B {row[f"B_{Y}"]:.3f} 全体 {row[f"all_auc_{Y}"]:.3f}' for Y in C_YEARS)
                + ('(途中結果を読み込み)' if hit else ''))
        res[name] = {'per_seed': pd.DataFrame(rows),
                     'null': {'W': np.mean(nullW, axis=0), 'B': np.mean(nullB, axis=0),
                              'all_auc': np.mean(nullYear, axis=0)} if P_use else None}
        if name == 'D':
            p21 = pd.read_csv(os.path.join(S21, 'per_seed.csv'), encoding='utf-8-sig')
            p21 = p21[p21.config == 'M_diff_LOYO'].set_index('seed')
            cols = ['W', 'B', 'W_2016', 'B_2016', 'all_auc_2016', 'W_2017', 'B_2017', 'all_auc_2017']
            pd_ = res['D']['per_seed'].set_index('seed')
            okd = all(np.allclose(pd_.loc[s, cols].astype(float).values, p21.loc[s, cols].astype(float).values,
                                  rtol=0, atol=1e-12) for s in seeds)
            log(f'確認C1 D_LOYO が段階21の M_diff_LOYO と一致(乱数 {seeds[0]}〜{seeds[-1]}、W・B・年ごとの指標): {okd}')
            assert okd, 'D_LOYO が段階21と一致しない。止めて報告する'
    srows = []
    for name in ['N', 'D']:
        p, nl = res[name]['per_seed'], res[name]['null']
        mets = ['W', 'B', 'all_auc'] + [f'{t}_{Y}' for Y in C_YEARS for t in ['W', 'B', 'all_auc', 'all_bacc']]
        for met in mets:
            r = {'part': 'C', 'config': f'{name}_LOYO', 'metric': met, 'mean': p[met].mean(), 'seed_min': p[met].min(),
                 'seed_max': p[met].max(), 'n_seeds': len(p), 'n_perm': P if nl else 0,
                 'alpha_at_max': p.alpha_at_max.mean(), 'role': '判定' if (name == 'N' and met == 'W') else '記述'}
            if nl is not None and met in nl:
                v = nl[met]
                np.save(os.path.join(OUT_DIR, f'null_partC_N_{met}.npy'), v)
                r.update({'null_mean': np.nanmean(v), 'null_q95': np.nanquantile(v, .95),
                          'p': (1 + np.sum(v >= p[met].mean())) / (1 + P)})
            if met == 'W':
                r['n_pairs'] = p.n_pairs_W.mean()
            srows.append(r)
    summ = pd.DataFrame(srows)
    # 段階21との比較
    s21 = pd.read_csv(os.path.join(S21, 'summary.csv'), encoding='utf-8-sig')
    s21 = s21[s21.config.isin(['R_all_LOYO', 'M_diff_LOYO'])]
    ref21 = s21[['config', 'metric', 'mean'] + [c for c in ['null_q95', 'p'] if c in s21.columns]]
    summ = pd.concat([summ, ref21.assign(part='C(段階21、参考)', role='参考')], ignore_index=True)
    summ.to_csv(os.path.join(OUT_DIR, 'partC_summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat([res[n]['per_seed'] for n in res]).to_csv(os.path.join(OUT_DIR, 'partC_per_seed.csv'), index=False,
                                                       encoding='utf-8-sig')
    return summ, res


# ---------------------------------------------------------------- 図
def plot_b(summ):
    sm = summ[summ.part == 'B'].set_index(['config', 'metric'])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    mets = [('all_auc', '全体'), ('W', '地域内 W(N が判定)'), ('B', '地域間 B')]
    w = 0.36
    for k, c in enumerate(['D', 'N']):
        for i, (m, _) in enumerate(mets):
            r = sm.loc[(c, m)]
            x = i + (k - 0.5) * w
            ax.bar(x, r['mean'], width=w * 0.92, color=COL[c], alpha=0.8, label=c if i == 0 else None)
            ax.text(x, r['mean'] + 0.006, f'{r["mean"]:.3f}', ha='center', fontsize=9, color=INK)
            if pd.notna(r.get('null_q95')):
                ax.hlines(r['null_q95'], x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--',
                          label='N の帰無分布の95%点' if i == 0 else None)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(0.4, 0.72)
    ax.set_ylabel('AUC')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_title('部分B: 乱数(カーネル10〜19・分割20〜39・置換 [1, i])を変えた再確認', loc='left', fontsize=11, color=INK)
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'partB_w.png'), dpi=150)
    plt.close(fig)


def plot_c(summ):
    s = summ.set_index(['config', 'metric'])['mean']
    cfgs = [('N_LOYO', 'N', COL['N']), ('D_LOYO', 'D', COL['D']), ('M_diff_LOYO', 'M_diff(段階21)', '#9a9890'),
            ('R_all_LOYO', 'R_all(段階21)', '#2a78d6')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (lab, keys) in zip(axs, [('W(2年合わせ)と年ごとの W', ['W', 'W_2016', 'W_2017']),
                                      ('B', ['B', 'B_2016', 'B_2017']),
                                      ('全体 AUC', ['all_auc', 'all_auc_2016', 'all_auc_2017'])]):
        w = 0.2
        for k, (c, nm, col) in enumerate(cfgs):
            vals = [s.get((c, m), np.nan) for m in keys]
            xs = np.arange(len(keys)) + (k - 1.5) * w
            ax.bar(xs, vals, width=w * 0.92, color=col, alpha=0.8, label=nm)
        if lab.startswith('W'):
            q = summ[(summ.config == 'N_LOYO') & (summ.metric == 'W')].null_q95.iloc[0]
            ax.hlines(q, -0.45, 0.45, color=s6.C_RED, lw=2, ls='--', label='N の W の95%点')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks(range(len(keys)), ['2年合わせ' if '_' not in m or m == 'all_auc' else m.split('_')[-1] + '年' for m in keys])
        ax.set_title(lab, loc='left', fontsize=11, color=INK)
        ax.set_ylim(0.3, 0.8)
        s6.style(ax)
    axs[0].set_ylabel('AUC')
    axs[0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('部分C: 年をまたいだ確認(leave-one-year-out、テストの年 2016・2017)', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'partC_years.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='+', default=['A', 'B', 'C'], choices=['A', 'B', 'C'])
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    s6.N_JOBS = s13.N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = '_'.join(args.part)
    logf = open(os.path.join(OUT_DIR, f'run_log_{tag}.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {s13.N_JOBS}')
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 段階28・EXP-037 が未使用(docs に EXP-037 なし): {"EXP-037" not in docs}')
    trial = '' if args.n_seeds == 10 and args.n_perm == 200 and args.n_boot == 2000 else '(試運転)'
    if 'A' in args.part:
        part_a(log)
    if 'B' in args.part or 'C' in args.part:
        inputs, meta = load(log)
    if 'B' in args.part:
        sb, _ = part_b(inputs, meta, B_SEEDS[:args.n_seeds], args.n_perm, args.n_boot, log)
        plot_b(sb)
        log('\n[部分B のまとめ]')
        log(sb.drop(columns=['part']).round(3).to_string(index=False))
        r = sb[(sb.config == 'N') & (sb.metric == 'W')].iloc[0]
        log(f'[判定 部分B] N の W {r["mean"]:.3f}、帰無分布の95%点 {r.null_q95:.3f}、p = {r.p:.3f} → '
            + ('上回る: 同じデータの中で再現した' if r['mean'] > r.null_q95 else '上回らない: 再現しなかった') + trial)
    if 'C' in args.part:
        sc, _ = part_c(inputs, meta, C_SEEDS[:args.n_seeds], args.n_perm, log)
        plot_c(sc)
        log('\n[部分C のまとめ]')
        log(sc.drop(columns=['part']).round(3).to_string(index=False))
        r = sc[(sc.config == 'N_LOYO') & (sc.metric == 'W')].iloc[0]
        log(f'[判定 部分C] N の W(2年合わせ){r["mean"]:.3f}、帰無分布の95%点 {r.null_q95:.3f}、p = {r.p:.3f} → '
            + ('上回る: 年をまたいで再現した' if r['mean'] > r.null_q95 else '上回らない: 年をまたいで再現しなかった') + trial)
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
