"""
段階20: R_all の地域内 W の再確認(新しい乱数と分割、EXP-029)。

段階19(EXP-028)の R_all(相対収量 y_rel の回帰、学習235件、評価は median の123件)で、地域内 W が 0.602 となり、
主ラベルで初めて帰無分布の95%点(0.590)を上回った。W は段階19では事前の判定に入れていない記述の指標だったので、
R_all の設定を変えずに、カーネルの乱数と分割の乱数を新しくし、判定を W に固定して回し直す。
データは同じなので独立した検証ではない(同じデータの中での再確認)。仮説は段階19の結果を見てから立てたもの。

段階19から変えるもの
  カーネルの乱数 10〜19(段階19は0〜9)
  分割の乱数 20〜39(make_splits(y, muniCd, 5, 20, seed0=20)。段階13・19は0〜19)
  置換 i の乱数 [1, i](段階19は [0, i])
変えないもの: データ(data/processed/stage19/)、学習側の決め方、目的変数、モデル、置換の設計
比べる相手 M_diff も同じ乱数・分割で計算し直す(学習は123件、RidgeClassifierCV)。置換は市町村×年の中でラベルを
並べ替え、200回(記述用、乱数 [1, i])。

判定(事前に固定、主は1つだけ): R_all の W の AUC(乱数10通りの平均)が、W の帰無分布(置換ごとの W を乱数10通りで
平均)の95%点を上回るか。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage20\\run_stage20.py
  試運転: ... run_stage20.py --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage20/
"""
import argparse
import os
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage19'))
import run_stage19 as s19  # noqa: E402  (stage6〜16 と scripts/ も sys.path に入る)
from run_stage19 import (BASE_ALPHAS, LABEL, N_JOBS, N_REPEATS, N_SPLITS, TYPES, RidgeNTrain,  # noqa: E402
                         RidgeRegNTrain, fast, s6, s7, s14, s16)
from src.evaluation import make_splits, permute_within, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DATA19 = os.path.join(PROC, 'stage19')
OUT19 = os.path.join(ROCKET_DIR, 'outputs', 'stage19')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage20')
KERNEL_SEEDS = list(range(10, 20))
SPLIT_SEED0 = 20
PERM_SEED = 1  # 置換 i の乱数は [PERM_SEED, i]
CFGS = ['M_diff', 'R_all']
CFG_LABEL = {'M_diff': 'M_diff 分類・123件(計算し直し)', 'R_all': 'R_all 回帰・y_rel・235件'}
CFG_COLOR = {'M_diff': '#9a9890', 'R_all': '#2a78d6'}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- データ(段階19の入力を読む)
def load_data(log):
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    lab['muniyear'] = lab['muniCd'] + '_' + lab['year'].astype(str)
    lab['group_n_all'] = lab.muniyear.map(lab.groupby('muniyear').size())
    lab['group_mean_yield'] = lab.groupby('muniyear')['yield'].transform('mean')
    lab['y_rel'] = np.log(lab['yield'] / lab['group_mean_yield'])
    X = np.load(os.path.join(DATA19, 'eligible_X.npy'))
    meta = pd.read_csv(os.path.join(DATA19, 'eligible_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    meta['muniyear'] = meta['muniCd'] + '_' + meta['year'].astype(str)
    ok = np.allclose(meta.y_rel.values, lab.y_rel.values[meta.lab_pos.values], rtol=0, atol=1e-12) and \
        (meta[['field_id', 'year']].values == lab[['field_id', 'year']].values[meta.lab_pos.values]).all()
    log(f'[データ] data/processed/stage19/ の学習に使える件 {len(meta)}、全 {len(lab)} 件から計算し直した y_rel と一致 {ok}')
    assert ok
    return lab, X, meta


def build_folds(meta, i123, meta123, log):
    """新しい分割(seed0=20)。[(r, f, tr_all, pr_all, te123, ev_in_pr)]"""
    y123 = meta123.y.values
    sp = make_splits(y123, meta123.muniCd.values, N_SPLITS, N_REPEATS, SPLIT_SEED0)
    sp_old = make_splits(y123, meta123.muniCd.values, N_SPLITS, N_REPEATS, 0)
    n_diff = sum(not np.array_equal(np.sort(a[3]), np.sort(b[3])) for a, b in zip(sp, sp_old))
    old_sets = {tuple(np.sort(b[3])) for b in sp_old}
    n_new = sum(tuple(np.sort(a[3])) not in old_sets for a in sp)
    log(f'  確認2 新しい分割(乱数 {SPLIT_SEED0}〜{SPLIT_SEED0 + N_REPEATS - 1})と段階13・19の分割(0〜19): 同じ位置の fold で'
        f'テスト件が違う {n_diff} / {len(sp)}、段階13・19のどの fold とも違うテスト件の fold {n_new} / {len(sp)}')
    log(f'    乱数の重なり: カーネル {KERNEL_SEEDS[0]}〜{KERNEL_SEEDS[-1]}(段階19は0〜9)、分割 {SPLIT_SEED0}〜'
        f'{SPLIT_SEED0 + N_REPEATS - 1}(0〜19)、置換 [{PERM_SEED}, i]([0, i])→ 重なりなし '
        f'{not (set(KERNEL_SEEDS) & set(range(10))) and SPLIT_SEED0 >= N_REPEATS and PERM_SEED != 0}')
    assert n_diff >= 1 and not (set(KERNEL_SEEDS) & set(range(10))) and SPLIT_SEED0 >= N_REPEATS and PERM_SEED != 0
    muni = meta.muniCd.to_numpy(dtype=str)
    folds, ok_leak = [], True
    for r, f, tr, te in sp:
        tm = np.unique(meta123.muniCd.values[te])
        in_tm = np.isin(muni, tm)
        tr_all, pr_all = np.flatnonzero(~in_tm), np.flatnonzero(in_tm)
        assert np.array_equal(np.sort(pr_all[np.isin(pr_all, i123)]), np.sort(i123[te]))
        ok_leak &= not np.isin(muni[tr_all], tm).any() and not np.isin(i123[te], tr_all).any() \
            and not np.isin(meta123.muniCd.values[tr], tm).any()
        folds.append((r, f, tr_all, pr_all, te, np.array([np.flatnonzero(pr_all == i)[0] for i in i123[te]])))
    log(f'  確認3 学習側にテストの市町村の件が無い(R_all・M_diff、全 {len(sp)} fold): {ok_leak}')
    assert ok_leak
    ntr = np.array([len(x[2]) for x in folds])
    log(f'    R_all の学習件数: 平均 {ntr.mean():.1f}(最小 {ntr.min()}・最大 {ntr.max()})、M_diff の学習件数の平均 '
        f'{np.mean([len(tr) for _, _, tr, _ in sp]):.1f}')
    return sp, folds


# ---------------------------------------------------------------- 起動時の確認(4・5、乱数10)
def precheck_pipeline(Xd, yrel, folds, y123, seed, log):
    ok = True
    for r, f, tr, pr, te, ev in folds[:10]:
        pipe = make_pipeline(s6.Panel('minirocket', random_state=seed), StandardScaler(with_mean=False),
                             RidgeRegNTrain()).fit(Xd[tr], yrel[tr])
        a = pipe.predict(Xd[pr])
        Ftr, Fpr = s19._feats(seed, Xd, tr, pr)
        b = RidgeRegNTrain().fit(Ftr, yrel[tr]).predict(Fpr)
        ok &= np.allclose(a, b, rtol=0, atol=1e-12) and roc_auc_score(y123[te], a[ev]) == roc_auc_score(y123[te], b[ev])
    log(f'  確認4 Pipeline をそのまま回した結果 = 特徴量を使い回した結果(R_all、乱数{seed}、先頭10 fold): {ok}')
    assert ok


def precheck_batched(Xd, Ytr_perm, folds, Lpr_perm, meta, seed, log, n_perm=20):
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(s19._fold_check)(seed, Xd, tr, pr, Ytr_perm[k][:n_perm])
                                  for k, (r, f, tr, pr, te, ev) in enumerate(folds))
    n_tot = n_unsure = n_bad = 0
    preds_b, preds_s = [], []
    for a, p, u, ra, rp in res:
        n_tot += len(a)
        n_unsure += int(u.sum())
        n_bad += int((~(np.isclose(a, ra) & np.isclose(p, rp, rtol=1e-7, atol=1e-10).all(axis=1)) & ~u).sum())
        pb = p.copy()
        pb[u] = rp[u]
        preds_b.append(pb)
        preds_s.append(rp)
    alpha_same = all(np.array_equal(a[~u], ra[~u]) for a, p, u, ra, rp in res)
    Lp = [L[:n_perm] for L in Lpr_perm]
    _, na = s19.analyse([np.vstack([x[:1], x]) for x in preds_b], [np.vstack([L[:1], L]) for L in Lp], meta, folds, 0.0, True)
    _, nb = s19.analyse([np.vstack([x[:1], x]) for x in preds_s], [np.vstack([L[:1], L]) for L in Lp], meta, folds, 0.0, True)
    ok_null = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'  確認5 置換{n_perm}回 × {len(folds)} fold(乱数{seed}): まとめて解いた alpha・予測値 = sklearn RidgeCV を1回ずつ '
        f'(あいまいでない {n_tot - n_unsure} 組のうち不一致 {n_bad}、alpha の一致 {alpha_same}、あいまいな組 {n_unsure})。'
        f'帰無分布(全体・W・B)が完全に一致 {ok_null}  [{time.time() - t:.0f}s]')
    assert n_bad == 0 and alpha_same and ok_null


# ---------------------------------------------------------------- M_diff(fold ごと)
def _fold_m(seed, X, tr, pr, Ytr):
    """M_diff: 123件の学習側で分類。行0 = 観測ラベル(sklearn)、行1.. = 置換(まとめて解き、あいまいな行は学習し直す)。"""
    Ftr, Fpr = s19._feats(seed, X, tr, pr)
    alpha, dec, unsure = fast._batched_ridge(Ftr, Fpr, Ytr)
    ref = RidgeNTrain().fit(Ftr, Ytr[0])
    rd = ref.decision_function(Fpr)
    ok = bool(np.isclose(alpha[0], ref.alpha_) and np.allclose(dec[0], rd, rtol=1e-7, atol=1e-10))
    alpha[0], dec[0] = ref.alpha_, rd
    redo = np.flatnonzero(unsure[1:]) + 1
    for i in redo:
        dec[i] = RidgeNTrain().fit(Ftr, Ytr[i]).decision_function(Fpr)
    return alpha[0], dec, ok, len(redo)


# ---------------------------------------------------------------- 図
def plot_overview(summary, s19sum):
    fig, ax = plt.subplots(figsize=(10, 5))
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内(主の判定)'), ('B', 'B 地域間')]
    sm = summary.set_index(['config', 'metric'])
    s9 = s19sum.set_index(['config', 'metric'])
    w = 0.34
    for k, c in enumerate(CFGS):
        xs = np.arange(len(mets)) + (k - 0.5) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.9, color=CFG_COLOR[c], alpha=0.9, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95']
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8.5, color=INK)
            if (c, m) in s9.index:
                ax.plot(x + w * 0.3, s9.loc[(c, m), 'mean'], marker='o', ms=7, mfc='none', mec=INK, mew=1.3, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点(段階20)')
    ax.plot([], [], marker='o', ms=7, mfc='none', mec=INK, ls='', label='段階19の値(参考)')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(0.45, max(0.72, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('段階20: 新しい乱数と分割での再確認(label_muniyear_median の123件で評価)', fontsize=12, color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stage20_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-perm-m', type=int, default=200, help='M_diff の置換の回数(記述用)')
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = KERNEL_SEEDS[:args.n_seeds]
    P, Pm = args.n_perm, args.n_perm_m
    if args.n_seeds < 10 and Pm > P:
        Pm = P  # 試運転では M_diff の置換も減らす
    s6.N_JOBS = N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  カーネルの乱数 {seeds}  M_diff の置換 {Pm}  並列数 {N_JOBS}')
    lab, X, meta = load_data(log)
    X123, meta123 = s14.load()
    y123 = meta123.y.values

    log('\n[起動時の確認]')
    i123, in123 = s19.check_counts(lab, meta, X, meta123, X123, log)  # 確認1(段階19と同じ想定の件数)
    sp, folds = build_folds(meta, i123, meta123, log)
    split_table(sp, y123, meta123.muniCd.values).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False,
                                                        encoding='utf-8-sig')

    # ---- 置換(R_all: 収量を集団の中で並べ替え / M_diff: ラベルを市町村×年の中で並べ替え)
    relabel = s19.Relabel(lab)
    yld, my_all = lab['yield'].values, lab.muniyear.to_numpy(dtype=str)
    Yp = np.array([permute_within(yld, my_all, np.random.default_rng([PERM_SEED, i])) for i in range(P)]).reshape(P, -1)
    s19.check_relabel(lab, relabel, meta, i123, meta123, Yp, log)  # 確認6
    gm, lpos = lab.group_mean_yield.values, meta.lab_pos.values
    Yrel_p = np.log(Yp[:, lpos] / gm[lpos])
    Lab_p = np.array([relabel(v) for v in Yp])[:, lpos]
    L_obs = relabel(yld)[lpos]
    yrel = meta.y_rel.values
    yrank = np.zeros(len(meta))  # s19._fold_all の行1(使わない)
    evm_mask = meta.muniCd.isin(set(meta123.muniCd)).values
    ntc = pd.DataFrame({'n_test': [int((~np.isnan(L) & evm_mask).sum()) for L in Lab_p],
                        'n_high': [int(((L == 1) & evm_mask).sum()) for L in Lab_p],
                        'n_low': [int(((L == 0) & evm_mask).sum()) for L in Lab_p],
                        'n_added': [int((~np.isnan(L) & evm_mask & ~in123).sum()) for L in Lab_p],
                        'n_dropped': [int((np.isnan(L) & in123).sum()) for L in Lab_p]}).rename_axis('perm')
    ntc.to_csv(os.path.join(OUT_DIR, 'null_test_counts.csv'), encoding='utf-8-sig')
    log('  置換ごとのテスト件の数: ' + '、'.join(f'{k}件 {v}回' for k, v in ntc.n_test.value_counts().sort_index().items()))
    Ytr_rows = [np.vstack([yrel[tr], yrank[tr], Yrel_p[:, tr]]) for r, f, tr, pr, te, ev in folds]
    Lpr_rows = [np.vstack([L_obs[pr], Lab_p[:, pr]]) for r, f, tr, pr, te, ev in folds]
    for (r, f, tr, pr, te, ev), L in zip(folds, Lpr_rows):
        assert np.array_equal(np.flatnonzero(~np.isnan(L[0])), np.sort(ev)) and np.array_equal(L[0][ev], y123[te])
    Ym = np.array([permute_within(y123, meta123.muniyear.to_numpy(dtype=str), np.random.default_rng([PERM_SEED, i]))
                   for i in range(Pm)]).reshape(Pm, -1)
    Ym_all = np.vstack([y123, Ym])  # (1 + Pm, 123)
    Ytr_m = [Ym_all[:, tr13] for r, f, tr13, te13 in sp]
    Lpr_m = []
    for (r, f, tr, pr, te, ev) in folds:
        L = np.full((1 + Pm, len(pr)), np.nan)
        L[:, ev] = Ym_all[:, te]
        Lpr_m.append(L)

    Xd = first_diff(X)
    precheck_pipeline(Xd, yrel, folds, y123, seeds[0], log)
    precheck_batched(Xd, [Y[2:] for Y in Ytr_rows], folds, [L[1:] for L in Lpr_rows], meta, seeds[0], log,
                     n_perm=min(20, P))

    # ---- 本番
    out = {c: {'folds': [], 'pair': [], 'seed': [], 'scores': [], 'spear': []} for c in CFGS}
    S = {c: np.full((len(seeds), N_REPEATS, len(meta123)), np.nan) for c in CFGS}
    nulls = {c: {'all': [], 'W': [], 'B': []} for c in CFGS}
    for k, seed in enumerate(seeds):
        t1 = time.time()
        ra = Parallel(n_jobs=N_JOBS)(delayed(s19._fold_all)(seed, Xd, tr, pr, Ytr_rows[j])
                                     for j, (r, f, tr, pr, te, ev) in enumerate(folds))
        assert all(x[2] for x in ra), 'R_all: 観測の目的変数で、まとめて解いた結果が sklearn と一致しない fold がある'
        rm = Parallel(n_jobs=N_JOBS)(delayed(_fold_m)(seed, Xd, i123[tr13], pr, Ytr_m[j])
                                     for j, ((r, f, tr13, te13), (_, _, _, pr, _, _)) in enumerate(zip(sp, folds)))
        assert all(x[2] for x in rm), 'M_diff: 観測ラベルで、まとめて解いた結果が sklearn と一致しない fold がある'
        res = {'R_all': ([a[0] for a, *_ in ra], [np.vstack([p[:1], p[2:]]) for _, p, *_ in ra], Lpr_rows,
                         sum(x[3] for x in ra), [len(x[2]) for x in folds]),
               'M_diff': ([x[0] for x in rm], [x[1] for x in rm], Lpr_m, sum(x[3] for x in rm),
                          [len(x[2]) for x in sp])}
        for c in CFGS:
            alphas, preds, Lpr, n_redo, ntr = res[c]
            obs, null = s19.analyse(preds, Lpr, meta, folds, 0.0, True)
            s19.check_obs(f'{c} 乱数{seed}', obs, preds, y123, folds, 0.0, log)  # 確認7
            for kk in nulls[c]:
                nulls[c][kk].append(null[kk])
            al = np.array(alphas) / np.array(ntr)
            fd = pd.DataFrame({'config': c, 'seed': seed, 'repeat': [x[0] for x in folds], 'fold': [x[1] for x in folds],
                               'n_train': ntr, 'n_test': [len(x[4]) for x in folds], 'alpha_over_n': al,
                               'auc': obs['fold_auc'], 'bacc': obs['fold_bacc']})
            out[c]['folds'].append(fd)
            sc_all = np.full((N_REPEATS, len(meta)), np.nan)
            for (r, f, tr, pr, te, ev), p in zip(folds, preds):
                sc_all[r, pr] = p[0]
                S[c][k, r, te] = p[0][ev]
                mm = meta123.iloc[te]
                out[c]['scores'].append(pd.DataFrame({
                    'config': c, 'seed': seed, 'repeat': r, 'fold': f, 'field_id': mm.field_id.values,
                    'year': mm.year.values, 'muniCd': mm.muniCd.values, 'muniyear': mm.muniyear.values,
                    'y': mm.y.values, 'y_rel': yrel[i123[te]], 'score': p[0][ev]}))
            spd = pd.DataFrame([{'config': c, 'seed': seed, 'repeat': r, 'muniyear': g, 'n': n, 'rho': rho}
                                for r in range(N_REPEATS) for g, n, rho in s19.within_spearman(sc_all[r], meta, evm_mask)])
            out[c]['spear'].append(spd)
            wm = spd.dropna().groupby('repeat').apply(lambda d: np.average(d.rho, weights=d.n), include_groups=False)
            row = {'config': c, 'seed': seed, 'all_auc': fd.auc.mean(), 'all_bacc': fd.bacc.mean(),
                   'alpha_at_max': np.isclose(al, BASE_ALPHAS[-1]).mean(),
                   'alpha_at_min': np.isclose(al, BASE_ALPHAS[0]).mean(), 'spearman_w': wm.mean(), 'n_redo': n_redo}
            for t in TYPES:
                a, n = obs[t]
                row[t] = np.nanmean(a)
                out[c]['pair'].append(pd.DataFrame({'config': c, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                                    'auc': a, 'n_pairs': n}))
            out[c]['seed'].append(row)
            log(f'    {c:6s} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
                + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES)
                + f'  順位相関 {row["spearman_w"]:+.3f}  alpha 上限 {row["alpha_at_max"]:.2f}(学習し直した組 {n_redo})')
        log(f'  乱数{seed} 終了 [{time.time() - t1:.0f}s]')

    # ---- まとめ
    nmean = {c: {kk: np.mean(np.stack(v), axis=0) for kk, v in nulls[c].items()} for c in CFGS}
    np.save(os.path.join(OUT_DIR, 'null_R_all_all.npy'), nmean['R_all']['all'])
    np.save(os.path.join(OUT_DIR, 'null_R_all_W.npy'), nmean['R_all']['W'])
    np.save(os.path.join(OUT_DIR, 'null_R_all_B.npy'), nmean['R_all']['B'])
    np.save(os.path.join(OUT_DIR, 'null_M_diff_all.npy'), nmean['M_diff']['all'])
    np.save(os.path.join(OUT_DIR, 'null_M_diff_W.npy'), nmean['M_diff']['W'])
    np.save(os.path.join(OUT_DIR, 'null_M_diff_B.npy'), nmean['M_diff']['B'])
    folds_df = pd.concat([f for c in CFGS for f in out[c]['folds']])
    per_seed = pd.DataFrame([r for c in CFGS for r in out[c]['seed']])
    pair_auc = pd.concat([p for c in CFGS for p in out[c]['pair']])
    spear = pd.concat([s for c in CFGS for s in out[c]['spear']])
    summary = []
    for c in CFGS:
        ps = per_seed[per_seed.config == c]
        fd = folds_df[folds_df.config == c]
        npm = P if c == 'R_all' else Pm
        base = {'config': c, 'n_eval': len(meta123), 'n_seeds': len(ps), 'n_perm': npm,
                'alpha_at_max': ps.alpha_at_max.mean(), 'alpha_at_min': ps.alpha_at_min.mean(),
                'n_train_mean': fd.groupby(['repeat', 'fold']).n_train.first().mean()}
        nl_map = {'all_auc': nmean[c]['all'][:, 1], 'all_bacc': nmean[c]['all'][:, 0], 'W': nmean[c]['W'],
                  'B': nmean[c]['B']}
        for met in ['all_auc', 'all_bacc', 'W', 'B', 'X', 'spearman_w']:
            v = ps[met]
            r = {**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max()}
            if met in TYPES:
                r['n_pairs_per_repeat'] = pair_auc[(pair_auc.config == c) & (pair_auc.type == met)] \
                    .groupby('repeat').n_pairs.first().mean()
            if met in nl_map:
                nl = nl_map[met]
                r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                          'p': (1 + np.sum(nl >= v.mean())) / (1 + npm)})
            summary.append(r)
    summary = pd.DataFrame(summary)
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds_df.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat([s for c in CFGS for s in out[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                   index=False, encoding='utf-8-sig')
    pair_auc.to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    sg = spear.groupby(['config', 'muniyear']).agg(n=('n', 'first'), rho_mean=('rho', 'mean'), rho_sd=('rho', 'std'),
                                                  n_nan=('rho', lambda s: int(s.isna().sum()))).reset_index()
    tot = per_seed.groupby('config').spearman_w.agg(['mean', 'min', 'max']).reset_index() \
        .rename(columns={'mean': 'rho_mean', 'min': 'rho_seed_min', 'max': 'rho_seed_max'})
    tot['muniyear'] = '(全体: 件数で重み付けした平均)'
    tot['n'] = sg.groupby('config').n.sum().reindex(tot.config).values
    pd.concat([tot, sg]).to_csv(os.path.join(OUT_DIR, 'within_group_spearman.csv'), index=False, encoding='utf-8-sig')

    # 対ごとの差
    fc = {c: folds_df[folds_df.config == c][['seed', 'repeat', 'fold', 'auc', 'bacc']] for c in CFGS}
    pds = s7.paired_summary(s7.paired(fc['R_all'], fc['M_diff'], 'R_all − M_diff'))
    pds['positive_ci'] = pds.rep_q025 > 0
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差 R_all − M_diff(同じテスト件・同じ乱数)]')
    for _, r in pds.iterrows():
        log(f'  {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{r.rep_q025:+.3f}, {r.rep_q975:+.3f}]、'
            f'正の割合 {r.rep_frac_pos:.2f}')
    boot, pt = s16.w_diff_bootstrap(S['R_all'], S['M_diff'], meta123, sp, args.n_boot)
    boot = boot.rename(columns={'W_GC_date': 'W_R_all', 'W_M_diff': 'W_M_diff'})
    boot.to_csv(os.path.join(OUT_DIR, 'w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    bq = boot.W_diff.quantile([.025, .975]).values
    log(f'  W の差(市町村のブートストラップ {args.n_boot}回): 点推定 {pt["a"] - pt["b"]:+.3f}({pt["a"]:.3f} − '
        f'{pt["b"]:.3f})、95%区間 [{bq[0]:+.3f}, {bq[1]:+.3f}]、正の割合 {(boot.W_diff > 0).mean():.3f}')

    # 段階19との比較
    s19sum = pd.read_csv(os.path.join(OUT19, 'summary.csv'), encoding='utf-8-sig')
    s19sum = s19sum[s19sum.config.isin(CFGS)]
    s19boot = pd.read_csv(os.path.join(OUT19, 'w_diff_bootstrap.csv'), encoding='utf-8-sig')
    s19boot = s19boot[s19boot.comparison == 'R_all − M_diff']
    cmp_rows = []
    for c in CFGS:
        for met in ['W', 'all_auc', 'B', 'X', 'spearman_w', 'all_bacc']:
            a = s19sum[(s19sum.config == c) & (s19sum.metric == met)].iloc[0]
            b = summary[(summary.config == c) & (summary.metric == met)].iloc[0]
            cmp_rows.append({'config': c, 'metric': met, 'stage19_mean': a['mean'], 'stage19_null_q95': a.get('null_q95'),
                             'stage19_p': a.get('p'), 'stage20_mean': b['mean'], 'stage20_null_q95': b.get('null_q95'),
                             'stage20_p': b.get('p'), 'diff_20_minus_19': b['mean'] - a['mean'],
                             'stage19_alpha_at_max': a.alpha_at_max, 'stage20_alpha_at_max': b.alpha_at_max})
    for qq in (.025, .975):
        cmp_rows.append({'config': 'R_all − M_diff', 'metric': f'W_diff_boot_q{qq * 1000:03.0f}',
                         'stage19_mean': s19boot.W_diff.quantile(qq), 'stage20_mean': boot.W_diff.quantile(qq)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(OUT_DIR, 'compare_stage19.csv'), index=False, encoding='utf-8-sig')
    log('\n[段階19との比較]')
    log(cmp.drop(columns=['stage19_alpha_at_max', 'stage20_alpha_at_max']).round(3).to_string(index=False))

    plot_overview(summary, s19sum)
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'n_perm', 'alpha_at_max', 'n_train_mean']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.config == 'R_all') & (summary.metric == 'W')].iloc[0]
    hit = bool(r['mean'] > r.null_q95)
    log(f'\n[主の判定] R_all の W {r["mean"]:.3f}(乱数 {r.seed_min:.3f}〜{r.seed_max:.3f})、W の帰無分布の95%点 '
        f'{r.null_q95:.3f}、p={r.p:.3f} → ' + (
            '上回る: 新しい乱数と分割でも、相対収量の回帰で学習したモデルは、同じ地域・年の中の多収・低収を偶然以上に見分けた'
            '(同じデータの中での再確認)' if hit else
            '上回らない: 段階19の W の結果は、乱数と分割を変えると再現しなかった。段階19の W は偶然の可能性が残る'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
