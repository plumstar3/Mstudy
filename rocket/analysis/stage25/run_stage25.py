"""
段階25: 時間の順番の置換検定を label_muniyear_median で行う(EXP-034、EXP-019 の再実施)。

段階13の median(123件、field_year_labeled.csv の flower_date、開花起点 0〜+59 日)の1次差分(59点)の
日の順番をシャッフルした入力で、カーネルの乱数 k = 0, 1, 2 ごとに、シャッフル s = 0〜199 の AUC を求める。
ラベルの置換検定はしない(EXP-019 と同じ)。シャッフルは1次差分を取った後にかける。
  S2a 系列ごと:        default_rng([1, s]) から123系列を行の順に permutation(59)(段階10の shuffle_each)
  S2b 全系列で同じ:    default_rng([2, s]).permutation(59)(段階10の shuffle_common)
  S3  5日のかたまり(参考): 59点を5日ずつ12のかたまり(最後は4点)に分け、default_rng([3, s]) から
                       系列ごとに permutation(12) でかたまりの順番だけを並べ替える(かたまりの中はそのまま)
シャッフルの乱数 s はカーネルの乱数 k と独立(どの k でも同じ200通り)。
カーネルの乱数は k = 0 だけで本番を行う(EXP-019 では k の間の AUC の相関が 0.995〜0.997 だったため)。
確認として k = 1 で s = 0〜19(各種類20通り)を計算し、同じ s の k = 0 との AUC の相関を出す。0.99 未満なら止める。

モデル・分割は段階13と同じ: MiniRocket(10000)→ StandardScaler(with_mean=False)→ RidgeNTrain、
make_splits(y, muniCd, 5, 20, 0)。
統計量(事前に固定): 全体の AUC(fold の AUC の平均)。p = (1 + #{シャッフルの AUC >= D}) / (1 + 200)。
主の判定: S2a・S2b のそれぞれで、k = 0 の p < 0.05 なら「時間の並び方に意味がある」。
EXP-019 は k = 0・1・2 で判定していたので、比べるときは EXP-019 の k = 0 と並べる。
S3 は参考で判定に使わない。D は段階13の median・M の k = 0・1・2(計算し直して per_seed と一致を確かめる)。
副(記述): W・B(段階12の数え方)、bacc、alpha の張り付き、カーネルの乱数間の相関、EXP-019 との比較。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage25\\run_stage25.py
  試運転: ... run_stage25.py --n-shuffles 5 --kernel-seeds 0
出力: rocket/outputs/stage25/
"""
import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage13'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage10'))
import run_stage13 as s13  # noqa: E402
import run_stage10 as s10  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

s6, s12 = s13.s6, s13.s12
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage25')
DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13', 'label_muniyear_median')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
EXP019 = os.path.join(ROCKET_DIR, 'outputs', 'stage10', 'shuffle_null', 'shuffle_pvalues.csv')
N_EXPECT = 123
BLOCK = 5
ALPHA_P = 0.05
VARIANTS = {'S2a': '系列ごと', 'S2b': '全系列で同じ', 'S3': '5日のかたまり(参考)'}
COLORS = {'S2a': '#eb6834', 'S2b': '#9b59b6', 'S3': '#3a9a5b'}
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- シャッフル
def blocks(n=59, size=BLOCK):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def shuffle_blocks(D, s):
    """系列ごとに、かたまりの順番だけを並べ替える。戻り値: シャッフルした系列と、系列ごとの位置の並べ替え。"""
    rng = np.random.default_rng([3, s])
    bl = blocks(D.shape[-1])
    perms = np.array([np.concatenate([bl[j] for j in rng.permutation(len(bl))]) for _ in range(len(D))])
    return np.take_along_axis(D, perms[:, None, :], axis=-1), perms


def shuffle(variant, D, s):
    if variant == 'S2a':
        return s10.shuffle_each(D, s)
    if variant == 'S2b':
        return s10.shuffle_common(D, s)
    if variant == 'S3':
        return shuffle_blocks(D, s)
    if variant == 'D':
        return D, np.arange(D.shape[-1])
    raise ValueError(variant)


# ---------------------------------------------------------------- 1つの (種類, k, s)
def one(variant, k, s, Xd, y, meta, sp):
    """全体の AUC・bacc(fold の平均)、W・B(繰り返しごとに fold をまたいで合算し、20回で平均)、alpha の張り付き。"""
    Xs, _ = shuffle(variant, Xd, s)
    aucs, baccs, at_max, counts = [], [], [], []
    for r, f, tr, te in sp:
        Ftr, Fte = s6._one_fold('minirocket', k, Xs, tr, te)
        clf = s6.RidgeNTrain().fit(Ftr, y[tr])
        dec = clf.decision_function(Fte)
        aucs.append(roc_auc_score(y[te], dec) if len(np.unique(y[te])) == 2 else np.nan)
        baccs.append(balanced_accuracy_score(y[te], clf.predict(Fte)))
        at_max.append(np.isclose(clf.alpha_ / len(tr), s6.BASE_ALPHAS[-1]))
        counts.append(s12.pair_counts(dec[None, :], y[te][None, :], s12.pair_masks(meta, te)))
    out = {'variant': variant, 'kernel_seed': k, 'shuffle_seed': s, 'auc': np.nanmean(aucs), 'bacc': np.mean(baccs),
           'alpha_at_max': np.mean(at_max), 'n_auc_nan': int(np.isnan(aucs).sum())}
    for t in ['W', 'B', 'X']:
        a, _ = s12.repeat_auc(counts, sp, t)
        out[t] = np.nanmean(a[:, 0])
    return out


# ---------------------------------------------------------------- 起動時の確認
def prechecks(Xd, y, meta, sp, log):
    ok_all = True
    # 4. 値の集まりが同じ
    for v in ['S2a', 'S2b', 'S3']:
        for s in [0, 1, 199]:
            Xs, _ = shuffle(v, Xd, s)
            ok = np.array_equal(np.sort(Xs, axis=-1), np.sort(Xd, axis=-1))
            ok_all &= ok
        log(f'  確認4 {v}: シャッフルの後も各系列の値の集まりが同じ(s = 0, 1, 199): {ok}')
    # 5. S3 のかたまりの中の順番
    Xs, perms = shuffle_blocks(Xd, 0)
    p0 = perms[0]
    bl = blocks(Xd.shape[-1])
    starts = np.flatnonzero(np.isin(p0, [b[0] for b in bl]))
    pieces = [p0[a:b] for a, b in zip(starts, list(starts[1:]) + [len(p0)])]
    ok5 = all(np.array_equal(pc, np.arange(pc[0], pc[0] + len(pc))) for pc in pieces) and len(pieces) == len(bl) \
        and np.array_equal(Xs[0, 0], Xd[0, 0, p0])
    ok_all &= ok5
    log(f'  確認5 S3(s = 0、1系列目)のかたまりの並び: ' + ' | '.join(f'{pc[0]}〜{pc[-1]}' for pc in pieces))
    log(f'         値: 元の位置 {pieces[0][0]}〜{pieces[0][-1]} = {np.round(Xd[0, 0, pieces[0]], 4).tolist()}、'
        f'シャッフル後の先頭 {len(pieces[0])} 点 = {np.round(Xs[0, 0, :len(pieces[0])], 4).tolist()}')
    log(f'  確認5 かたまりの中の順番が保たれている(全12かたまり): {ok5}')
    # 6. Pipeline と特徴量の使い回し(S2a、k = 0、s = 0、先頭10 fold)
    Xs, _ = shuffle('S2a', Xd, 0)
    a = evaluate(s6.pipeline('minirocket', 0), Xs, y, sp[:10])
    b = s6.eval_cached([s6._one_fold('minirocket', 0, Xs, tr, te) for r, f, tr, te in sp[:10]], y, sp[:10])
    ok6 = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    ok_all &= ok6
    log(f'  確認6 Pipeline と特徴量の使い回しの一致(S2a、k = 0、s = 0、先頭10 fold): {ok6}')
    assert ok_all, '確認4〜6のどれかが一致しない'


# ---------------------------------------------------------------- 図
def plot(res, D, pv, kernel_seeds, n_sh):
    vs = list(VARIANTS)
    fig, axs = plt.subplots(len(vs), len(kernel_seeds), figsize=(4.8 * len(kernel_seeds), 3.2 * len(vs)),
                            sharex=True, squeeze=False)
    lo = min(res.auc.min(), D.auc.min()) - 0.02
    hi = max(res.auc.max(), D.auc.max()) + 0.02
    bins = np.linspace(lo, hi, 36)
    for i, v in enumerate(vs):
        for j, k in enumerate(kernel_seeds):
            ax = axs[i, j]
            a = res[(res.variant == v) & (res.kernel_seed == k)].auc
            ax.hist(a, bins=bins, color=COLORS[v], alpha=0.6)
            d = D.loc[k, 'auc']
            r = pv[(pv.variant == v) & (pv.kernel_seed == k)].iloc[0]
            ax.axvline(d, color=INK, lw=2, label=f'D(本当の並び){d:.3f}')
            ax.axvline(a.quantile(0.95), color=MUTED, lw=1.2, ls='--', label=f'95%点 {a.quantile(0.95):.3f}')
            ax.axvline(0.5, color=MUTED, lw=0.6)
            ax.set_title(f'{v}({VARIANTS[v]})・k = {k}: p = {r.p_auc:.3f}(D 以上 {int(r.n_ge_auc)}/{len(a)})',
                         fontsize=9.5, color=INK, loc='left')
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            s6.style(ax)
            if i == len(vs) - 1:
                ax.set_xlabel('AUC(fold の平均)', color=INK)
            if j == 0:
                ax.set_ylabel('シャッフルの数', color=INK)
    fig.suptitle(f'段階25: 1次差分の日の順番をシャッフルした AUC の分布\n'
                 f'(label_muniyear_median 123件、各 {n_sh} 通り)と D の位置', fontsize=11, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'shuffle_null_hist.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-shuffles', type=int, default=200)
    ap.add_argument('--kernel-seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--check-k', type=int, default=1, help='相関の確認に使うカーネルの乱数')
    ap.add_argument('--check-n', type=int, default=20, help='相関の確認に使うシャッフルの数(s = 0〜n−1)')
    ap.add_argument('--plot-only', action='store_true', help='保存済みの CSV から図だけを描き直す')
    args = ap.parse_args()
    ks, n_sh = args.kernel_seeds, args.n_shuffles
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.plot_only:
        res = pd.read_csv(os.path.join(OUT_DIR, 'shuffle_scores.csv'), encoding='utf-8-sig')
        pv = pd.read_csv(os.path.join(OUT_DIR, 'shuffle_pvalues.csv'), encoding='utf-8-sig')
        D = pv.drop_duplicates('kernel_seed').set_index('kernel_seed')[['D_auc']].rename(columns={'D_auc': 'auc'})
        plot(res, D, pv, sorted(res.kernel_seed.unique()), int(res.groupby(['variant', 'kernel_seed']).size().max()))
        print('図を描き直した:', os.path.join(OUT_DIR, 'shuffle_null_hist.png'))
        return
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    n_jobs = s13.N_JOBS
    log(f'引数: {vars(args)}  並列数 {n_jobs}')

    # ---- データと分割
    X = np.load(DATA + '_X.npy')
    meta = pd.read_csv(DATA + '_meta.csv', encoding='utf-8-sig', dtype={'muniCd': str})
    y = meta.y.values
    Xd = first_diff(X)
    ok_n = len(meta) == N_EXPECT and Xd.shape[-1] == 59
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    ref = pd.read_csv(os.path.join(S13_DIR, 'splits.csv'), encoding='utf-8-sig')
    ref = ref[ref.label == 'median'].drop(columns='label').reset_index(drop=True)
    mine = split_table(sp, y, meta.muniCd.values)
    ok_sp = mine.shape == ref.shape and np.allclose(mine.values.astype(float), ref.values.astype(float))
    log(f'確認1 段階25・EXP-034 は未使用(analysis/stage25・outputs/stage25 は今回作成、docs に EXP-034 なし): '
        f'{"EXP-034" not in open(os.path.join(ROCKET_DIR, "docs", "experiments.md"), encoding="utf-8").read()}')
    log(f'確認2 件数 {len(meta)}(多収 {int(y.sum())} / 低収 {int((1 - y).sum())})・1次差分 {Xd.shape[-1]} 点: {ok_n}、'
        f'分割が段階13の median と同じ(splits.csv の件数・クラス・集団数): {ok_sp}')
    assert ok_n and ok_sp

    prechecks(Xd, y, meta, sp, log)

    # ---- D(段階13の k = 0, 1, 2)
    D = pd.DataFrame(Parallel(n_jobs=min(n_jobs, len(ks)))(delayed(one)('D', k, -1, Xd, y, meta, sp) for k in ks)) \
        .set_index('kernel_seed')
    ps13 = pd.read_csv(os.path.join(S13_DIR, 'per_seed.csv'), encoding='utf-8-sig')
    ps13 = ps13[(ps13.label == 'median') & (ps13.config == 'M')].set_index('seed')
    ok3 = True
    for k in ks:
        r = ps13.loc[float(k)]
        same = all(np.isclose(D.loc[k, a], r[b], rtol=0, atol=1e-12)
                   for a, b in [('auc', 'all_auc'), ('bacc', 'all_bacc'), ('W', 'W'), ('B', 'B'), ('X', 'X')])
        ok3 &= same
        log(f'  D k={k}: AUC {D.loc[k, "auc"]:.4f} / 段階13 {r.all_auc:.4f}、bacc {D.loc[k, "bacc"]:.4f} / '
            f'{r.all_bacc:.4f}、W {D.loc[k, "W"]:.4f} / {r.W:.4f}、B {D.loc[k, "B"]:.4f} / {r.B:.4f}  {same}')
    log(f'確認3 D が段階13の per_seed と一致(AUC・bacc・W・B・X): {ok3}')
    assert ok3

    # ---- シャッフル
    t1 = time.time()
    tasks = [(v, k, s) for v in VARIANTS for k in ks for s in range(n_sh)]
    res = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(v, k, s, Xd, y, meta, sp) for v, k, s in tasks))
    dt = time.time() - t1
    log(f'\nシャッフルの計算 {dt:.0f} 秒({len(tasks)} 通り、1通りあたり実時間 {dt / len(tasks):.2f} 秒・'
        f'並列 {n_jobs})。本番(3種類 × 200 × k 3通り = 1800 通り)の見積もり 約 {dt / len(tasks) * 1800 / 60:.0f} 分')
    res.to_csv(os.path.join(OUT_DIR, 'shuffle_scores.csv'), index=False, encoding='utf-8-sig')

    # ---- 使ったシャッフルの記録
    rec = []
    for s in range(n_sh):
        rec.append({'variant': 'S2b', 'shuffle_seed': s, 'rule': 'default_rng([2, s]).permutation(59) を全系列に',
                    'perm': ' '.join(map(str, s10.shuffle_common(Xd[:1], s)[1]))})
        rec.append({'variant': 'S2a', 'shuffle_seed': s,
                    'rule': 'rng = default_rng([1, s]); 123系列を行の順に rng.permutation(59)',
                    'perm': '1系列目: ' + ' '.join(map(str, s10.shuffle_each(Xd, s)[1][0]))})
        rec.append({'variant': 'S3', 'shuffle_seed': s,
                    'rule': 'rng = default_rng([3, s]); 123系列を行の順に rng.permutation(12) でかたまり(5日、最後は4点)を並べ替え',
                    'perm': '1系列目: ' + ' '.join(map(str, shuffle_blocks(Xd, s)[1][0]))})
    pd.DataFrame(rec).to_csv(os.path.join(OUT_DIR, 'shuffles_used.csv'), index=False, encoding='utf-8-sig')

    # ---- p 値
    rows = []
    for v in VARIANTS:
        for k in ks:
            r = res[(res.variant == v) & (res.kernel_seed == k)]
            n = len(r)
            row = {'variant': v, 'kernel_seed': k, 'n_shuffles': n}
            for m in ['auc', 'bacc']:
                d = D.loc[k, m]
                n_ge = int((r[m] >= d).sum())
                row.update({f'D_{m}': d, f'shuffle_{m}_mean': r[m].mean(), f'shuffle_{m}_sd': r[m].std(),
                            f'shuffle_{m}_q05': r[m].quantile(0.05), f'shuffle_{m}_median': r[m].median(),
                            f'shuffle_{m}_q95': r[m].quantile(0.95), f'shuffle_{m}_max': r[m].max(),
                            f'n_ge_{m}': n_ge, f'p_{m}': (1 + n_ge) / (1 + n)})
            row.update({'D_alpha_at_max': D.loc[k, 'alpha_at_max'], 'shuffle_alpha_at_max_mean': r.alpha_at_max.mean()})
            rows.append(row)
    pv = pd.DataFrame(rows)
    pv.to_csv(os.path.join(OUT_DIR, 'shuffle_pvalues.csv'), index=False, encoding='utf-8-sig')

    # ---- まとめ(偶然を上回った分の内訳、W・B、EXP-019 との比較)
    Dm = D.loc[ks, ['auc', 'bacc', 'W', 'B', 'X', 'alpha_at_max']].mean()
    summ = [{'variant': 'D', 'n_shuffles': np.nan, 'auc': Dm.auc, 'bacc': Dm.bacc, 'W': Dm.W, 'B': Dm.B, 'X': Dm.X,
             'alpha_at_max': Dm.alpha_at_max, 'share_above_chance': 1.0}]
    for v in VARIANTS:
        r = res[res.variant == v]
        m = r.groupby('kernel_seed')[['auc', 'bacc', 'W', 'B', 'X', 'alpha_at_max']].mean().mean()
        summ.append({'variant': v, 'n_shuffles': n_sh, **m.to_dict(),
                     'share_above_chance': (m.auc - 0.5) / (Dm.auc - 0.5),
                     'p_auc_by_k': ', '.join(f'{x:.3f}' for x in pv[pv.variant == v].p_auc)})
    summ = pd.DataFrame(summ)
    e19 = pd.read_csv(EXP019, encoding='utf-8-sig')
    cmp_rows = []
    for v in ['S2a', 'S2b']:
        for k in ks:
            a = pv[(pv.variant == v) & (pv.kernel_seed == k)].iloc[0]
            b = e19[(e19.variant == v) & (e19.kernel_seed == k)]
            b = b.iloc[0] if len(b) else None
            cmp_rows.append({'variant': v, 'kernel_seed': k,
                             'EXP019_D_auc': b.D_auc if b is not None else np.nan,
                             'EXP019_shuffle_mean': b.shuffle_auc_mean if b is not None else np.nan,
                             'EXP019_q95': b.shuffle_auc_q95 if b is not None else np.nan,
                             'EXP019_p': b.p_auc if b is not None else np.nan,
                             'now_D_auc': a.D_auc, 'now_shuffle_mean': a.shuffle_auc_mean,
                             'now_q95': a.shuffle_auc_q95, 'now_p': a.p_auc})
    cmp = pd.DataFrame(cmp_rows)
    with open(os.path.join(OUT_DIR, 'shuffle_summary.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        fh.write('# 偶然を上回った分の内訳と W・B(k ごとに平均して k で平均)\n')
        summ.to_csv(fh, index=False)
        fh.write('\n# EXP-019(label_muniyear、82件)との比較\n')
        cmp.to_csv(fh, index=False)

    plot(res, D, pv, ks, n_sh)

    # ---- カーネルの乱数間の相関の確認(k = check_k、s = 0〜check_n−1 を k = 0 と比べる)
    k0, kc, nc = ks[0], args.check_k, min(args.check_n, n_sh)
    t2 = time.time()
    chk = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(v, kc, s, Xd, y, meta, sp)
                                               for v in VARIANTS for s in range(nc)))
    chk.to_csv(os.path.join(OUT_DIR, 'kernel_check_scores.csv'), index=False, encoding='utf-8-sig')
    corr_rows = []
    for v in list(VARIANTS) + ['全種類']:
        a = res[(res.kernel_seed == k0) & (res.shuffle_seed < nc) & ((res.variant == v) if v != '全種類' else True)]
        b = chk[(chk.variant == v) if v != '全種類' else slice(None)]
        j = a.merge(b, on=['variant', 'shuffle_seed'], suffixes=('_a', '_b'))
        corr_rows.append({'variant': v, 'k_a': k0, 'k_b': kc, 'n_shuffles': len(j),
                          'corr_auc': np.corrcoef(j.auc_a, j.auc_b)[0, 1],
                          'mean_abs_diff_auc': (j.auc_a - j.auc_b).abs().mean(),
                          'max_abs_diff_auc': (j.auc_a - j.auc_b).abs().max()})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(OUT_DIR, 'kernel_corr.csv'), index=False, encoding='utf-8-sig')
    log(f'\n[カーネルの乱数間の相関の確認] k = {k0} と k = {kc}、s = 0〜{nc - 1}({time.time() - t2:.0f} 秒)')
    log(corr.round(4).to_string(index=False))
    ok_corr = bool((corr.corr_auc >= 0.99).all())
    log(f'確認7 相関がすべて 0.99 以上: {ok_corr}')

    # ---- ログ
    log('\n[p 値]')
    show = ['variant', 'kernel_seed', 'n_shuffles', 'D_auc', 'shuffle_auc_mean', 'shuffle_auc_sd', 'shuffle_auc_q95',
            'shuffle_auc_max', 'n_ge_auc', 'p_auc', 'D_bacc', 'shuffle_bacc_mean', 'p_bacc',
            'D_alpha_at_max', 'shuffle_alpha_at_max_mean']
    log(pv[show].round(4).to_string(index=False))
    log('\n[主の判定]')
    for v in ['S2a', 'S2b']:
        p = pv[pv.variant == v].p_auc
        full = ks == [0] and n_sh == 200
        verdict = ('時間の並び方に意味がある' if (p < ALPHA_P).all() else '時間の並び方の効果は確認できなかった')
        log(f'  {v}(k = {", ".join(map(str, ks))}): p = {", ".join(f"{x:.3f}" for x in p)} → {verdict}'
            + ('' if full else '(試運転)'))
    p = pv[pv.variant == 'S3'].p_auc
    log(f'  S3(参考、判定に使わない): p = {", ".join(f"{x:.3f}" for x in p)}')
    log('\n[偶然を上回った分の内訳と W・B]')
    log(summ.round(4).to_string(index=False))
    log('\n[EXP-019 との比較]')
    log(cmp.round(4).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()
    assert ok_corr, 'k の間の AUC の相関が 0.99 未満。止めて報告する'


if __name__ == '__main__':
    main()
