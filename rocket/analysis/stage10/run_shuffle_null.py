"""
段階10の追加: 時間の順番の置換検定(シャッフルの分布の中での D の位置)(EXP-019)。

カーネルの乱数 k = 0, 1, 2 を固定し、1次差分の日の順番をシャッフルした入力で、シャッフル s = 0〜199 の
AUC・bacc(20回の繰り返し平均)を求める。ラベルの置換検定は行わない。
  S2a 系列ごとに別々: default_rng([1, s]) から82系列を行の順に permutation(59)(run_stage10.shuffle_each)
  S2b 全系列で同じ:   default_rng([2, s]).permutation(59)(run_stage10.shuffle_common)
シャッフルの乱数 s はカーネルの乱数 k と独立(どの k でも同じ200通りのシャッフルを使う)。
段階10の S2 は s = k の組なので、(k, s) = (0, 0), (1, 1), (2, 2) は段階10の乱数ごとの値と一致するはず(起動時に確かめる)。

統計量(事前に決めたもの): 片側の p = (1 + #{シャッフルの AUC >= D の AUC}) / (1 + 200)。
D は同じカーネルの乱数での本当の並びの AUC(段階6の保存済みの per_seed.csv)。k ごと、S2a・S2b ごとに求める。
判断: 3通りの k すべてで p < 0.05 → 時間の並び方に意味がある。それ以外 → 効果は確認できなかった。
bacc の p も同じ式で出す(参考)。

データ・分割・モデル・alpha は段階10の S2 と同じ。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage10\\run_shuffle_null.py
  試運転: ... run_shuffle_null.py --n-shuffles 5
出力: rocket/outputs/stage10/shuffle_null/
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

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_DIR)
import run_stage10 as s10  # noqa: E402
from run_stage10 import ROCKET_DIR, fast, s6, s7  # noqa: E402
from src.evaluation import make_splits, summarize  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage10', 'shuffle_null')
S10_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage10')
KERNEL_SEEDS = [0, 1, 2]
VARIANTS = {'S2a': ('S2a_shuffle_each', s10.shuffle_each, '系列ごとに別々'),
            'S2b': ('S2b_shuffle_common', s10.shuffle_common, '全系列で同じ')}
ALPHA_P = 0.05
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def one(variant, k, s, Xd, y, sp):
    """カーネルの乱数 k・シャッフル s での、20回の繰り返し平均の AUC・bacc(段階10の S2 と同じ計算)。"""
    Xs, _ = VARIANTS[variant][1](Xd, s)
    feats = [s6._one_fold('minirocket', k, Xs, tr, te) for r, f, tr, te in sp]
    f = s6.eval_cached(feats, y, sp)
    sm = summarize(f)
    return {'variant': variant, 'kernel_seed': k, 'shuffle_seed': s, 'auc': sm['auc_mean'], 'bacc': sm['bacc_mean'],
            'alpha_at_max': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[-1]).mean(), 'n_auc_nan': sm['n_auc_nan']}


def plot(res, D, pv):
    fig, axs = plt.subplots(len(VARIANTS), len(KERNEL_SEEDS), figsize=(14, 7.5), sharex=True)
    bins = np.linspace(0.40, 0.75, 36)
    for i, (v, (_, _, ja)) in enumerate(VARIANTS.items()):
        for j, k in enumerate(KERNEL_SEEDS):
            ax = axs[i, j]
            a = res[(res.variant == v) & (res.kernel_seed == k)].auc
            ax.hist(a, bins=bins, color='#eb6834' if v == 'S2a' else '#9b59b6', alpha=0.6)
            d = D.loc[k, 'auc_mean']
            r = pv[(pv.variant == v) & (pv.kernel_seed == k)].iloc[0]
            ax.axvline(d, color=INK, lw=2, label=f'D(本当の並び){d:.3f}')
            ax.axvline(a.quantile(0.95), color=MUTED, lw=1.2, ls='--', label=f'シャッフルの95%点 {a.quantile(0.95):.3f}')
            ax.axvline(0.5, color=MUTED, lw=0.6)
            ax.set_title(f'{v}({ja})・カーネルの乱数 {k}: p = {r.p_auc:.3f}(D 以上 {int(r.n_ge_auc)}/{len(a)})',
                         fontsize=9.5, color=INK, loc='left')
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            s6.style(ax)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel('AUC(20回の繰り返し平均)', color=INK)
            if j == 0:
                ax.set_ylabel('シャッフルの数', color=INK)
    fig.suptitle(f'時間の順番の置換検定: 1次差分の日の順番をシャッフルした AUC の分布(各 {len(a)} 通り)と D の位置',
                 fontsize=12.5, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'shuffle_null_hist.png'), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-shuffles', type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    Xm, mm = s6.s5.datasets()['main']
    y = mm.y.values
    Xd = first_diff(Xm)
    sp = make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    assert s7.check_same_splits(y, mm), '段階6の実験Dと分割が違う'
    _, D_seed, _, _, _ = s7.load_stage6_D()
    D = D_seed.set_index('seed').loc[KERNEL_SEEDS, ['auc_mean', 'bacc_mean']]
    log(f'並列数 {fast.N_JOBS}、カーネルの乱数 {KERNEL_SEEDS}、シャッフル 0〜{args.n_shuffles - 1}')
    log('段階6の実験Dと同じ分割: True')
    log('D(段階6の保存済み、カーネルの乱数ごと):')
    log(D.round(4).to_string())

    tasks = [(v, k, s) for v in VARIANTS for k in KERNEL_SEEDS for s in range(args.n_shuffles)]
    res = pd.DataFrame(Parallel(n_jobs=fast.N_JOBS, verbose=0)(delayed(one)(v, k, s, Xd, y, sp) for v, k, s in tasks))
    res.to_csv(os.path.join(OUT_DIR, 'shuffle_scores.csv'), index=False, encoding='utf-8-sig')
    log(f'計算 {time.time() - t0:.0f} 秒({len(tasks)} 通り)')

    # 段階10の S2(s = k の組)との一致
    s10_seed = pd.read_csv(os.path.join(S10_DIR, 'per_seed.csv'), encoding='utf-8-sig')
    ok = True
    for v, (cfg, _, _) in VARIANTS.items():
        for k in KERNEL_SEEDS:
            if k >= args.n_shuffles:
                continue
            a = res[(res.variant == v) & (res.kernel_seed == k) & (res.shuffle_seed == k)].iloc[0]
            b = s10_seed[(s10_seed.config == cfg) & (s10_seed.seed == k)].iloc[0]
            same = np.isclose(a.auc, b.auc_mean, rtol=0, atol=1e-12) and np.isclose(a.bacc, b.bacc_mean, rtol=0, atol=1e-12)
            ok &= bool(same)
            log(f'  段階10との一致 {v} k={k} s={k}: AUC {a.auc:.4f} / 段階10 {b.auc_mean:.4f}  {same}')
    log(f'段階10の S2(s = k)と一致: {ok}')
    assert ok

    rows = []
    for v in VARIANTS:
        for k in KERNEL_SEEDS:
            r = res[(res.variant == v) & (res.kernel_seed == k)]
            n = len(r)
            row = {'variant': v, 'kernel_seed': k, 'n_shuffles': n}
            for m in ['auc', 'bacc']:
                d = D.loc[k, f'{m}_mean']
                n_ge = int((r[m] >= d).sum())
                row.update({f'D_{m}': d, f'shuffle_{m}_mean': r[m].mean(), f'shuffle_{m}_sd': r[m].std(),
                            f'shuffle_{m}_q05': r[m].quantile(0.05), f'shuffle_{m}_median': r[m].median(),
                            f'shuffle_{m}_q95': r[m].quantile(0.95), f'shuffle_{m}_max': r[m].max(),
                            f'n_ge_{m}': n_ge, f'p_{m}': (1 + n_ge) / (1 + n)})
            row['alpha_at_max_mean'] = r.alpha_at_max.mean()
            rows.append(row)
    pv = pd.DataFrame(rows)
    pv.to_csv(os.path.join(OUT_DIR, 'shuffle_pvalues.csv'), index=False, encoding='utf-8-sig')

    # カーネルの乱数の間での、シャッフルごとの AUC の相関(同じシャッフルを使っているので)
    corr = {v: res[res.variant == v].pivot(index='shuffle_seed', columns='kernel_seed', values='auc').corr()
            for v in VARIANTS}
    for v, c in corr.items():
        c.to_csv(os.path.join(OUT_DIR, f'kernel_corr_{v}.csv'), encoding='utf-8-sig')

    plot(res, D, pv)

    log('\n[結果]')
    show = ['variant', 'kernel_seed', 'n_shuffles', 'D_auc', 'shuffle_auc_mean', 'shuffle_auc_sd', 'shuffle_auc_q95',
            'shuffle_auc_max', 'n_ge_auc', 'p_auc', 'D_bacc', 'shuffle_bacc_mean', 'n_ge_bacc', 'p_bacc',
            'alpha_at_max_mean']
    log(pv[show].round(4).to_string(index=False))
    for v in VARIANTS:
        p = pv[pv.variant == v].p_auc
        verdict = '時間の並び方に意味がある' if (p < ALPHA_P).all() else '時間の並び方の効果は確認できなかった'
        log(f'判断 {v}: p = {", ".join(f"{x:.3f}" for x in p)} → {verdict}')
    for v, c in corr.items():
        log(f'\nカーネルの乱数の間の相関({v}、シャッフルごとの AUC):\n{c.round(3).to_string()}')
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
