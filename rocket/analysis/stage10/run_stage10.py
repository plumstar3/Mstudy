"""
段階10: 1次差分の情報は「変化の大きさ」(a) か「時間の並び方」(b) か を切り分ける(EXP-018。探索的。主な設定は変えない)。

  S1  1次差分の要約統計(日の順番によらない8つ)→ StandardScaler → LogisticRegression(C=1, max_iter=1000)。置換1000回
  S2a 系列ごとに別々にシャッフルした1次差分 → MiniRocket(段階6の実験Dと同じ)。置換200回
  S2b すべての系列で同じシャッフルをした1次差分 → MiniRocket。置換200回
  D   段階6の実験D(MiniRocket・1次差分)。outputs/stage6/ の保存済みの結果を使い、計算し直さない

共通: main(82件、開花起点 0〜+59 日)の1次差分(59点)、label_muniyear、段階6と同じ分割
      (StratifiedGroupKFold(5, groups=muniCd)× 20 回、分割の乱数 0〜19)。
      置換は市町村×年の中で並べ替え、置換 i は乱数 [0, i](段階6と同じ)。
S1 の特徴量8「大きく増えた日の数」のしきい値は、fold の学習データの d すべての90%点(Pipeline の中で fold ごとに決める)。
S2 のシャッフル(ラベルを使わず、fold によらず1回だけ決める。乱数 k = MiniRocket の random_state と組):
  S2a: rng = default_rng([1, k])。82系列を行の順に、系列ごとに rng.permutation(59) で並べ替える
  S2b: rng = default_rng([2, k])。rng.permutation(59) を1通り決めて全系列に使う
置換検定は run_stage6_local.py の高速版。起動時に元の方法との一致を確かめる。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage10\\run_stage10.py
  試運転: ... run_stage10.py --n-seeds 1 --n-perm 20 --n-perm-s1 20
出力: rocket/outputs/stage10/
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage7'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
from src.evaluation import evaluate, make_splits, permute_within, split_table, summarize  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage10')
D_CONFIG = s7.D_CONFIG
INK, MUTED, GRID = s6.INK, s6.MUTED, s6.GRID
COLORS = {D_CONFIG: '#2a78d6', 'S1_diff_stats': '#1baf7a', 'S2a_shuffle_each': '#eb6834',
          'S2b_shuffle_common': '#9b59b6'}
LABELS = {D_CONFIG: 'D 1次差分\n(段階6)', 'S1_diff_stats': 'S1 要約統計\n+ LR',
          'S2a_shuffle_each': 'S2a 系列ごとに\nシャッフル', 'S2b_shuffle_common': 'S2b 全系列で同じ\nシャッフル'}
BL_NAMES = {'base_vwc_stats_main': 'VWC の要約統計', 'base_seed_doy_main': '播種日の通日',
            'base_irrigated_main': '灌水の有無'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

FEATURES = ['mean_d', 'sd_d', 'mean_abs_d', 'max_d', 'min_d', 'sum_pos_d', 'mean_neg_d', 'n_big_up']
FEATURE_JA = {'mean_d': 'd の平均', 'sd_d': 'd の標準偏差', 'mean_abs_d': '|d| の平均', 'max_d': 'd の最大値',
              'min_d': 'd の最小値', 'sum_pos_d': '正の d の合計', 'mean_neg_d': '負の d の平均',
              'n_big_up': '大きく増えた日の数'}
Q_BIG = 0.90


# ---------------------------------------------------------------- S1: 1次差分の要約統計
def diff_stats(D, thr):
    """D: (n, 59) の1次差分。どれも日の順番を入れ替えても値が変わらない。"""
    neg = np.where(D < 0, D, np.nan)
    n_neg = (D < 0).sum(1)
    mean_neg = np.where(n_neg > 0, np.nansum(neg, 1) / np.maximum(n_neg, 1), 0.0)  # 負の d が無ければ0
    return np.c_[D.mean(1), D.std(1, ddof=1), np.abs(D).mean(1), D.max(1), D.min(1),
                 np.where(D > 0, D, 0).sum(1), mean_neg, (D > thr).sum(1)]


class DiffStats(BaseEstimator, TransformerMixin):
    """(n, 1, 59) の1次差分 → 8つの要約統計。しきい値は学習データの d すべての q 点(fit のときだけ決める)。"""

    def __init__(self, q=Q_BIG):
        self.q = q

    def fit(self, X, y=None):
        self.thr_ = float(np.quantile(np.asarray(X, float)[:, 0, :], self.q))
        return self

    def transform(self, X):
        return diff_stats(np.asarray(X, float)[:, 0, :], self.thr_)


def s1_pipeline():
    return make_pipeline(DiffStats(), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def run_s1(X, y, meta, sp, n_perm, log):
    t = time.time()
    folds = evaluate(s1_pipeline(), X, y, sp)
    obs = summarize(folds)

    def chunk(idx):
        out = []
        for i in idx:
            yp = permute_within(y, meta.muniyear.values, np.random.default_rng([0, i]))
            s = summarize(evaluate(s1_pipeline(), X, yp, sp))
            out.append((s['bacc_mean'], s['auc_mean']))
        return out
    res = Parallel(n_jobs=fast.N_JOBS)(delayed(chunk)(c) for c in np.array_split(np.arange(n_perm), fast.N_JOBS))
    null = np.array([v for part in res for v in part])
    np.save(os.path.join(OUT_DIR, 'null_S1_diff_stats.npy'), null)
    r = {'config': 'S1_diff_stats', 'label': 'S1 diff stats + LR', 'experiment': 'S1', 'dataset': 'main',
         'kind': 'logreg', 'n': len(y), 'n_high': int(y.sum()), 'n_low': int((1 - y).sum()), 'n_seeds': 1,
         'n_perm': n_perm}
    for m in ['bacc', 'auc']:
        j = 0 if m == 'bacc' else 1
        r[f'{m}_mean'] = obs[f'{m}_mean']
        r[f'{m}_seed_min'] = r[f'{m}_seed_median'] = r[f'{m}_seed_max'] = obs[f'{m}_mean']
        r[f'{m}_sd_fold'], r[f'{m}_sd_repeat'] = obs[f'{m}_sd_fold'], obs[f'{m}_sd_repeat']
        r[f'{m}_null_mean'] = null[:, j].mean()
        r[f'{m}_null_q95'] = np.quantile(null[:, j], 0.95)
        r[f'{m}_p'] = (1 + np.sum(null[:, j] >= r[f'{m}_mean'])) / (1 + n_perm)
    r['seconds'] = round(time.time() - t, 1)
    thr = [DiffStats().fit(X[tr]).thr_ for _, _, tr, _ in sp]
    log(f'  S1_diff_stats                  n={len(y)} AUC {r["auc_mean"]:.3f} (q95 {r["auc_null_q95"]:.3f}, '
        f'p={r["auc_p"]:.3f})  bacc {r["bacc_mean"]:.3f} (q95 {r["bacc_null_q95"]:.3f}, p={r["bacc_p"]:.3f})  '
        f'しきい値(fold ごと) {np.min(thr):.4f}〜{np.max(thr):.4f}  [{r["seconds"]}s]')
    return r, folds.assign(config='S1_diff_stats'), thr


# ---------------------------------------------------------------- S2: シャッフル
def shuffle_each(D, k):
    rng = np.random.default_rng([1, k])
    perms = np.array([rng.permutation(D.shape[-1]) for _ in range(len(D))])
    return np.take_along_axis(D, perms[:, None, :], axis=-1), perms


def shuffle_common(D, k):
    perm = np.random.default_rng([2, k]).permutation(D.shape[-1])
    return D[..., perm], perm


def run_seeded(config, label, X_by_seed, y, meta, sp, seeds, n_perm, log):
    """s6.run_config と同じ集計。ただし入力が乱数 k ごとに違う(k 番目のシャッフル)。"""
    t = time.time()
    folds, nulls, per_seed = [], [], []
    for k in seeds:
        feats = s6.fold_features('minirocket', k, X_by_seed[k], sp)
        f = s6.eval_cached(feats, y, sp).assign(seed=k)
        null = fast.perm_cached(feats, y, sp, meta.muniyear.values, n_perm)
        del feats
        s = summarize(f)
        per_seed.append({'config': config, 'seed': k, 'bacc_mean': s['bacc_mean'], 'auc_mean': s['auc_mean'],
                         'alpha_at_max': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[-1]).mean(),
                         'alpha_at_min': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[0]).mean()})
        folds.append(f)
        nulls.append(null)
    folds = pd.concat(folds)
    ps = pd.DataFrame(per_seed)
    null = np.mean(np.stack(nulls), axis=0)  # 置換 i ごとに乱数で平均
    np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null)
    r = {'config': config, 'label': label, 'experiment': config.split('_')[0], 'dataset': 'main',
         'kind': 'minirocket', 'n': len(y), 'n_high': int(y.sum()), 'n_low': int((1 - y).sum()),
         'n_seeds': len(seeds), 'n_perm': n_perm}
    for m in ['bacc', 'auc']:
        j = 0 if m == 'bacc' else 1
        r[f'{m}_mean'] = ps[f'{m}_mean'].mean()
        r[f'{m}_seed_min'], r[f'{m}_seed_median'], r[f'{m}_seed_max'] = \
            ps[f'{m}_mean'].min(), ps[f'{m}_mean'].median(), ps[f'{m}_mean'].max()
        rep = folds.groupby(['seed', 'repeat'])[m].mean()
        r[f'{m}_sd_fold'] = folds[m].std()
        r[f'{m}_sd_repeat'] = rep.groupby('repeat').mean().std()
        r[f'{m}_null_mean'] = null[:, j].mean()
        r[f'{m}_null_q95'] = np.quantile(null[:, j], 0.95)
        r[f'{m}_p'] = (1 + np.sum(null[:, j] >= r[f'{m}_mean'])) / (1 + n_perm)
    r['alpha_at_max'] = np.isclose(folds.alpha_over_n, s6.BASE_ALPHAS[-1]).mean()
    r['alpha_at_min'] = np.isclose(folds.alpha_over_n, s6.BASE_ALPHAS[0]).mean()
    r['alpha_over_n_median'] = folds.alpha_over_n.median()
    r['seconds'] = round(time.time() - t, 1)
    log(f'  {config:30s} n={len(y)} AUC {r["auc_mean"]:.3f} (乱数 {r["auc_seed_min"]:.3f}〜{r["auc_seed_max"]:.3f}; '
        f'q95 {r["auc_null_q95"]:.3f}, p={r["auc_p"]:.3f})  bacc {r["bacc_mean"]:.3f} '
        f'(q95 {r["bacc_null_q95"]:.3f}, p={r["bacc_p"]:.3f})  alpha 上限 {r["alpha_at_max"]:.2f}  [{r["seconds"]}s]')
    return r, folds.assign(config=config), ps


# ---------------------------------------------------------------- 図
def plot_boxes(summary, folds_all, per_seed_all, baselines):
    cfgs = [D_CONFIG, 'S1_diff_stats', 'S2a_shuffle_each', 'S2b_shuffle_common']
    bl_style = {'base_vwc_stats_main': ('--', '#e34948'), 'base_seed_doy_main': (':', '#e34948'),
                'base_irrigated_main': ('-.', '#9a9890')}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        for i, cfg in enumerate(cfgs):
            x = i + 1
            f = folds_all[folds_all.config == cfg]
            rep = f.groupby(['seed', 'repeat'])[m].mean().groupby('repeat').mean().values
            bp = ax.boxplot([rep], positions=[x], widths=0.5, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5), boxprops=dict(lw=0),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
            bp['boxes'][0].set_facecolor(COLORS[cfg])
            bp['boxes'][0].set_alpha(0.45)
            pts = per_seed_all[per_seed_all.config == cfg][f'{m}_mean'].values
            ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)) if len(pts) > 1 else [x], pts, s=14, color=INK, zorder=3)
            r = summary[summary.config == cfg].iloc[0]
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
        for _, b in baselines.iterrows():
            ls, c = bl_style[b.config]
            ax.axhline(b[f'{m}_mean'], color=c, lw=1.1, ls=ls, label=f'段階6の基準線: {BL_NAMES[b.config]} {b[f"{m}_mean"]:.3f}')
        ax.axhline(0.5, color=MUTED, lw=0.6)
        ax.set_xticks(range(1, len(cfgs) + 1), [LABELS[c] for c in cfgs], fontsize=9)
        ax.set_ylabel(f'{lab}(20回の繰り返し平均)', color=INK)
        ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1)
        s6.style(ax)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker='o', color=INK, ls='', ms=4), Line2D([], [], color=INK, lw=2)],
               ['乱数ごとの値(D・S2 は10通り、S1 は1つ)', '置換検定の帰無分布の95%点(D・S1 1000回、S2 200回)'],
               loc='upper left', bbox_to_anchor=(0.02, 0.93), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('1次差分の情報の切り分け(main 82件、箱 = 乱数で平均した20回の繰り返し平均)',
                 fontsize=13, color=INK, x=0.02, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_stage10.png'), dpi=150)
    plt.close(fig)


def plot_paired(pairs, ps):
    comps = ['D − S2a', 'D − S2b']
    fig, axs = plt.subplots(len(comps), 2, figsize=(12, 3.6 * len(comps)), squeeze=False)
    for i, c in enumerate(comps):
        s = pairs[pairs.comparison == c]
        avg = s.groupby(['repeat', 'fold'])[['d_auc', 'd_bacc']].mean().reset_index()
        for j, m in enumerate(['auc', 'bacc']):
            ax = axs[i, j]
            rep = avg.groupby('repeat')[f'd_{m}'].mean()
            seedm = s.groupby('seed')[f'd_{m}'].mean()
            ax.hist(avg[f'd_{m}'], bins=np.linspace(-0.6, 0.6, 31), color='#2a78d6', alpha=0.35,
                    label='fold ごとの差(乱数平均、100)')
            ax2 = ax.twinx()
            ax2.hist(rep, bins=np.linspace(-0.6, 0.6, 61), color='#eb6834', alpha=0.9, label='繰り返し平均の差(20)')
            ax2.set_yticks([])
            ax.scatter(seedm, np.full(len(seedm), -0.6), marker='|', s=200, color=INK, clip_on=False, zorder=4,
                       label='乱数ごとの平均の差(10)')
            ax.axvline(0, color=INK, lw=1)
            r = ps[(ps.comparison == c) & (ps.metric == m)].iloc[0]
            ax.set_title(f'{c}  {m}: 平均 {r.mean_diff:+.3f}、繰り返し平均の 2.5〜97.5% '
                         f'[{r.rep_q025:+.3f}, {r.rep_q975:+.3f}]、正 {r.rep_frac_pos:.0%}',
                         fontsize=9, color=INK, loc='left')
            s6.style(ax)
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('シャッフルによる差(D − S2、同じ分割・同じカーネルの乱数)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'paired_diff_shuffle.png'), dpi=150)
    plt.close(fig)


def plot_features(F, y, uni):
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    rng = np.random.default_rng(0)
    for ax, (j, name) in zip(axs.ravel(), enumerate(FEATURES)):
        data = [F[y == 0, j], F[y == 1, j]]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                        boxprops=dict(lw=0), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        showfliers=False)
        for patch, c in zip(bp['boxes'], ['#eb6834', '#2a78d6']):
            patch.set_facecolor(c)
            patch.set_alpha(0.4)
        for x, v in zip([1, 2], data):
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(v)), v, s=8, color=INK, alpha=0.6, zorder=3)
        a = uni.loc[uni.feature == name, 'auc_all82'].iloc[0]
        ax.set_title(f'{FEATURE_JA[name]}(単変量 AUC {a:.3f})', fontsize=9.5, color=INK, loc='left')
        ax.set_xticks([1, 2], [f'低収 ({int((y == 0).sum())})', f'多収 ({int((y == 1).sum())})'], fontsize=9)
        s6.style(ax)
    fig.suptitle('S1 の8つの特徴量の多収・低収別の分布(全82件、記述用。8 のしきい値は全82件の d の90%点)',
                 fontsize=12, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 's1_features_by_class.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=200, help='S2 の置換回数')
    ap.add_argument('--n-perm-s1', type=int, default=1000, help='S1 の置換回数')
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    s6.OUT_DIR = OUT_DIR
    s6.N_JOBS = fast.N_JOBS
    seeds = list(range(args.n_seeds))
    log(f'並列数 {fast.N_JOBS}、乱数 {seeds[0]}〜{seeds[-1]}、置換 S1 {args.n_perm_s1} 回 / S2 {args.n_perm} 回')

    Xm, mm = s6.s5.datasets()['main']
    y = mm.y.values
    Xd = first_diff(Xm)  # (82, 1, 59)
    sp = make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    D_folds, D_seed, D_row, baselines, D_null = s7.load_stage6_D()
    best_bl = {m: baselines[f'{m}_mean'].max() for m in ['auc', 'bacc']}
    log(f'段階6の実験D を読み込んだ: AUC {D_row["auc_mean"]:.3f}、bacc {D_row["bacc_mean"]:.3f}(置換 {int(D_row["n_perm"])} 回)')
    for _, b in baselines.iterrows():
        log(f'  段階6の基準線 {BL_NAMES[b.config]}: AUC {b.auc_mean:.3f}、bacc {b.bacc_mean:.3f}')
    log(f'  最良の基準線: AUC {best_bl["auc"]:.3f}、bacc {best_bl["bacc"]:.3f}')

    # ---- 確認
    log('\n[確認]')
    assert s7.check_same_splits(y, mm), '段階6の実験Dと分割が違う'
    log('段階6の実験Dと同じ分割: True')
    shuf_a = {k: shuffle_each(Xd, k) for k in seeds}
    shuf_b = {k: shuffle_common(Xd, k) for k in seeds}
    Xa = {k: v[0] for k, v in shuf_a.items()}
    Xb = {k: v[0] for k, v in shuf_b.items()}
    srt = np.sort(Xd, axis=-1)
    ok = all(np.array_equal(np.sort(Xa[k], -1), srt) and np.array_equal(np.sort(Xb[k], -1), srt) for k in seeds)
    log(f'シャッフル後も各系列の値の集まりが同じ: {ok}')
    assert ok
    F0 = diff_stats(Xd[:, 0, :], 0.01)
    ok = all(np.allclose(diff_stats(Xa[k][:, 0, :], 0.01), F0) and np.allclose(diff_stats(Xb[k][:, 0, :], 0.01), F0)
             for k in seeds)
    log(f'S1 の特徴量がシャッフルで変わらない(日の順番によらない): {ok}')
    assert ok
    ok = s7.check_pipeline(Xa[0], y, mm) and s7.check_pipeline(Xb[0], y, mm)
    log(f'Pipeline と特徴量の使い回しの一致(S2a・S2b、乱数0、先頭10 fold): {ok}')
    assert ok
    ok, mx, ta, tb = s7.check_fast_vs_original(Xa[0], y, mm, s6.perm_cached)
    log(f'高速版と元の perm_cached の一致(S2a、乱数0、置換20回): {ok}  最大差 {mx:.1e}  元 {ta:.1f}s / 新 {tb:.1f}s')
    assert ok

    # シャッフルの記録
    rows = []
    for k in seeds:
        pb = shuf_b[k][1]
        pa = shuf_a[k][1]
        adj_b = np.mean(np.abs(np.diff(pb)) == 1)  # 並べ替えた後も隣り合う元の日の組の割合
        adj_a = np.mean(np.abs(np.diff(pa, axis=1)) == 1)
        rows.append({'seed': k, 'S2b_perm': ' '.join(map(str, pb)), 'S2b_fixed_points': int((pb == np.arange(59)).sum()),
                     'S2b_adjacent_kept': adj_b, 'S2a_fixed_points_mean': (pa == np.arange(59)).sum(1).mean(),
                     'S2a_adjacent_kept': adj_a})
        np.save(os.path.join(OUT_DIR, f'shuffle_S2a_perms_seed{k}.npy'), pa)
    shuf_tab = pd.DataFrame(rows)
    shuf_tab.to_csv(os.path.join(OUT_DIR, 'shuffles.csv'), index=False, encoding='utf-8-sig')
    log('シャッフルの記録(S2a: default_rng([1, k]) で系列ごと、S2b: default_rng([2, k]) で1通り):')
    log(shuf_tab.drop(columns='S2b_perm').round(3).to_string(index=False))

    # ---- S1
    log(f'\n[S1] 1次差分の要約統計 + LR(置換 {args.n_perm_s1} 回)')
    r1, f1, thr = run_s1(Xd, y, mm, sp, args.n_perm_s1, log)
    pd.DataFrame({'repeat': [r for r, *_ in sp], 'fold': [f for _, f, *_ in sp], 'threshold_q90': thr}).to_csv(
        os.path.join(OUT_DIR, 's1_thresholds.csv'), index=False, encoding='utf-8-sig')
    F_all = diff_stats(Xd[:, 0, :], np.quantile(Xd[:, 0, :], Q_BIG))
    uni = pd.DataFrame({'feature': FEATURES, 'feature_ja': [FEATURE_JA[f] for f in FEATURES],
                        'auc_all82': [roc_auc_score(y, F_all[:, j]) for j in range(len(FEATURES))],
                        'mean_low': [F_all[y == 0, j].mean() for j in range(len(FEATURES))],
                        'mean_high': [F_all[y == 1, j].mean() for j in range(len(FEATURES))]})
    uni.to_csv(os.path.join(OUT_DIR, 's1_univariate_auc.csv'), index=False, encoding='utf-8-sig')
    log('S1 の単変量 AUC(参考。全82件、学習・テストに分けない):')
    log(uni.round(4).to_string(index=False))
    corr = pd.DataFrame(F_all, columns=FEATURES).corr()
    corr.to_csv(os.path.join(OUT_DIR, 's1_feature_corr.csv'), encoding='utf-8-sig')

    # ---- S2
    log(f'\n[S2] シャッフルした1次差分 + MiniRocket(置換 {args.n_perm} 回)')
    r2a, f2a, ps2a = run_seeded('S2a_shuffle_each', 'S2a shuffle each', Xa, y, mm, sp, seeds, args.n_perm, log)
    r2b, f2b, ps2b = run_seeded('S2b_shuffle_common', 'S2b shuffle common', Xb, y, mm, sp, seeds, args.n_perm, log)

    # ---- まとめ
    D_row = {**D_row, 'experiment': 'D(段階6)', 'label': 'D diff(段階6)'}
    summary = pd.DataFrame([D_row, r1, r2a, r2b])
    for m in ['bacc', 'auc']:
        summary[f'{m}_best_baseline'] = best_bl[m]
        summary[f'{m}_hit'] = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > best_bl[m])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    f1_seeded = pd.concat([f1.assign(seed=k) for k in seeds])  # S1 は乱数によらないので、D の各乱数と組にする
    folds_all = pd.concat([D_folds, f1.assign(seed=0), f2a, f2b])
    folds_all.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed_all = pd.concat([D_seed, pd.DataFrame([{'config': 'S1_diff_stats', 'seed': 0, 'bacc_mean': r1['bacc_mean'],
                                                       'auc_mean': r1['auc_mean']}]), ps2a, ps2b])
    per_seed_all.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')

    fd = D_folds[D_folds.seed.isin(seeds)]
    pairs = pd.concat([s7.paired(fd, f2a, 'D − S2a'), s7.paired(fd, f2b, 'D − S2b'),
                       s7.paired(fd, f1_seeded, 'D − S1(記述)'), s7.paired(f2a, f2b, 'S2a − S2b')])
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = s7.paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds_all, per_seed_all, baselines)
    plot_paired(pairs, ps)
    plot_features(F_all, y, uni)

    show = ['config', 'n', 'auc_mean', 'auc_seed_min', 'auc_seed_max', 'auc_null_mean', 'auc_null_q95', 'auc_p',
            'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p', 'bacc_hit', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log('\n[fold ごとの差(乱数で平均)]')
    log(ps.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
