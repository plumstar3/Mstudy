"""
段階7(追加): プーリングの違い(PPV のみ / PPV + max)の比較。記述的な比較で、主な設定(MiniRocket・1次差分)は変えない。

  変換: sktime 1.2.0 の Rocket(num_kernels=10000、normalise=False を明示、random_state=0〜9)
        1回の変換の出力から2通りの特徴量を作る(カーネルが同じなので、違いは max の有無だけ)
          ppv     PPV の列だけ(10000 列)
          ppvmax  PPV と max の両方(20000 列)
        sktime の Rocket の出力は、カーネル j について 2j 列目が PPV、2j+1 列目が max
        (sktime/transformations/rocket/_rocket_numba.py の _apply_kernels)。起動時に、カーネルの重み・bias・
        dilation・padding から PPV と max を計算し直して照合する(check_rocket_columns)。
  入力: main(82件)の1次差分(59点)と生の値(60点)
  標準化・分類器・評価・統計量・置換検定(1000回)は段階6の実験Dと同じ(alpha 候補 × n_train)。
  StandardScaler は列ごとなので、ppvmax を標準化してから PPV の列を取り出したものは、ppv だけを標準化したものと同じ。

  比較(主): 同じ入力の中で ppvmax − ppv(同じ分割・同じ乱数の fold ごとの差)
  参考(記述のみ): Rocket の ppv と MiniRocket(1次差分は段階6の実験D、生の値は段階7の G1)
  探索的な参考: 乱数0での Ridge の係数の大きさ(PPV の列と max の列)

実行(別の PC で。outputs/stage6/ が必要。生の値の参考比較には、先に run_stage7.py を実行しておく):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage7\\run_stage7_pooling.py
  試運転: ... run_stage7_pooling.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage7/pooling/
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, STAGE_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
from src.evaluation import evaluate, make_splits  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage7', 'pooling')
S7_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage7')
N_KERNELS = 10000
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 変換
class RocketPool(BaseEstimator, TransformerMixin):
    """sktime の Rocket(normalise=False)。pool='ppv' なら PPV の列だけ、'ppvmax' なら全列(PPV と max が交互)。"""

    def __init__(self, pool='ppvmax', num_kernels=N_KERNELS, random_state=0):
        self.pool, self.num_kernels, self.random_state = pool, num_kernels, random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import Rocket
        self.t_ = Rocket(num_kernels=self.num_kernels, normalise=False,
                         random_state=self.random_state).fit(np.asarray(X, float))
        return self

    def transform(self, X):
        F = np.asarray(self.t_.transform(np.asarray(X, float)), dtype=float)
        return F[:, 0::2] if self.pool == 'ppv' else F


def _rocket_fold(seed, X, tr, te):
    """ppvmax の特徴量(学習データで fit した StandardScaler で標準化)。ppv はこの偶数列を取り出す。"""
    tf = make_pipeline(RocketPool('ppvmax', random_state=seed), StandardScaler(with_mean=False)).fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[te])


def fold_features(kind, seed, X, splits):
    """s6.fold_features の差し替え。kind が 'rocket_ppv' / 'rocket_ppvmax' のときは Rocket を使う。"""
    if not kind.startswith('rocket_'):
        return _ORIGINAL_FOLD_FEATURES(kind, seed, X, splits)
    feats = Parallel(n_jobs=s6.N_JOBS)(delayed(_rocket_fold)(seed, X, tr, te) for r, f, tr, te in splits)
    if kind == 'rocket_ppv':
        return [(np.ascontiguousarray(a[:, 0::2]), np.ascontiguousarray(b[:, 0::2])) for a, b in feats]
    return feats


_ORIGINAL_FOLD_FEATURES = s6.fold_features


# ---------------------------------------------------------------- 確認
def recompute_ppv_max(x, weights, length, bias, dilation, padding):
    """sktime の _apply_kernel_univariate と同じ計算を numpy(float64)で行う。"""
    n = len(x)
    out_len = (n + 2 * padding) - (length - 1) * dilation
    end = (n + padding) - (length - 1) * dilation
    sums = []
    for i in range(-padding, end):
        idx = i + dilation * np.arange(length)
        ok = (idx > -1) & (idx < n)
        sums.append(bias + np.sum(weights[ok].astype(np.float64) * x[idx[ok]].astype(np.float64)))
    sums = np.array(sums)
    return (sums > 0).sum() / out_len, sums.max(), np.abs(sums).min()


def check_rocket_columns(X, n_series=5, seed=0, n_check=300):
    """先頭 n_check 本のカーネルについて、2j 列 = PPV、2j+1 列 = max かを確かめる。"""
    from sktime.transformations.panel.rocket import Rocket
    t = Rocket(num_kernels=N_KERNELS, normalise=False, random_state=seed)
    assert t.get_params()['normalise'] is False
    t.fit(X[:n_series])
    F = np.asarray(t.transform(X[:n_series]), dtype=np.float64)
    weights, lengths, biases, dilations, paddings, nci, ci = t.kernels
    assert (nci == 1).all(), '1変量のはず'
    x32 = X[:n_series, 0].astype(np.float32)  # sktime は float32 で計算する
    rows = []
    a1 = 0
    for j in range(N_KERNELS):
        b1 = a1 + lengths[j]
        if j < n_check:
            for s in range(n_series):
                ppv, mx, near0 = recompute_ppv_max(x32[s], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])
                rows.append({'kernel': j, 'series': s, 'ppv_recomputed': ppv, 'max_recomputed': mx,
                             'col_2j': F[s, 2 * j], 'col_2j1': F[s, 2 * j + 1], 'min_abs_sum': near0,
                             'out_len': (X.shape[2] + 2 * paddings[j]) - (lengths[j] - 1) * dilations[j]})
        a1 = b1
    df = pd.DataFrame(rows)
    df['d_ppv'] = (df.col_2j - df.ppv_recomputed).abs()
    df['d_max'] = (df.col_2j1 - df.max_recomputed).abs()
    df['d_max_rel'] = df.d_max / np.maximum(1.0, df.max_recomputed.abs())
    # 入れ替えた場合(2j が max、2j+1 が PPV)にどれだけ合わないか
    df['d_swapped'] = (df.col_2j - df.max_recomputed).abs() + (df.col_2j1 - df.ppv_recomputed).abs()
    ppv_ok = df.d_ppv <= 1.0 / df.out_len + 1e-6  # 0 付近の和の丸めで1件分ずれることだけを許す
    max_ok = df.d_max_rel <= 1e-4
    ok = bool(ppv_ok.all() and max_ok.all() and (df.col_2j.between(0, 1)).all())
    return ok, df


def check_pipeline(X, y, meta, n=10):
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)[:n]
    res = {}
    for pool in ['ppv', 'ppvmax']:
        a = evaluate(make_pipeline(RocketPool(pool, random_state=0), StandardScaler(with_mean=False), s6.RidgeNTrain()),
                     X, y, sp)
        b = s6.eval_cached(fold_features(f'rocket_{pool}', 0, X, sp), y, sp)
        res[pool] = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    return res


# ---------------------------------------------------------------- 係数(探索的な参考)
def _coef_fold(Ftr, y_tr):
    clf = s6.RidgeNTrain().fit(Ftr, y_tr)
    c = np.abs(clf.clf_.coef_.ravel())
    ppv, mx = c[0::2], c[1::2]
    top = np.argsort(c)[::-1][:100]
    return {'share_abs_coef_max': mx.sum() / c.sum(), 'share_top100_max': float((top % 2 == 1).mean()),
            'mean_abs_coef_ppv': ppv.mean(), 'mean_abs_coef_max': mx.mean()}


def coef_importance(X, y, meta, seed=0):
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    feats = fold_features('rocket_ppvmax', seed, X, sp)
    rows = Parallel(n_jobs=s6.N_JOBS)(delayed(_coef_fold)(Ftr, y[tr]) for (r, f, tr, te), (Ftr, Fte) in zip(sp, feats))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 図
def plot_boxes(summary, folds_all, per_seed_all, best_bl):
    order = [c for c in ['H_diff_ppv', 'H_diff_ppvmax', 'ref_MR_diff', 'H_raw_ppv', 'H_raw_ppvmax', 'ref_MR_raw']
             if c in set(summary.config)]
    names = {'H_diff_ppv': '1次差分\nRocket PPV', 'H_diff_ppvmax': '1次差分\nRocket PPV+max',
             'ref_MR_diff': '1次差分\nMiniRocket(参考)', 'H_raw_ppv': '生の値\nRocket PPV',
             'H_raw_ppvmax': '生の値\nRocket PPV+max', 'ref_MR_raw': '生の値\nMiniRocket(参考)'}
    color = {'ppv': '#2a78d6', 'ppvmax': '#eb6834', 'MR': '#9a9890'}
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        for x, cfg in enumerate(order, start=1):
            f = folds_all[folds_all.config == cfg]
            rep = f.groupby(['seed', 'repeat'])[m].mean().groupby('repeat').mean().values
            bp = ax.boxplot([rep], positions=[x], widths=0.5, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5), boxprops=dict(lw=0),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
            k = 'MR' if cfg.startswith('ref') else cfg.split('_')[-1]
            bp['boxes'][0].set_facecolor(color[k])
            bp['boxes'][0].set_alpha(0.45)
            pts = per_seed_all[per_seed_all.config == cfg][f'{m}_mean'].values
            ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)), pts, s=12, color=INK, zorder=3)
            r = summary[summary.config == cfg].iloc[0]
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
            ax.plot(x + 0.4, best_bl[m], marker='D', ms=6, color='#e34948', ls='')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.axvline(3.5, color=MUTED, lw=0.8)
        ax.set_ylabel(f'{lab}', color=INK)
        s6.style(ax)
    axs[1].set_xticks(range(1, len(order) + 1), [names[c] for c in order], fontsize=9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=color['ppv'], alpha=0.45), Patch(color=color['ppvmax'], alpha=0.45),
                Patch(color=color['MR'], alpha=0.45), Line2D([], [], marker='o', color=INK, ls='', ms=4),
                Line2D([], [], color=INK, lw=2), Line2D([], [], marker='D', color='#e34948', ls='')],
               ['Rocket PPV', 'Rocket PPV+max', 'MiniRocket(参考、カーネルの作り方が違う)', '乱数ごとの値(10通り)',
                '置換検定の95%点', '最良の基準線(段階6の main)'],
               loc='upper left', bbox_to_anchor=(0.02, 0.955), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('プーリングの比較(main 82件、箱 = 10通りの乱数で平均した20回の繰り返し平均)',
                 fontsize=13, color=INK, x=0.02, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_pooling.png'), dpi=150)
    plt.close(fig)


def plot_paired(pairs, ps):
    """対の差の分布。見出しは2行に分ける。"""
    comps = list(pairs.comparison.unique())
    fig, axs = plt.subplots(len(comps), 2, figsize=(12, 3.6 * len(comps)), squeeze=False)
    for i, c in enumerate(comps):
        s = pairs[pairs.comparison == c]
        avg = s.groupby(['repeat', 'fold'])[['d_auc', 'd_bacc']].mean().reset_index()
        for j, m in enumerate(['auc', 'bacc']):
            ax = axs[i, j]
            rep = avg.groupby('repeat')[f'd_{m}'].mean()
            seedm = s.groupby('seed')[f'd_{m}'].mean()
            ax.hist(avg[f'd_{m}'], bins=np.linspace(-0.4, 0.4, 33), color='#2a78d6', alpha=0.35,
                    label='fold ごとの差(乱数平均、100)')
            ax2 = ax.twinx()
            ax2.hist(rep, bins=np.linspace(-0.4, 0.4, 65), color='#eb6834', alpha=0.9, label='繰り返し平均の差(20)')
            ax2.set_yticks([])
            ax.scatter(seedm, np.full(len(seedm), -0.6), marker='|', s=200, color=INK, clip_on=False, zorder=4,
                       label='乱数ごとの平均の差(10)')
            ax.axvline(0, color=INK, lw=1)
            r = ps[(ps.comparison == c) & (ps.metric == m)].iloc[0]
            metric = 'AUC' if m == 'auc' else 'bacc'
            ax.set_title(f'{c}({metric})\n平均 {r.mean_diff:+.3f}、繰り返し平均の 2.5〜97.5% '
                         f'[{r.rep_q025:+.3f}, {r.rep_q975:+.3f}]、正 {r.rep_frac_pos:.0%}',
                         fontsize=9, color=INK, loc='left')
            s6.style(ax)
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('プーリングの差(同じ分割・同じ乱数)。上2段が主な比較、下2段は参考(カーネルの作り方が違う)',
                 fontsize=12, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, 'paired_diff_pooling.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    original_perm_cached = s6.perm_cached
    s6.OUT_DIR = OUT_DIR
    s6.N_JOBS = fast.N_JOBS
    s6.SEEDS = list(range(args.n_seeds))
    s6.fold_features = fold_features
    log(f'並列数 {fast.N_JOBS}、乱数 {s6.SEEDS[0]}〜{s6.SEEDS[-1]}、置換 {args.n_perm} 回')

    Xm, mm = s6.s5.datasets()['main']
    y = mm.y.values
    inputs = {'diff': first_diff(Xm), 'raw': Xm.astype(float)}
    D_folds, D_seed, D_row, baselines, _ = s7.load_stage6_D()
    best_bl = {m: baselines[f'{m}_mean'].max() for m in ['auc', 'bacc']}

    # 1. sktime の Rocket の列の並び
    for name, X in inputs.items():
        ok, df = check_rocket_columns(X)
        df.to_csv(os.path.join(OUT_DIR, f'check_rocket_columns_{name}.csv'), index=False, encoding='utf-8-sig')
        log(f'Rocket の列の確認({name}、5系列 × 先頭300カーネル): 2j 列 = PPV、2j+1 列 = max → {ok}'
            f'  PPV の最大差 {df.d_ppv.max():.2e}(1件分 = 1/出力長)、max の最大相対差 {df.d_max_rel.max():.2e}、'
            f'入れ替えた場合の最小のずれ {df.d_swapped.min():.2e}')
        assert ok
    # 2. 分割・Pipeline・高速版の置換検定
    assert s7.check_same_splits(y, mm), '段階6の実験Dと分割が違う'
    log('段階6の実験Dと同じ分割: True')
    for name, X in inputs.items():
        res = check_pipeline(X, y, mm)
        log(f'Pipeline と特徴量の使い回しの一致({name}、先頭10 fold、乱数0): {res}')
        assert all(res.values())
    sp = make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    feats = fold_features('rocket_ppvmax', 0, inputs['diff'], sp)
    a = original_perm_cached(feats, y, sp, mm.muniyear.values, 20)
    b = fast.perm_cached(feats, y, sp, mm.muniyear.values, 20)
    ok = np.array_equal(a, b, equal_nan=True)
    log(f'高速版と元の perm_cached の一致(1次差分・Rocket PPV+max、乱数0、置換20回): {ok}')
    assert ok
    s6.perm_cached = fast.perm_cached

    # 3. 本番
    rows, fold_rows, seed_rows = [], [], []
    for name, X in inputs.items():
        for pool in ['ppvmax', 'ppv']:
            cfg = f'H_{name}_{pool}'
            r, f, ps, _ = s6.run_config(cfg, f'H {name} {pool}', 'H', 'main', X, y, mm, f'rocket_{pool}', args.n_perm, log)
            rows.append(r)
            fold_rows.append(f)
            seed_rows.append(ps)
    folds = pd.concat(fold_rows)
    per_seed = pd.concat(seed_rows)

    # 参考: MiniRocket(1次差分は段階6の実験D、生の値は段階7の G1。あれば)
    ref_rows = [{**D_row, 'config': 'ref_MR_diff', 'label': 'MiniRocket diff(段階6 実験D)', 'experiment': 'ref'}]
    ref_folds = [D_folds.assign(config='ref_MR_diff')]
    ref_seed = [D_seed.assign(config='ref_MR_diff')]
    g1 = os.path.join(S7_DIR, 'summary.csv')
    if os.path.exists(g1):
        s = pd.read_csv(g1, encoding='utf-8-sig')
        if (s.config == 'G1_main_raw').any():
            ref_rows.append({**s[s.config == 'G1_main_raw'].iloc[0].to_dict(), 'config': 'ref_MR_raw',
                             'label': 'MiniRocket raw(段階7 G1)', 'experiment': 'ref'})
            f7 = pd.read_csv(os.path.join(S7_DIR, 'folds.csv'), encoding='utf-8-sig')
            ref_folds.append(f7[f7.config == 'G1_main_raw'].astype({'seed': int}).assign(config='ref_MR_raw'))
            p7 = pd.read_csv(os.path.join(S7_DIR, 'per_seed.csv'), encoding='utf-8-sig')
            ref_seed.append(p7[p7.config == 'G1_main_raw'].assign(config='ref_MR_raw'))
    if len(ref_rows) == 1:
        log('段階7の G1(MiniRocket・生の値)の結果が無いので、生の値の MiniRocket との参考比較は省いた')

    summary = pd.DataFrame(rows + ref_rows)
    for m in ['bacc', 'auc']:
        summary[f'{m}_best_baseline'] = best_bl[m]
        summary[f'{m}_hit'] = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > best_bl[m])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds_all = pd.concat([folds] + ref_folds)
    folds_all.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed_all = pd.concat([per_seed] + ref_seed)
    per_seed_all.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')

    # 対の差。主: ppvmax − ppv。参考(カーネルの作り方が違うので記述のみ): Rocket ppv − MiniRocket
    fz = lambda c: folds_all[(folds_all.config == c) & folds_all.seed.isin(s6.SEEDS)]  # noqa: E731
    pairs = [s7.paired(fz('H_diff_ppvmax'), fz('H_diff_ppv'), '主 diff: ppvmax − ppv'),
             s7.paired(fz('H_raw_ppvmax'), fz('H_raw_ppv'), '主 raw: ppvmax − ppv'),
             s7.paired(fz('H_diff_ppv'), fz('ref_MR_diff'), '参考 diff: Rocket ppv − MiniRocket')]
    if (folds_all.config == 'ref_MR_raw').any():
        pairs.append(s7.paired(fz('H_raw_ppv'), fz('ref_MR_raw'), '参考 raw: Rocket ppv − MiniRocket'))
    pairs = pd.concat(pairs)
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = s7.paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    # 探索的な参考: Ridge の係数の大きさ(乱数0、PPV+max)
    coef = pd.concat([coef_importance(X, y, mm).assign(input=name) for name, X in inputs.items()])
    coef.to_csv(os.path.join(OUT_DIR, 'coef_importance_seed0.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds_all, per_seed_all, best_bl)
    plot_paired(pairs, ps)

    show = ['config', 'n', 'auc_mean', 'auc_seed_min', 'auc_seed_median', 'auc_seed_max', 'auc_null_mean',
            'auc_null_q95', 'auc_p', 'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p', 'bacc_hit',
            'alpha_at_max', 'alpha_at_min']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log('\n[fold ごとの差(10通りの乱数で平均)]')
    log(ps.round(3).to_string(index=False))
    log('\n[探索的な参考: Ridge の |係数| のうち max の列が占める割合(乱数0、100 fold の平均)]')
    log(coef.groupby('input')[['share_abs_coef_max', 'share_top100_max']].mean().round(3).to_string())
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
