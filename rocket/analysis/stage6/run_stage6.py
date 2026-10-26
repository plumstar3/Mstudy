"""
段階6: 事前に固定した確認用の解析(実験D)と、1次差分での揃え方の比較(実験E)、探索的な解析(実験F)。

  実験D  確認用: main(82件)・diff・MiniRocket。統計量 = 乱数10通りそれぞれの「20回の繰り返し平均の AUC」の平均。
         置換1000回(置換 i の並べ方は10通りの乱数で共通)。判定はこの1つだけ(多重比較の補正なし)
  実験E  main_common vs dap_common(77件)を diff で。乱数10通り、置換各200回、fold ごとの差(main − dap)
  実験F  探索的(結論は出さない)。乱数10通り、置換各200回
         F1 生の値(60点)と1次差分(59点)に別々の MiniRocket(各 num_kernels=5000、乱数 2k と 2k+1)をかけ、
            特徴量を横につなぐ(MultiRocket の「元の系列 + 1次差分」と同じ構成。Tan et al., 2022)
         F2 集団内中心化してから1次差分
         F3 main_obs / main_no_kakogawa / main_no_irrigation を diff で
         F4 label_year(main の82件のうち label_year が付く58件)、diff
  基準線 窓内 VWC の要約統計 / 播種日の通日 / 灌水の有無 + LogisticRegression(C=1)、置換200回

モデル: sktime 1.2.0 の MiniRocket(num_kernels=10000、random_state=0〜9)→ StandardScaler(with_mean=False)
        → RidgeClassifierCV(alphas = logspace(−3, 3, 10) × n_train)。MiniRocket 原論文の
        RidgeClassifierCV(alphas=logspace(−3, 3, 10), normalize=True) と等価(precheck.py で確認)。

計算の工夫は段階5と同じ(fold ごとの特徴量を一度だけ計算し、置換のたびに Ridge だけを学習し直す)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage6\\run_stage6.py
出力: rocket/outputs/stage6/
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')  # sktime・sklearn の将来の仕様変更の警告(結果には影響しない)

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage5'))
import run_stage5 as s5  # noqa: E402
from src.evaluation import evaluate, make_splits, permute_within, split_table, summarize  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage6')
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
N_SPLITS, N_REPEATS, SEED0 = 5, 20, 0
N_KERNELS = 10000
SEEDS = list(range(10))
BASE_ALPHAS = np.logspace(-3, 3, 10)
N_JOBS = 20

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_MAIN, C_ALT, C_RED = '#2a78d6', '#eb6834', '#e34948'


# ---------------------------------------------------------------- モデル
class Panel(BaseEstimator, TransformerMixin):
    """sktime の MiniRocket。

    kind='minirocket': 入力そのものに MiniRocket(num_kernels)をかける。
    kind='raw_diff_concat': 生の値(60点)と1次差分(59点)に、それぞれ別の MiniRocket(num_kernels // 2)を
        かけて特徴量を横につなぐ(MultiRocket の「元の系列 + 1次差分」と同じ構成。Tan et al., 2022)。
        乱数は生の値に 2k、1次差分に 2k+1(k = random_state)。入力 X は生の値。
    """

    def __init__(self, kind='minirocket', num_kernels=N_KERNELS, random_state=0):
        self.kind, self.num_kernels, self.random_state = kind, num_kernels, random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MiniRocket
        X = np.asarray(X, float)
        if self.kind == 'minirocket':
            self.t_ = [MiniRocket(num_kernels=self.num_kernels, random_state=self.random_state).fit(X)]
        elif self.kind == 'raw_diff_concat':
            k, half = self.random_state, self.num_kernels // 2
            self.t_ = [MiniRocket(num_kernels=half, random_state=2 * k).fit(X),
                       MiniRocket(num_kernels=half, random_state=2 * k + 1).fit(first_diff(X))]
        else:
            raise ValueError(self.kind)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        inputs = [X] if self.kind == 'minirocket' else [X, first_diff(X)]
        return np.hstack([np.asarray(t.transform(x), dtype=float) for t, x in zip(self.t_, inputs)])


class RidgeNTrain(BaseEstimator, ClassifierMixin):
    """RidgeClassifierCV(alphas = base_alphas × 学習データの件数)。"""

    def __init__(self, base_alphas=BASE_ALPHAS):
        self.base_alphas = base_alphas

    def fit(self, X, y):
        self.n_train_ = len(X)
        self.clf_ = RidgeClassifierCV(alphas=np.asarray(self.base_alphas) * self.n_train_).fit(X, y)
        self.alpha_ = self.clf_.alpha_
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(X)

    def decision_function(self, X):
        return self.clf_.decision_function(X)


def pipeline(kind, seed):
    return make_pipeline(Panel(kind, random_state=seed), StandardScaler(with_mean=False), RidgeNTrain())


def _one_fold(kind, seed, X, tr, te):
    tf = make_pipeline(Panel(kind, random_state=seed), StandardScaler(with_mean=False)).fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[te])


def fold_features(kind, seed, X, splits):
    """fold ごとに、学習データだけで変換器と StandardScaler を fit する。"""
    return Parallel(n_jobs=N_JOBS)(delayed(_one_fold)(kind, seed, X, tr, te) for r, f, tr, te in splits)


def eval_cached(feats, y, splits):
    rows = []
    for (r, f, tr, te), (Ftr, Fte) in zip(splits, feats):
        clf = RidgeNTrain().fit(Ftr, y[tr])
        pred, score = clf.predict(Fte), clf.decision_function(Fte)
        auc = roc_auc_score(y[te], score) if len(np.unique(y[te])) == 2 else np.nan
        rows.append({'repeat': r, 'fold': f, 'n_test': len(te), 'n_train': len(tr),
                     'alpha_over_n': clf.alpha_ / len(tr),
                     'bacc': balanced_accuracy_score(y[te], pred), 'auc': auc})
    return pd.DataFrame(rows)


def _perm_chunk(feats, y, splits, perm_groups, idx):
    out = []
    for i in idx:
        yp = permute_within(y, perm_groups, np.random.default_rng([0, i]))  # 置換 i は乱数(カーネル)によらず共通
        s = summarize(eval_cached(feats, yp, splits))
        out.append((s['bacc_mean'], s['auc_mean']))
    return out


def perm_cached(feats, y, splits, perm_groups, n_perm):
    chunks = np.array_split(np.arange(n_perm), N_JOBS)
    res = Parallel(n_jobs=N_JOBS)(delayed(_perm_chunk)(feats, y, splits, perm_groups, c) for c in chunks)
    return np.array([v for part in res for v in part])


# ---------------------------------------------------------------- 1つの設定を乱数10通りで
def run_config(config, label, exp, dataset, X, y, meta, kind, n_perm, log):
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    t = time.time()
    folds, nulls, per_seed = [], [], []
    for seed in SEEDS:
        feats = fold_features(kind, seed, X, sp)
        f = eval_cached(feats, y, sp).assign(seed=seed)
        null = perm_cached(feats, y, sp, meta.muniyear.values, n_perm) if n_perm else np.empty((0, 2))
        del feats
        s = summarize(f)
        per_seed.append({'config': config, 'seed': seed, 'bacc_mean': s['bacc_mean'], 'auc_mean': s['auc_mean'],
                         'alpha_at_max': (np.isclose(f.alpha_over_n, BASE_ALPHAS[-1])).mean(),
                         'alpha_at_min': (np.isclose(f.alpha_over_n, BASE_ALPHAS[0])).mean()})
        folds.append(f)
        nulls.append(null)
    folds = pd.concat(folds)
    ps = pd.DataFrame(per_seed)
    null = np.mean(np.stack(nulls), axis=0) if n_perm else np.empty((0, 2))  # 置換 i ごとに10通りの乱数で平均
    r = {'config': config, 'label': label, 'experiment': exp, 'dataset': dataset, 'kind': kind, 'n': len(y),
         'n_high': int(y.sum()), 'n_low': int((1 - y).sum()), 'n_seeds': len(SEEDS), 'n_perm': n_perm}
    for m in ['bacc', 'auc']:
        r[f'{m}_mean'] = ps[f'{m}_mean'].mean()  # 乱数ごとの「20回の繰り返し平均」の、10通りでの平均
        r[f'{m}_seed_min'], r[f'{m}_seed_median'], r[f'{m}_seed_max'] = \
            ps[f'{m}_mean'].min(), ps[f'{m}_mean'].median(), ps[f'{m}_mean'].max()
        rep = folds.groupby(['seed', 'repeat'])[m].mean()
        r[f'{m}_sd_fold'] = folds[m].std()
        r[f'{m}_sd_repeat'] = rep.groupby('repeat').mean().std()
        if n_perm:
            j = 0 if m == 'bacc' else 1
            r[f'{m}_null_mean'] = null[:, j].mean()
            r[f'{m}_null_q95'] = np.quantile(null[:, j], 0.95)
            r[f'{m}_p'] = (1 + np.sum(null[:, j] >= r[f'{m}_mean'])) / (1 + n_perm)
    r['alpha_at_max'] = np.isclose(folds.alpha_over_n, BASE_ALPHAS[-1]).mean()
    r['alpha_at_min'] = np.isclose(folds.alpha_over_n, BASE_ALPHAS[0]).mean()
    r['alpha_over_n_median'] = folds.alpha_over_n.median()
    r['seconds'] = round(time.time() - t, 1)
    if n_perm:
        np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null)
    log(f'  {config:30s} n={len(y)} AUC {r["auc_mean"]:.3f} (乱数 {r["auc_seed_min"]:.3f}〜{r["auc_seed_max"]:.3f}; '
        f'q95 {r.get("auc_null_q95", np.nan):.3f}, p={r.get("auc_p", np.nan):.3f})  bacc {r["bacc_mean"]:.3f} '
        f'(q95 {r.get("bacc_null_q95", np.nan):.3f}, p={r.get("bacc_p", np.nan):.3f})  '
        f'alpha 上限 {r["alpha_at_max"]:.2f}  [{r["seconds"]}s]')
    return r, folds.assign(config=config), ps, split_table(sp, y, meta.muniCd.values).assign(config=config)


# ---------------------------------------------------------------- データ
def label_year_subset(X, m):
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    x = m.merge(lab[['field_id', 'year', 'label_year']], on=['field_id', 'year'], how='left')
    keep = x.label_year.notna().values
    mm = x[keep].reset_index(drop=True).assign(y_muniyear=lambda d: d.y, y=lambda d: d.label_year.astype(int))
    return X[keep], mm




# ---------------------------------------------------------------- 図
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_boxes(summary, folds, per_seed, baselines):
    cfgs = summary[summary.kind != 'baseline'].reset_index(drop=True)
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        data = [folds[folds.config == c].groupby('repeat')[m].mean().values for c in cfgs.config]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                        boxprops=dict(lw=0), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
        for patch, e in zip(bp['boxes'], cfgs.experiment):
            patch.set_facecolor(C_MAIN if e in ('D', 'E') else '#9a9890')
            patch.set_alpha(0.5)
        for i, r in cfgs.iterrows():
            x = i + 1
            pts = per_seed[per_seed.config == r.config][f'{m}_mean'].values
            ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)), pts, s=12, color=C_ALT, zorder=3)
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
            best = baselines[baselines.dataset == r.dataset][f'{m}_mean'].max()
            ax.plot(x + 0.4, best, marker='D', ms=6, color=C_RED, ls='')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylabel(lab, color=INK)
        style(ax)
    axs[1].set_xticks(range(1, len(cfgs) + 1), [c.replace(' ', '\n', 1) for c in cfgs.label], fontsize=8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_MAIN, alpha=0.5), Patch(color='#9a9890', alpha=0.5),
                Line2D([], [], marker='o', color=C_ALT, ls='', ms=4), Line2D([], [], color=INK, lw=2),
                Line2D([], [], marker='D', color=C_RED, ls='')],
               ['実験D・E(結論に使う)', '実験F(探索的)', '乱数ごとの値(10通り)', '置換検定の帰無分布の95%点',
                '同じデータセットの最良の基準線'], loc='upper right', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('設定ごとの性能(箱 = 20回の繰り返し平均。10通りの乱数で平均した値の分布)',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_configs.png'), dpi=150)
    plt.close(fig)


def plot_null_D(summary):
    r = summary[summary.config == 'D_main_diff'].iloc[0]
    null = np.load(os.path.join(OUT_DIR, 'null_D_main_diff.npy'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, j, m, lab in [(axs[0], 1, 'auc', 'AUC(主)'), (axs[1], 0, 'bacc', 'balanced accuracy(副)')]:
        ax.hist(null[:, j], bins=40, color='#9a9890', alpha=0.7)
        ax.axvline(r[f'{m}_null_q95'], color=INK, lw=1.5, ls='--', label=f'95%点 {r[f"{m}_null_q95"]:.3f}')
        ax.axvline(r[f'{m}_mean'], color=C_RED, lw=2, label=f'観測値 {r[f"{m}_mean"]:.3f}(p={r[f"{m}_p"]:.3f})')
        ax.set_xlabel(f'{lab}(10通りの乱数で平均)', color=INK)
        ax.legend(frameon=False, fontsize=9)
        style(ax)
    axs[0].set_ylabel('置換の回数', color=INK)
    fig.suptitle(f'実験D: 置換検定の帰無分布({int(r.n_perm)}回)と観測値', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'expD_null_hist.png'), dpi=150)
    plt.close(fig)


def plot_paired_E(pairs_seedavg, pairs_seed):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, m in zip(axs, ['auc', 'bacc']):
        d = pairs_seedavg[f'd_{m}']
        rep = pairs_seedavg.groupby('repeat')[f'd_{m}'].mean()
        seedm = pairs_seed.groupby('seed')[f'd_{m}'].mean()
        ax.hist(d, bins=np.linspace(-0.5, 0.5, 26), color=C_MAIN, alpha=0.35, label='fold ごとの差(乱数平均、100)')
        ax2 = ax.twinx()
        ax2.hist(rep, bins=np.linspace(-0.5, 0.5, 51), color=C_ALT, alpha=0.9, label='繰り返し平均の差(20)')
        ax2.set_yticks([])
        ax.scatter(seedm, np.zeros(len(seedm)) - 0.6, marker='|', s=200, color=INK, label='乱数ごとの平均の差(10)',
                   clip_on=False, zorder=4)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f'{m}: 平均 {d.mean():+.3f}、繰り返し平均の 2.5〜97.5% '
                     f'[{rep.quantile(.025):+.3f}, {rep.quantile(.975):+.3f}]', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('main_common − dap_common(diff)', color=INK)
        style(ax)
        if m == 'auc':
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('実験E: 揃え方の差(1次差分、同じ分割)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'expE_paired_diff.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm-confirm', type=int, default=1000)
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    global SEEDS
    SEEDS = list(range(args.n_seeds))
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    ds = s5.datasets()
    Xm, mm = ds['main']
    y = mm.y.values

    # Pipeline と特徴量の使い回しの一致(先頭10 fold、乱数0)
    sp = make_splits(y, mm.muniCd.values, N_SPLITS, N_REPEATS, SEED0)[:10]
    a = evaluate(pipeline('minirocket', 0), first_diff(Xm), y, sp)
    b = eval_cached(fold_features('minirocket', 0, first_diff(Xm), sp), y, sp)
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'Pipeline と特徴量の使い回しの一致(先頭10 fold): {ok}')
    assert ok

    # 参考: 段階5の設定の再現(alpha × n_train なし、乱数0)。判定には使わない
    spf = make_splits(y, mm.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    f5 = s5.eval_cached(s5.fold_features('minirocket', first_diff(Xm), spf), y, spf)
    s5r = summarize(f5)
    ref = {'config': 'ref_stage5_main_diff', 'label': '参考 段階5の設定', 'experiment': 'ref', 'dataset': 'main',
           'kind': 'minirocket', 'n': len(y), 'n_seeds': 1, 'n_perm': 0,
           'bacc_mean': s5r['bacc_mean'], 'auc_mean': s5r['auc_mean']}
    log(f'参考: 段階5の設定の再現 AUC {s5r["auc_mean"]:.3f} / bacc {s5r["bacc_mean"]:.3f}(段階5: 0.627 / 0.590)')

    rows, fold_rows, seed_rows, split_rows = [], [], [], []

    def add(res):
        r, f, ps, st = res
        rows.append(r)
        fold_rows.append(f)
        seed_rows.append(ps)
        split_rows.append(st)
        return f

    log(f'\n[実験D] 確認用: main・diff(置換 {args.n_perm_confirm} 回)')
    add(run_config('D_main_diff', 'D main diff', 'D', 'main', first_diff(Xm), y, mm, 'minirocket',
                   args.n_perm_confirm, log))

    log(f'\n[実験E] 揃え方の比較(diff、置換 {args.n_perm} 回)')
    fE = {}
    for dset in ['main_common', 'dap_common']:
        X, m = ds[dset]
        fE[dset] = add(run_config(f'E_{dset}_diff', f'E {dset} diff', 'E', dset, first_diff(X), m.y.values, m,
                                  'minirocket', args.n_perm, log))

    log(f'\n[実験F] 探索的(置換 {args.n_perm} 回)')
    add(run_config('F1_main_raw+diff', 'F1 main raw+diff', 'F1', 'main', Xm.astype(float), y, mm,
                   'raw_diff_concat', args.n_perm, log))
    add(run_config('F2_main_groupcenter_diff', 'F2 main groupcenter→diff', 'F2', 'main',
                   first_diff(s5.groupcenter(Xm, mm)), y, mm, 'minirocket', args.n_perm, log))
    for dset in ['main_obs', 'main_no_kakogawa', 'main_no_irrigation']:
        X, m = ds[dset]
        add(run_config(f'F3_{dset}_diff', f'F3 {dset} diff', 'F3', dset, first_diff(X), m.y.values, m,
                       'minirocket', args.n_perm, log))
    Xl, ml = label_year_subset(Xm, mm)
    ds['main_labelyear'] = (Xl, ml)
    add(run_config('F4_main_labelyear_diff', 'F4 main label_year diff', 'F4', 'main_labelyear', first_diff(Xl),
                   ml.y.values, ml, 'minirocket', args.n_perm, log))

    log(f'\n[基準線](置換 {args.n_perm} 回)')
    bl = []
    for dset in ['main', 'main_common', 'dap_common', 'main_obs', 'main_no_kakogawa', 'main_no_irrigation',
                 'main_labelyear']:
        X, m = ds[dset]
        yy = m.y.values
        spb = make_splits(yy, m.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
        for bname, F in s5.baseline_features(X, m).items():
            folds, obs, null = s5.run_baseline(F, yy, spb, m.muniyear.values, args.n_perm)
            bl.append({'config': f'{bname}_{dset}', 'label': bname, 'experiment': 'baseline', 'dataset': dset,
                       'kind': 'baseline', 'n': len(yy), 'n_perm': args.n_perm, **obs, **s5.null_stats(obs, null)})
            fold_rows.append(folds.assign(config=f'{bname}_{dset}'))
            log(f'  {dset:20s} {bname:16s} AUC {obs["auc_mean"]:.3f}  bacc {obs["bacc_mean"]:.3f}')

    baselines = pd.DataFrame(bl)
    summary = pd.DataFrame(rows + bl + [ref])
    for m in ['bacc', 'auc']:
        best = baselines.groupby('dataset')[f'{m}_mean'].max()
        summary[f'{m}_best_baseline'] = summary.dataset.map(best)
        hit = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > summary[f'{m}_best_baseline'])
        summary[f'{m}_hit'] = hit.astype(object).where(~summary.kind.isin(['baseline']) & (summary.n_perm > 0), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat(fold_rows)
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed = pd.concat(seed_rows)
    per_seed.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat(split_rows).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')

    # 実験E: fold ごとの差(同じ件・同じ順番なので分割も同じ)
    mc, dc = ds['main_common'][1], ds['dap_common'][1]
    assert (mc[['field_id', 'year']].values == dc[['field_id', 'year']].values).all()
    a, b = fE['main_common'], fE['dap_common']
    p = a.merge(b, on=['seed', 'repeat', 'fold'], suffixes=('_main', '_dap'))
    p['d_auc'], p['d_bacc'] = p.auc_main - p.auc_dap, p.bacc_main - p.bacc_dap
    p[['seed', 'repeat', 'fold', 'd_auc', 'd_bacc']].to_csv(os.path.join(OUT_DIR, 'expE_paired_folds.csv'),
                                                           index=False, encoding='utf-8-sig')
    pa = p.groupby(['repeat', 'fold'])[['d_auc', 'd_bacc']].mean().reset_index()  # 10通りの乱数で平均した差
    pe = []
    for m in ['auc', 'bacc']:
        rep = pa.groupby('repeat')[f'd_{m}'].mean()
        seedm = p.groupby('seed')[f'd_{m}'].mean()
        seed_detect = p.groupby('seed').apply(
            lambda s: max((s.groupby('repeat')[f'd_{m}'].mean() > 0).mean(), (s.groupby('repeat')[f'd_{m}'].mean() < 0).mean()) >= 0.95)
        pe.append({'metric': m, 'mean_diff': pa[f'd_{m}'].mean(), 'sd_fold': pa[f'd_{m}'].std(),
                   'rep_q025': rep.quantile(.025), 'rep_q975': rep.quantile(.975),
                   'rep_frac_pos': (rep > 0).mean(), 'rep_frac_neg': (rep < 0).mean(),
                   'detected': bool(max((rep > 0).mean(), (rep < 0).mean()) >= 0.95),
                   'n_seeds_positive': int((seedm > 0).sum()), 'n_seeds_negative': int((seedm < 0).sum()),
                   'n_seeds_detected_alone': int(seed_detect.sum())})
    pe = pd.DataFrame(pe)
    pe.to_csv(os.path.join(OUT_DIR, 'expE_paired_summary.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds, per_seed, baselines)
    plot_null_D(summary)
    plot_paired_E(pa, p)

    show = ['config', 'n', 'auc_mean', 'auc_seed_min', 'auc_seed_median', 'auc_seed_max', 'auc_null_mean',
            'auc_null_q95', 'auc_p', 'auc_best_baseline', 'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p',
            'bacc_best_baseline', 'bacc_hit', 'alpha_at_max', 'alpha_at_min']
    log('\n[まとめ]')
    log(summary[summary.kind != 'baseline'].reindex(columns=show).round(3).to_string(index=False))
    log('\n[実験E fold ごとの差]')
    log(pe.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
