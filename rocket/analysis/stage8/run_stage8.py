"""
段階8: Rocket に前提どおりの大きさの入力を入れて比べ直す(すべて探索的。主な設定 MiniRocket・1次差分は変えない)。

  入力(どれも main 82件の1次差分、59点)
    R1 論文どおり:       sktime の Rocket(normalise=True)。Rocket の中で系列ごとに z 正規化される
    R2 全体での尺度合わせ: fold の学習データの1次差分すべての値の標準偏差 s(ddof=0)で、学習・テストの系列を割る。
                          平均は引かない。Rocket は normalise=False。s は Pipeline の中で fold ごとに学習データだけで決める
    参考(計算し直さない): R0 = 段階7の Rocket(normalise=False、1次差分)、段階6の実験D の MiniRocket(1次差分)
  特徴量: Rocket の1回の出力から PPV のみ(2j 列目、10000列)と PPV + max(20000列)
  モデル・評価: 段階6の実験Dと同じ(Rocket num_kernels=10000・random_state=0〜9 → StandardScaler(with_mean=False)
               → RidgeClassifierCV(alphas × n_train)、同じ分割)。置換200回(市町村×年の中、置換 i は乱数 [0, i])

本番の前の確認(run_log.txt に出す。想定と違えば、該当する比較を「記述だけ」に切り替える):
  - sktime の版が 1.2.0(列の並びを確かめた EXP-010 と同じ)
  - カーネルが random_state と系列の長さだけで決まり、normalise や入力の値によらない(R0・R1・R2 で同じカーネル)
  - PPV の張り付き(診断): R0・R1・R2 で、乱数0・最初の fold の学習データについて、全系列で PPV が同じ値になるカーネルの割合
  - 段階6と同じ分割、Pipeline と特徴量の使い回しの一致、高速版の置換検定と元の方法の一致

実行(別の PC で。outputs/stage6/ と outputs/stage7/pooling/ が必要):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage8\\run_stage8.py
  試運転: ... run_stage8.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage8/
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
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage7'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
from src.evaluation import evaluate, make_splits  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage8')
POOL_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage7', 'pooling')
N_KERNELS = 10000
SKTIME_VERSION = '1.2.0'  # EXP-010 で列の並び(2j = PPV、2j+1 = max)を確かめた版
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 入力の名前 → Rocket の normalise と、全体での尺度合わせをするか
INPUTS = {'R1': {'normalise': True, 'global_scale': False},
          'R2': {'normalise': False, 'global_scale': True}}


# ---------------------------------------------------------------- 変換
class GlobalScale(BaseEstimator, TransformerMixin):
    """学習データのすべての値の標準偏差 s(ddof=0)で割る。平均は引かない。"""

    def fit(self, X, y=None):
        self.s_ = float(np.asarray(X, float).std())
        return self

    def transform(self, X):
        return np.asarray(X, float) / self.s_


class RocketPool(BaseEstimator, TransformerMixin):
    """sktime の Rocket。pool='ppv' なら PPV の列(2j)だけ、'ppvmax' なら全列。"""

    def __init__(self, pool='ppvmax', normalise=False, num_kernels=N_KERNELS, random_state=0):
        self.pool, self.normalise, self.num_kernels, self.random_state = pool, normalise, num_kernels, random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import Rocket
        self.t_ = Rocket(num_kernels=self.num_kernels, normalise=self.normalise,
                         random_state=self.random_state).fit(np.asarray(X, float))
        return self

    def transform(self, X):
        F = np.asarray(self.t_.transform(np.asarray(X, float)), dtype=float)
        return F[:, 0::2] if self.pool == 'ppv' else F


def make_transform(inp, pool, seed):
    """入力 R1 / R2 の、Rocket と StandardScaler までの Pipeline。"""
    cfg = INPUTS[inp]
    steps = [GlobalScale()] if cfg['global_scale'] else []
    steps += [RocketPool(pool, normalise=cfg['normalise'], random_state=seed), StandardScaler(with_mean=False)]
    return make_pipeline(*steps)


def _fold(inp, pool, seed, X, tr, te):
    tf = make_transform(inp, pool, seed).fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[te])


_ORIGINAL_FOLD_FEATURES = s6.fold_features


def fold_features(kind, seed, X, splits):
    """s6.fold_features の差し替え。kind は 'R1_ppv' / 'R1_ppvmax' / 'R2_ppv' / 'R2_ppvmax'。"""
    if kind.split('_')[0] not in INPUTS:
        return _ORIGINAL_FOLD_FEATURES(kind, seed, X, splits)
    inp, pool = kind.split('_')
    return Parallel(n_jobs=s6.N_JOBS)(delayed(_fold)(inp, pool, seed, X, tr, te) for r, f, tr, te in splits)


# ---------------------------------------------------------------- 本番の前の確認
def check_kernels(X, seeds=(0, 1, 9)):
    """カーネルが random_state と系列の長さだけで決まるか。normalise の有無、入力の値(生の1次差分・s で割ったもの・
    乱数の系列)を変えて fit し、カーネルのすべての配列(重み・長さ・bias・dilation・padding・チャンネル)を比べる。"""
    from sktime.transformations.panel.rocket import Rocket
    rng = np.random.default_rng(0)
    variants = {'R0 (normalise=False, 1次差分)': (False, X),
                'R1 (normalise=True, 1次差分)': (True, X),
                'R2 (normalise=False, 1次差分 / s)': (False, X / X.std()),
                '対照 (normalise=False, 正規乱数の系列)': (False, rng.normal(size=X.shape))}
    rows = []
    for seed in seeds:
        ref = None
        for name, (norm, Z) in variants.items():
            k = Rocket(num_kernels=N_KERNELS, normalise=norm, random_state=seed).fit(Z).kernels
            if ref is None:
                ref = k
            same = all(np.array_equal(a, b) for a, b in zip(ref, k))
            rows.append({'seed': seed, 'variant': name, 'same_as_R0': same})
    df = pd.DataFrame(rows)
    return bool(df.same_as_R0.all()), df


def ppv_saturation(X, y, meta, seed=0):
    """乱数0・最初の fold の学習データで、全系列で PPV が同じ値になるカーネルの割合(R0・R1・R2)。"""
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    tr = sp[0][2]
    Xtr = X[tr]
    feats = {'R0': RocketPool('ppv', normalise=False, random_state=seed).fit(Xtr).transform(Xtr),
             'R1': RocketPool('ppv', normalise=True, random_state=seed).fit(Xtr).transform(Xtr)}
    g = GlobalScale().fit(Xtr)
    Z = g.transform(Xtr)
    feats['R2'] = RocketPool('ppv', normalise=False, random_state=seed).fit(Z).transform(Z)
    rows = []
    for name, F in feats.items():
        const = (F == F[:1]).all(axis=0)
        rows.append({'input': name, 'n_train': len(tr), 'frac_constant': const.mean(),
                     'frac_all_zero': (F == 0).all(axis=0).mean(), 'frac_all_one': (F == 1).all(axis=0).mean(),
                     'median_ppv_sd': float(np.median(F.std(axis=0)))})
    return pd.DataFrame(rows), g.s_


def check_pipeline(X, y, meta, n=10):
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)[:n]
    res = {}
    for inp in INPUTS:
        for pool in ['ppv', 'ppvmax']:
            pipe = make_transform(inp, pool, 0)
            pipe.steps.append(('ridge', s6.RidgeNTrain()))
            a = evaluate(pipe, X, y, sp)
            b = s6.eval_cached(fold_features(f'{inp}_{pool}', 0, X, sp), y, sp)
            res[f'{inp}_{pool}'] = bool(np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True))
    return res


# ---------------------------------------------------------------- 保存済みの参考(R0、MiniRocket)
def load_references():
    s = pd.read_csv(os.path.join(POOL_DIR, 'summary.csv'), encoding='utf-8-sig')
    f = pd.read_csv(os.path.join(POOL_DIR, 'folds.csv'), encoding='utf-8-sig')
    p = pd.read_csv(os.path.join(POOL_DIR, 'per_seed.csv'), encoding='utf-8-sig')
    rows, folds, seeds = [], [], []
    for old, new in [('H_diff_ppv', 'R0_ppv'), ('H_diff_ppvmax', 'R0_ppvmax')]:
        r = s[s.config == old].iloc[0].to_dict()
        assert r['n_seeds'] == 10, f'段階7の {old} が本番の結果ではない'
        rows.append({**r, 'config': new, 'label': f'{new}(段階7、計算し直していない)', 'experiment': 'R0(段階7)'})
        folds.append(f[f.config == old].astype({'seed': int}).assign(config=new))
        seeds.append(p[p.config == old].assign(config=new))
    D_folds, D_seed, D_row, baselines, _ = s7.load_stage6_D()
    rows.append({**D_row, 'config': 'MR_diff', 'label': 'MiniRocket 1次差分(段階6 実験D)', 'experiment': 'MR(段階6)'})
    folds.append(D_folds.assign(config='MR_diff'))
    seeds.append(D_seed.assign(config='MR_diff'))
    return rows, pd.concat(folds), pd.concat(seeds), baselines


# ---------------------------------------------------------------- 図
NAMES = {'R0_ppv': 'R0\nPPV', 'R0_ppvmax': 'R0\nPPV+max', 'R1_ppv': 'R1\nPPV', 'R1_ppvmax': 'R1\nPPV+max',
         'R2_ppv': 'R2\nPPV', 'R2_ppvmax': 'R2\nPPV+max', 'MR_diff': 'MiniRocket\n(参考)'}


def plot_boxes(summary, folds_all, per_seed_all, best_bl):
    order = [c for c in NAMES if c in set(summary.config)]
    color = {'ppv': '#2a78d6', 'ppvmax': '#eb6834', 'diff': '#9a9890'}
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        for x, cfg in enumerate(order, start=1):
            f = folds_all[folds_all.config == cfg]
            rep = f.groupby(['seed', 'repeat'])[m].mean().groupby('repeat').mean().values
            bp = ax.boxplot([rep], positions=[x], widths=0.5, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5), boxprops=dict(lw=0),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
            bp['boxes'][0].set_facecolor(color[cfg.split('_')[-1]])
            bp['boxes'][0].set_alpha(0.45)
            pts = per_seed_all[per_seed_all.config == cfg][f'{m}_mean'].values
            ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)), pts, s=12, color=INK, zorder=3)
            r = summary[summary.config == cfg].iloc[0]
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
            ax.plot(x + 0.4, best_bl[m], marker='D', ms=6, color='#e34948', ls='')
        for xv in (2.5, 4.5, 6.5):
            ax.axvline(xv, color=MUTED, lw=0.8)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylabel(lab, color=INK)
        s6.style(ax)
    axs[1].set_xticks(range(1, len(order) + 1), [NAMES[c] for c in order], fontsize=9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=color['ppv'], alpha=0.45), Patch(color=color['ppvmax'], alpha=0.45),
                Patch(color=color['diff'], alpha=0.45), Line2D([], [], marker='o', color=INK, ls='', ms=4),
                Line2D([], [], color=INK, lw=2), Line2D([], [], marker='D', color='#e34948', ls='')],
               ['PPV のみ', 'PPV + max', 'MiniRocket(参考)', '乱数ごとの値(10通り)', '置換検定の95%点',
                '最良の基準線(段階6の main)'],
               loc='upper left', bbox_to_anchor=(0.02, 0.955), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Rocket の入力の大きさの比較(main 82件・1次差分)。R0 = normalise なし、R1 = 系列ごとの z 正規化、'
                 'R2 = 全体の標準偏差で割る', fontsize=11, color=INK, x=0.02, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_stage8.png'), dpi=150)
    plt.close(fig)


def plot_paired(pairs, ps, title, fname):
    comps = list(pairs.comparison.unique())
    if not comps:
        return
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
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=200)
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
    X = first_diff(Xm)

    # ---- 本番の前の確認
    import sktime
    log(f'sktime の版: {sktime.__version__}(EXP-010 と同じ {SKTIME_VERSION} か: {sktime.__version__ == SKTIME_VERSION})')
    assert sktime.__version__ == SKTIME_VERSION, '列の並びを確かめた版と違う'

    kernels_same, kdf = check_kernels(X)
    kdf.to_csv(os.path.join(OUT_DIR, 'check_kernels.csv'), index=False, encoding='utf-8-sig')
    log(f'カーネルが R0・R1・R2(と正規乱数の系列)で同じか(乱数 0・1・9): {kernels_same}')
    if not kernels_same:
        log('  → 想定と違う。入力の大きさの効果(R1 − R0 など)は fold ごとの差を取らず、記述だけにする')
        log(kdf.to_string(index=False))

    sat, s0 = ppv_saturation(X, y, mm)
    sat.to_csv(os.path.join(OUT_DIR, 'ppv_saturation_seed0_fold0.csv'), index=False, encoding='utf-8-sig')
    log(f'PPV の張り付き(乱数0・最初の fold の学習データ {int(sat.n_train.iloc[0])} 系列。R2 の s = {s0:.4f}):')
    log(sat.round(3).to_string(index=False))

    assert s7.check_same_splits(y, mm), '段階6の実験Dと分割が違う'
    log('段階6の実験Dと同じ分割: True')
    res = check_pipeline(X, y, mm)
    log(f'Pipeline と特徴量の使い回しの一致(先頭10 fold、乱数0): {res}')
    assert all(res.values())
    sp = make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    feats = fold_features('R2_ppvmax', 0, X, sp)
    a = original_perm_cached(feats, y, sp, mm.muniyear.values, 20)
    b = fast.perm_cached(feats, y, sp, mm.muniyear.values, 20)
    ok = np.array_equal(a, b, equal_nan=True)
    log(f'高速版と元の perm_cached の一致(R2・PPV+max、乱数0、置換20回): {ok}')
    assert ok
    s6.perm_cached = fast.perm_cached

    # ---- 本番
    rows, fold_rows, seed_rows = [], [], []
    for inp in INPUTS:
        for pool in ['ppvmax', 'ppv']:
            cfg = f'{inp}_{pool}'
            r, f, ps, _ = s6.run_config(cfg, cfg, inp, 'main', X, y, mm, cfg, args.n_perm, log)
            rows.append(r)
            fold_rows.append(f)
            seed_rows.append(ps)

    ref_rows, ref_folds, ref_seed, baselines = load_references()
    best_bl = {m: baselines[f'{m}_mean'].max() for m in ['auc', 'bacc']}
    summary = pd.DataFrame(rows + ref_rows)
    for m in ['bacc', 'auc']:
        summary[f'{m}_best_baseline'] = best_bl[m]
        summary[f'{m}_hit'] = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > best_bl[m])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds_all = pd.concat(fold_rows + [ref_folds])
    folds_all.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed_all = pd.concat(seed_rows + [ref_seed])
    per_seed_all.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')

    # ---- 比較(同じ分割・同じ乱数の fold ごとの差)
    fz = lambda c: folds_all[(folds_all.config == c) & folds_all.seed.isin(s6.SEEDS)]  # noqa: E731
    max_pairs = [s7.paired(fz(f'{i}_ppvmax'), fz(f'{i}_ppv'), f'max の効果 {i}: PPV+max − PPV') for i in INPUTS]
    size_pairs = []
    if kernels_same:
        size_pairs = [s7.paired(fz('R1_ppv'), fz('R0_ppv'), '入力の大きさ PPV: R1 − R0'),
                      s7.paired(fz('R2_ppv'), fz('R0_ppv'), '入力の大きさ PPV: R2 − R0'),
                      s7.paired(fz('R2_ppv'), fz('R1_ppv'), '入力の大きさ PPV: R2 − R1')]
    ref_pairs = [s7.paired(fz(f'{i}_ppv'), fz('MR_diff'), f'参考 {i} PPV − MiniRocket') for i in INPUTS]
    pairs = pd.concat(max_pairs + size_pairs + ref_pairs)
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = s7.paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds_all, per_seed_all, best_bl)
    plot_paired(pd.concat(max_pairs), ps, 'max の効果(PPV+max − PPV のみ、同じカーネル・同じ分割・同じ乱数)',
                'paired_diff_max.png')
    if size_pairs:
        plot_paired(pd.concat(size_pairs), ps, '入力の大きさの効果(PPV のみ、同じカーネル・同じ分割・同じ乱数)',
                    'paired_diff_input_size.png')

    show = ['config', 'n', 'n_perm', 'auc_mean', 'auc_seed_min', 'auc_seed_median', 'auc_seed_max', 'auc_null_mean',
            'auc_null_q95', 'auc_p', 'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p', 'bacc_hit',
            'alpha_at_max', 'alpha_at_min']
    log('\n[まとめ](R0 は段階7、MR_diff は段階6の保存済みの結果。置換の回数は n_perm の列)')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log('\n[fold ごとの差(10通りの乱数で平均)]')
    log(ps.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
