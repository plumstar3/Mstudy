"""
段階7: 入力の形の比較を、段階6の実験Dと同じ条件でやり直す(記述的な比較。主な設定の1次差分は変えない)。

  G1 raw    生の値(60点)
  G2 znorm  系列ごとの z 正規化(60点)
  1次差分(段階6の実験D)は outputs/stage6/ の保存済みの結果を使い、計算し直さない

条件は段階6の実験Dと同じ:
  main(82件)、sktime 1.2.0 の MiniRocket(num_kernels=10000、random_state=0〜9)→ StandardScaler(with_mean=False)
  → RidgeClassifierCV(alphas = logspace(−3, 3, 10) × n_train)、StratifiedGroupKFold(groups=muniCd)5分割×20回、
  統計量は乱数ごとの「20回の繰り返し平均」の10通りでの平均、置換1000回(市町村×年の中で並べ替え、置換 i は乱数 [0, i])
置換検定は run_stage6_local.py の高速版を使う。起動時に、生の値の入力でも元の方法と完全に一致することを確かめる。

比較: diff − raw、diff − znorm、znorm − raw の fold ごとの差(同じ分割・同じ乱数)
補足: 生の値を「学習データ全体の平均・標準偏差」で標準化した場合(乱数0だけ、置換なし)と、生の値との差

実行(別の PC で。段階6の出力 outputs/stage6/ が必要):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage7\\run_stage7.py
  試運転: ... run_stage7.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage7/
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
from src.evaluation import evaluate, make_splits  # noqa: E402
from src.windows import znorm  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage7')
S6_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage6')
D_CONFIG = 'D_main_diff'
INK, MUTED, GRID = s6.INK, s6.MUTED, s6.GRID
COLORS = {'diff': '#2a78d6', 'raw': '#eb6834', 'znorm': '#1baf7a'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 確認
def check_fast_vs_original(X, y, meta, original_perm_cached, n_perm=20):
    """生の値の入力・乱数0で、元の perm_cached と高速版の null が完全に一致するか。"""
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    feats = s6.fold_features('minirocket', 0, X, sp)
    t = time.time()
    a = original_perm_cached(feats, y, sp, meta.muniyear.values, n_perm)
    ta = time.time() - t
    t = time.time()
    b = fast.perm_cached(feats, y, sp, meta.muniyear.values, n_perm)
    tb = time.time() - t
    ok = np.array_equal(a, b, equal_nan=True)
    return ok, float(np.nanmax(np.abs(a - b))), ta, tb


def check_pipeline(X, y, meta, n=10):
    """Pipeline をそのまま回した結果と、特徴量の使い回しの結果が一致するか(先頭 n fold、乱数0)。"""
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)[:n]
    a = evaluate(s6.pipeline('minirocket', 0), X, y, sp)
    b = s6.eval_cached(s6.fold_features('minirocket', 0, X, sp), y, sp)
    return np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)


def check_same_splits(y, meta):
    """段階6の実験Dと同じ分割か(fold ごとのテスト件数・多収数を splits.csv と比べる)。"""
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    mine = s6.split_table(sp, y, meta.muniCd.values)
    s6sp = pd.read_csv(os.path.join(S6_DIR, 'splits.csv'), encoding='utf-8-sig')
    s6sp = s6sp[s6sp.config == D_CONFIG].reset_index(drop=True)
    cols = ['repeat', 'fold', 'n_train', 'n_test', 'n_test_pos', 'n_test_groups']
    return mine[cols].reset_index(drop=True).equals(s6sp[cols])


# ---------------------------------------------------------------- 補足: 学習データ全体の平均・標準偏差で標準化
def _global_std_fold(X, y, tr, te, seed):
    mu, sd = X[tr].mean(), X[tr].std()
    Xg = (X - mu) / sd
    out = {}
    for name, Z in [('raw', X), ('global_std', Xg)]:
        tf = s6.Panel('minirocket', random_state=seed).fit(Z[tr])
        Ftr, Fte = tf.transform(Z[tr]), tf.transform(Z[te])
        sc = StandardScaler(with_mean=False).fit(Ftr)
        clf = s6.RidgeNTrain().fit(sc.transform(Ftr), y[tr])
        d = clf.decision_function(sc.transform(Fte))
        out[name] = {'F': np.vstack([Ftr, Fte]), 'auc': roc_auc_score(y[te], d) if len(np.unique(y[te])) == 2 else np.nan,
                     'bacc': balanced_accuracy_score(y[te], (d > 0).astype(int)), 'alpha': clf.alpha_}
    return {'mu': mu, 'sd': sd,
            'max_abs_diff_features': float(np.abs(out['raw']['F'] - out['global_std']['F']).max()),
            'n_feature_values_diff': int((out['raw']['F'] != out['global_std']['F']).sum()),
            'auc_raw': out['raw']['auc'], 'auc_global_std': out['global_std']['auc'],
            'bacc_raw': out['raw']['bacc'], 'bacc_global_std': out['global_std']['bacc'],
            'same_alpha': bool(np.isclose(out['raw']['alpha'], out['global_std']['alpha']))}


def global_std_check(X, y, meta, seed=0):
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    res = Parallel(n_jobs=s6.N_JOBS)(delayed(_global_std_fold)(X, y, tr, te, seed) for r, f, tr, te in sp)
    df = pd.DataFrame([{'repeat': r, 'fold': f, **x} for (r, f, tr, te), x in zip(sp, res)])
    df['d_auc'] = df.auc_global_std - df.auc_raw
    df['d_bacc'] = df.bacc_global_std - df.bacc_raw
    return df


# ---------------------------------------------------------------- 段階6の実験D(1次差分)を読み込む
def load_stage6_D():
    folds = pd.read_csv(os.path.join(S6_DIR, 'folds.csv'), encoding='utf-8-sig')
    folds = folds[folds.config == D_CONFIG].copy()
    folds = folds.astype({'seed': int, 'n_train': int})  # 基準線の行と同じ CSV に入っていたため小数になっている
    per_seed = pd.read_csv(os.path.join(S6_DIR, 'per_seed.csv'), encoding='utf-8-sig')
    per_seed = per_seed[per_seed.config == D_CONFIG].copy()
    summary = pd.read_csv(os.path.join(S6_DIR, 'summary.csv'), encoding='utf-8-sig')
    row = summary[summary.config == D_CONFIG].iloc[0].to_dict()
    baselines = summary[(summary.kind == 'baseline') & (summary.dataset == 'main')].copy()
    null = np.load(os.path.join(S6_DIR, f'null_{D_CONFIG}.npy'))
    assert len(null) == 1000 and row['n_perm'] == 1000 and row['n_seeds'] == 10, '段階6の本番の結果ではない'
    return folds, per_seed, row, baselines, null


# ---------------------------------------------------------------- 対の差
def paired(fa, fb, name):
    """fa − fb を、乱数・繰り返し・fold ごとに。"""
    m = fa.merge(fb, on=['seed', 'repeat', 'fold'], suffixes=('_a', '_b'))
    assert len(m) == len(fa) == len(fb), f'{name}: 乱数・繰り返し・fold が揃っていない'
    return pd.DataFrame({'comparison': name, 'seed': m.seed, 'repeat': m.repeat, 'fold': m.fold,
                         'd_auc': m.auc_a - m.auc_b, 'd_bacc': m.bacc_a - m.bacc_b})


def paired_summary(p):
    rows = []
    for c, s in p.groupby('comparison', sort=False):
        avg = s.groupby(['repeat', 'fold'])[['d_auc', 'd_bacc']].mean().reset_index()  # 10通りの乱数で平均した差
        for m in ['auc', 'bacc']:
            rep = avg.groupby('repeat')[f'd_{m}'].mean()
            seedm = s.groupby('seed')[f'd_{m}'].mean()
            seed_rep = s.groupby(['seed', 'repeat'])[f'd_{m}'].mean().unstack('repeat')
            seed_detect = ((seed_rep > 0).mean(axis=1) >= 0.95) | ((seed_rep < 0).mean(axis=1) >= 0.95)
            rows.append({'comparison': c, 'metric': m, 'mean_diff': avg[f'd_{m}'].mean(), 'sd_fold': avg[f'd_{m}'].std(),
                         'rep_q025': rep.quantile(.025), 'rep_q975': rep.quantile(.975),
                         'rep_frac_pos': (rep > 0).mean(), 'rep_frac_neg': (rep < 0).mean(),
                         'detected': bool(max((rep > 0).mean(), (rep < 0).mean()) >= 0.95),
                         'n_seeds_positive': int((seedm > 0).sum()), 'n_seeds_negative': int((seedm < 0).sum()),
                         'n_seeds_detected_alone': int(seed_detect.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 図
def plot_boxes(summary, folds_all, per_seed_all, best_bl):
    order = [('diff', D_CONFIG, '1次差分(段階6 実験D)'), ('raw', 'G1_main_raw', 'G1 生の値'),
             ('znorm', 'G2_main_znorm', 'G2 z 正規化')]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        for i, (k, cfg, name) in enumerate(order):
            x = i + 1
            f = folds_all[folds_all.config == cfg]
            rep = f.groupby(['seed', 'repeat'])[m].mean().groupby('repeat').mean().values
            bp = ax.boxplot([rep], positions=[x], widths=0.5, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5), boxprops=dict(lw=0),
                            whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
            bp['boxes'][0].set_facecolor(COLORS[k])
            bp['boxes'][0].set_alpha(0.45)
            pts = per_seed_all[per_seed_all.config == cfg][f'{m}_mean'].values
            ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)), pts, s=14, color=INK, zorder=3)
            r = summary[summary.config == cfg].iloc[0]
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
            ax.plot(x + 0.4, best_bl[m], marker='D', ms=6, color='#e34948', ls='')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks([1, 2, 3], [n for _, _, n in order], fontsize=9)
        ax.set_ylabel(f'{lab}(20回の繰り返し平均)', color=INK)
        s6.style(ax)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker='o', color=INK, ls='', ms=4), Line2D([], [], color=INK, lw=2),
                Line2D([], [], marker='D', color='#e34948', ls='')],
               ['乱数ごとの値(10通り)', '置換検定の帰無分布の95%点(1000回)', '最良の基準線(段階6の main)'],
               loc='upper left', bbox_to_anchor=(0.02, 0.93), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('入力の形の比較(main 82件、箱 = 10通りの乱数で平均した20回の繰り返し平均)',
                 fontsize=13, color=INK, x=0.02, ha='left', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_inputs.png'), dpi=150)
    plt.close(fig)


def plot_paired(pairs, ps):
    comps = list(pairs.comparison.unique())
    fig, axs = plt.subplots(len(comps), 2, figsize=(12, 3.5 * len(comps)), squeeze=False)
    for i, c in enumerate(comps):
        s = pairs[pairs.comparison == c]
        avg = s.groupby(['repeat', 'fold'])[['d_auc', 'd_bacc']].mean().reset_index()
        for j, m in enumerate(['auc', 'bacc']):
            ax = axs[i, j]
            rep = avg.groupby('repeat')[f'd_{m}'].mean()
            seedm = s.groupby('seed')[f'd_{m}'].mean()
            ax.hist(avg[f'd_{m}'], bins=np.linspace(-0.5, 0.5, 26), color='#2a78d6', alpha=0.35,
                    label='fold ごとの差(乱数平均、100)')
            ax2 = ax.twinx()
            ax2.hist(rep, bins=np.linspace(-0.5, 0.5, 51), color='#eb6834', alpha=0.9, label='繰り返し平均の差(20)')
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
    fig.suptitle('入力の形の差(同じ分割・同じ乱数)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'paired_diff_inputs.png'), dpi=150)
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
    # 段階6の関数を、この段階の出力先・並列数・高速版の置換検定で使う
    original_perm_cached = s6.perm_cached
    s6.OUT_DIR = OUT_DIR
    s6.N_JOBS = fast.N_JOBS
    s6.SEEDS = list(range(args.n_seeds))
    log(f'並列数 {fast.N_JOBS}、乱数 {s6.SEEDS[0]}〜{s6.SEEDS[-1]}、置換 {args.n_perm} 回')

    Xm, mm = s6.s5.datasets()['main']
    y = mm.y.values
    X_raw, X_z = Xm.astype(float), znorm(Xm)
    D_folds, D_seed, D_row, baselines, D_null = load_stage6_D()
    best_bl = {m: baselines[f'{m}_mean'].max() for m in ['auc', 'bacc']}
    log(f'段階6の実験D(1次差分)を読み込んだ: AUC {D_row["auc_mean"]:.3f}、bacc {D_row["bacc_mean"]:.3f}'
        f'(置換 {int(D_row["n_perm"])} 回)。最良の基準線(main): AUC {best_bl["auc"]:.3f}、bacc {best_bl["bacc"]:.3f}')

    assert check_same_splits(y, mm), '段階6の実験Dと分割が違う'
    log('段階6の実験Dと同じ分割: True')
    ok = check_pipeline(X_raw, y, mm)
    log(f'Pipeline と特徴量の使い回しの一致(生の値、先頭10 fold): {ok}')
    assert ok
    ok, mx, ta, tb = check_fast_vs_original(X_raw, y, mm, original_perm_cached)
    log(f'高速版と元の perm_cached の一致(生の値、乱数0、置換20回): {ok}  最大差 {mx:.1e}  元 {ta:.1f}s / 新 {tb:.1f}s')
    assert ok
    s6.perm_cached = fast.perm_cached

    rows, fold_rows, seed_rows = [], [], []
    for cfg, label, X in [('G1_main_raw', 'G1 raw', X_raw), ('G2_main_znorm', 'G2 znorm', X_z)]:
        r, f, ps, _ = s6.run_config(cfg, label, 'G', 'main', X, y, mm, 'minirocket', args.n_perm, log)
        rows.append(r)
        fold_rows.append(f)
        seed_rows.append(ps)

    # 段階6の実験D を同じ表に入れる(計算し直していない)
    D_row = {**D_row, 'experiment': 'D(段階6)', 'label': 'diff(段階6 実験D)'}
    summary = pd.DataFrame([D_row] + rows)
    for m in ['bacc', 'auc']:
        summary[f'{m}_best_baseline'] = best_bl[m]
        summary[f'{m}_hit'] = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > best_bl[m])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds_all = pd.concat([D_folds] + fold_rows)
    folds_all.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed_all = pd.concat([D_seed] + seed_rows)
    per_seed_all.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')

    # 対の差(同じ分割・同じ乱数)
    fd = D_folds[D_folds.seed.isin(s6.SEEDS)]
    fr = fold_rows[0]
    fz = fold_rows[1]
    pairs = pd.concat([paired(fd, fr, 'diff − raw'), paired(fd, fz, 'diff − znorm'), paired(fz, fr, 'znorm − raw')])
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    # 補足: 学習データ全体の平均・標準偏差による標準化(乱数0、置換なし)
    g = global_std_check(X_raw, y, mm, seed=0)
    g.to_csv(os.path.join(OUT_DIR, 'global_std_check.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds_all, per_seed_all, best_bl)
    plot_paired(pairs, ps)

    show = ['config', 'n', 'auc_mean', 'auc_seed_min', 'auc_seed_median', 'auc_seed_max', 'auc_null_mean',
            'auc_null_q95', 'auc_p', 'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p', 'bacc_hit',
            'alpha_at_max', 'alpha_at_min']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log('\n[fold ごとの差(10通りの乱数で平均)]')
    log(ps.round(3).to_string(index=False))
    log('\n[補足: 学習データ全体の平均・標準偏差で標準化した生の値 − 生の値(乱数0)]')
    log(f'  特徴量の最大差 {g.max_abs_diff_features.max():.2e}、値が違う特徴量の数(全 fold 合計) {g.n_feature_values_diff.sum()}')
    log(f'  AUC の差の最大 {g.d_auc.abs().max():.2e}、bacc の差の最大 {g.d_bacc.abs().max():.2e}、'
        f'alpha が同じ fold の割合 {g.same_alpha.mean():.2f}')
    log(f'  AUC 平均: 生の値 {g.auc_raw.mean():.4f} / 標準化 {g.auc_global_std.mean():.4f}')
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
