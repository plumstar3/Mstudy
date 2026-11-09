"""
段階11: label_year による MiniRocket の評価(EXP-020)。

label_muniyear の主データ(82件)より件数の多い label_year(年の中での相対評価)で、主な設定
(MiniRocket・1次差分)を評価する。label_muniyear との比較は目的に含めない。label_year は地域差を除けないので、
結果は「地域差も含めて、VWC から年内の多収/低収を当てられるか」として読む。

  M   主の解析: 1次差分(59点)→ MiniRocket(sktime 1.2.0、num_kernels=10000、乱数0〜9)
      → StandardScaler(with_mean=False) → RidgeClassifierCV(alphas = logspace(−3, 3, 10) × n_train)。
      段階6の実験Dと同じ構成。置換1000回
  B1  基準線①: 窓内の生の VWC の要約統計(平均・標準偏差(ddof=1)・最小・最大)→ StandardScaler
      → LogisticRegression(C=1, max_iter=1000)。置換1000回
  B2  基準線②: 緯度・経度・播種日の通日(seed_doy)だけ → B1 と同じモデル。置換1000回
  S   感度分析: 実測開花日の件だけ(flower_source == 'obs')で M・B1・B2。分割は作り直す。置換200回

データ: data/processed/field_year_labeled.csv の label_year 付き202件のうち、開花起点 0〜+59 日(60点)の窓の
すべての日に観測がある件。開花日は同じ CSV の flower_date(実測を優先し、無い件は EXP-016 の予測)。
src/windows.py の load_cases() はルートの DB の flower_date を merge するので使わず、読み込みはここで書く。

評価: src/evaluation.make_splits(groups=muniCd、5分割×20回)。統計量は乱数ごとの「20回の繰り返し平均」の
10通りでの平均。置換検定はラベルを年の中で並べ替える(置換 i の乱数は [0, i])。年の中で並べ替えると地域ごとの
クラス数が変わり、StratifiedGroupKFold の分割は置換前と同じにならないが、特徴量を使い回すため観測ラベルで
作った分割を置換でも固定して使う。置換の高速化は run_stage6_local.py の perm_cached / run_baseline。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage11\\run_stage11.py
  試運転: ... run_stage11.py --n-seeds 1 --n-perm 20 --n-perm-sens 20
出力: rocket/outputs/stage11/、入力: rocket/data/processed/stage11/
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

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage7'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
from run_stage6 import BASE_ALPHAS, s5  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table, summarize  # noqa: E402
from src.windows import extract_windows, first_diff, load_timeseries  # noqa: E402

PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DATA_DIR = os.path.join(PROC, 'stage11')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage11')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
SEEDS = list(range(10))
N_JOBS = fast.N_JOBS

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED, GRID = s6.INK, s6.MUTED, s6.GRID
C_MAIN, C_ALT, C_BASE = s6.C_MAIN, s6.C_ALT, '#9a9890'


# ---------------------------------------------------------------- データ
def load_cases_year():
    """label_year が付いた圃場×年。開花日は CSV の flower_date(実測優先、無ければ EXP-016 の予測)。"""
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    d = lab[lab['label_year'].notna()].copy()
    d['y'] = d['label_year'].astype(int)
    d['muniyear'] = d['muniCd'] + '_' + d['year'].astype(str)
    for c in ['seed_date', 'harvest_date', 'flower_date']:
        d[c] = pd.to_datetime(d[c])
    d['seed_doy'] = d['seed_date'].dt.dayofyear
    d['harvest_dap'] = (d['harvest_date'] - d['seed_date']).dt.days
    d['flower_dap'] = (d['flower_date'] - d['seed_date']).dt.days
    assert d['flower_dap'].notna().all() and d['harvest_dap'].notna().all(), 'flower_dap / harvest_dap に欠測がある'
    d['flower_dap'] = d['flower_dap'].astype(int)
    return d.reset_index(drop=True)


def build_dataset(log):
    cases = load_cases_year()
    ts = load_timeseries(cases)
    # 観測日 − seed_date が dap と一致すること(soilmoisture_timeseries.csv の dap と CSV の seed_date が同じ基準)
    chk = ts.merge(cases[['field_id', 'year', 'seed_date']], on=['field_id', 'year'])
    assert ((chk['date'] - chk['seed_date']).dt.days == chk['dap']).all(), '観測日 − seed_date が dap と一致しない'
    X, meta, excluded = extract_windows(cases, ts, 'flower', 0, 59)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'year_main_X.npy'), X)
    meta.to_csv(os.path.join(DATA_DIR, 'year_main_meta.csv'), index=False, encoding='utf-8-sig')
    excluded.to_csv(os.path.join(DATA_DIR, 'year_main_excluded.csv'), index=False, encoding='utf-8-sig')
    log(f'label_year 付き {len(cases)} 件(多収 {int(cases.y.sum())} / 低収 {int((1 - cases.y).sum())})、'
        f'収穫日より後で除いた観測 {ts.attrs["n_dropped_after_harvest"]} 行')
    log(f'窓(開花起点 0〜+59 日)を覆う件: {len(meta)}、除外 {len(excluded)}')
    log('  除外の理由(先頭の理由で集計): ' + str(excluded['reason'].str.split(' / ').str[0]
                                              .str.replace(r'\d+', 'N', regex=True).value_counts().to_dict()))
    return X, meta


def describe(meta, log, name):
    y = meta.y.values
    log(f'[{name}] {len(y)} 件(低収 {int((1 - y).sum())} / 多収 {int(y.sum())})、地域 {meta.muniCd.nunique()}')
    log(f'  年別: ' + ', '.join(f'{yr}年 {n}件(多収 {int(meta[meta.year == yr].y.sum())})'
                              for yr, n in meta.year.value_counts().sort_index().items()))
    log(f'  開花日: ' + str(meta.flower_source.value_counts().to_dict()))
    nc = meta.groupby('muniCd').y.nunique()
    one = nc[nc == 1].index
    log(f'  片方のクラスしかない地域: {len(one)} / {len(nc)}、その件数 {int(meta.muniCd.isin(one).sum())}')


def baseline_features(X, meta):
    v = X[:, 0, :]
    return {
        'B1': np.c_[v.mean(1), v.std(1, ddof=1), v.min(1), v.max(1)],
        'B2': meta[['lat', 'lon', 'seed_doy']].to_numpy(dtype=float),
    }


# ---------------------------------------------------------------- 評価
def run_minirocket(config, dataset, X, meta, n_perm, log):
    """段階6の run_config と同じ統計量。置換はラベルを年の中で並べ替え、観測ラベルで作った分割を固定して使う。"""
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    t = time.time()
    folds, nulls, per_seed = [], [], []
    for seed in SEEDS:
        feats = s6.fold_features('minirocket', seed, X, sp)
        f = s6.eval_cached(feats, y, sp).assign(seed=seed)
        null = fast.perm_cached(feats, y, sp, meta.year.values, n_perm) if n_perm else np.empty((0, 2))
        del feats
        s = summarize(f)
        per_seed.append({'config': config, 'seed': seed, 'bacc_mean': s['bacc_mean'], 'auc_mean': s['auc_mean'],
                         'alpha_at_max': np.isclose(f.alpha_over_n, BASE_ALPHAS[-1]).mean(),
                         'alpha_at_min': np.isclose(f.alpha_over_n, BASE_ALPHAS[0]).mean()})
        folds.append(f)
        nulls.append(null)
    folds = pd.concat(folds)
    ps = pd.DataFrame(per_seed)
    null = np.mean(np.stack(nulls), axis=0) if n_perm else np.empty((0, 2))  # 置換 i ごとに乱数で平均
    r = {'config': config, 'model': 'M', 'dataset': dataset, 'n': len(y), 'n_high': int(y.sum()),
         'n_low': int((1 - y).sum()), 'n_seeds': len(SEEDS), 'n_perm': n_perm}
    for m in ['bacc', 'auc']:
        r[f'{m}_mean'] = ps[f'{m}_mean'].mean()
        r[f'{m}_seed_min'], r[f'{m}_seed_median'], r[f'{m}_seed_max'] = \
            ps[f'{m}_mean'].min(), ps[f'{m}_mean'].median(), ps[f'{m}_mean'].max()
        r[f'{m}_sd_fold'] = folds[m].std()
        r[f'{m}_sd_repeat'] = folds.groupby(['seed', 'repeat'])[m].mean().groupby('repeat').mean().std()
    if n_perm:
        r.update(s5.null_stats(r, null))
    r['n_auc_nan'] = int(folds.auc.isna().sum())
    r['alpha_at_max'] = np.isclose(folds.alpha_over_n, BASE_ALPHAS[-1]).mean()
    r['alpha_at_min'] = np.isclose(folds.alpha_over_n, BASE_ALPHAS[0]).mean()
    r['alpha_over_n_median'] = folds.alpha_over_n.median()
    r['seconds'] = round(time.time() - t, 1)
    np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null)
    log(f'  {config:6s} n={len(y)} AUC {r["auc_mean"]:.3f} (乱数 {r["auc_seed_min"]:.3f}〜{r["auc_seed_max"]:.3f}; '
        f'q95 {r.get("auc_null_q95", np.nan):.3f}, p={r.get("auc_p", np.nan):.3f})  bacc {r["bacc_mean"]:.3f} '
        f'(q95 {r.get("bacc_null_q95", np.nan):.3f}, p={r.get("bacc_p", np.nan):.3f})  '
        f'alpha 上限 {r["alpha_at_max"]:.2f}  [{r["seconds"]}s]')
    return r, folds.assign(config=config), ps, split_table(sp, y, meta.muniCd.values).assign(config=config)


def run_baselines(prefix, dataset, X, meta, n_perm, log):
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    rows, folds = [], {}
    for b, F in baseline_features(X, meta).items():
        config = f'{prefix}{b}'
        t = time.time()
        f, obs, null = fast.run_baseline(F, y, sp, meta.year.values, n_perm)
        np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null)
        r = {'config': config, 'model': b, 'dataset': dataset, 'n': len(y), 'n_high': int(y.sum()),
             'n_low': int((1 - y).sum()), 'n_seeds': 1, 'n_perm': n_perm, **obs, **s5.null_stats(obs, null),
             'seconds': round(time.time() - t, 1)}
        rows.append(r)
        folds[b] = f.assign(config=config)
        log(f'  {config:6s} n={len(y)} AUC {r["auc_mean"]:.3f} (q95 {r["auc_null_q95"]:.3f}, p={r["auc_p"]:.3f})  '
            f'bacc {r["bacc_mean"]:.3f} (q95 {r["bacc_null_q95"]:.3f}, p={r["bacc_p"]:.3f})  [{r["seconds"]}s]')
    return rows, folds


def paired_vs_baselines(fM, fB, prefix):
    """M − B1、M − B2 を fold ごとに。基準線は乱数によらないので、各乱数に同じ値を並べて s7.paired に渡す。"""
    seeds = sorted(fM.seed.unique())
    out = []
    for b in ['B1', 'B2']:
        fb = pd.concat([fB[b].assign(seed=s) for s in seeds])
        out.append(s7.paired(fM, fb, f'{prefix}M − {prefix}{b}'))
    return pd.concat(out)


# ---------------------------------------------------------------- 確認
def precheck(Xd, meta, log):
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    a = evaluate(s6.pipeline('minirocket', 0), Xd, y, sp[:10])
    b = s6.eval_cached(s6.fold_features('minirocket', 0, Xd, sp[:10]), y, sp[:10])
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'確認1 Pipeline と特徴量の使い回しの一致(M、乱数0、先頭10 fold): {ok}')
    assert ok

    feats = s6.fold_features('minirocket', 0, Xd, sp)
    t = time.time()
    na = s6.perm_cached(feats, y, sp, meta.year.values, 20)
    ta = time.time() - t
    t = time.time()
    nb = fast.perm_cached(feats, y, sp, meta.year.values, 20)
    tb = time.time() - t
    ok = np.array_equal(na, nb, equal_nan=True)
    log(f'確認2 run_stage6_local.perm_cached と run_stage6.perm_cached の一致(乱数0、置換20回、年の中で並べ替え): '
        f'{ok}  最大差 {np.nanmax(np.abs(na - nb)):.1e}  元 {ta:.1f}s / 新 {tb:.1f}s')
    assert ok


# ---------------------------------------------------------------- 図
def plot_boxes(summary, folds, per_seed):
    order = ['M', 'B1', 'B2', 'S_M', 'S_B1', 'S_B2']
    labels = ['M\nMiniRocket・1次差分', 'B1\nVWC の要約統計', 'B2\n緯度・経度・播種日',
              'S: M\n実測開花のみ', 'S: B1\n実測開花のみ', 'S: B2\n実測開花のみ']
    sm = summary.set_index('config')
    fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for ax, m, lab in [(axs[0], 'auc', 'ROC-AUC(主)'), (axs[1], 'bacc', 'balanced accuracy(副)')]:
        data = [folds[folds.config == c].groupby('repeat')[m].mean().values for c in order]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                        boxprops=dict(lw=0), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
        for patch, c in zip(bp['boxes'], order):
            patch.set_facecolor(C_MAIN if c.endswith('M') else C_BASE)
            patch.set_alpha(0.5)
        for i, c in enumerate(order):
            x = i + 1
            pts = per_seed[per_seed.config == c][f'{m}_mean'].values
            if len(pts):
                ax.scatter(x + np.linspace(-0.15, 0.15, len(pts)), pts, s=12, color=C_ALT, zorder=3)
            ax.hlines(sm.loc[c, f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
        ax.axvline(3.5, color=MUTED, lw=0.8)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylabel(lab, color=INK)
        s6.style(ax)
    axs[1].set_xticks(range(1, len(order) + 1), labels, fontsize=9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_MAIN, alpha=0.5), Patch(color=C_BASE, alpha=0.5),
                Line2D([], [], marker='o', color=C_ALT, ls='', ms=4), Line2D([], [], color=INK, lw=2)],
               ['MiniRocket(M)', '基準線(B1・B2)', '乱数ごとの値', '置換検定の帰無分布の95%点(年の中で並べ替え)'],
               loc='upper left', bbox_to_anchor=(0.02, 0.955), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f'段階11 label_year({int(sm.loc["M", "n"])}件、右3つは実測開花の {int(sm.loc["S_M", "n"])}件)'
                 '。箱 = 20回の繰り返し平均', fontsize=13, color=INK, x=0.02, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_stage11.png'), dpi=150)
    plt.close(fig)


def plot_null_M(summary):
    r = summary[summary.config == 'M'].iloc[0]
    null = np.load(os.path.join(OUT_DIR, 'null_M.npy'))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, j, m, lab in [(axs[0], 1, 'auc', 'AUC(主)'), (axs[1], 0, 'bacc', 'balanced accuracy(副)')]:
        ax.hist(null[:, j], bins=40, color=C_BASE, alpha=0.7)
        ax.axvline(r[f'{m}_null_q95'], color=INK, lw=1.5, ls='--', label=f'95%点 {r[f"{m}_null_q95"]:.3f}')
        ax.axvline(r[f'{m}_mean'], color=s6.C_RED, lw=2, label=f'観測値 {r[f"{m}_mean"]:.3f}(p={r[f"{m}_p"]:.3f})')
        ax.set_xlabel(f'{lab}(乱数で平均)', color=INK)
        ax.legend(frameon=False, fontsize=9)
        s6.style(ax)
    axs[0].set_ylabel('置換の回数', color=INK)
    fig.suptitle(f'M: 置換検定の帰無分布({int(r.n_perm)}回、年の中で並べ替え、分割は固定)と観測値',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'null_M.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-perm-sens', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    global SEEDS
    SEEDS = list(range(args.n_seeds))
    s6.N_JOBS = N_JOBS  # fold_features と元の perm_cached の並列数
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {N_JOBS}')
    X, meta = build_dataset(log)
    obs = meta[meta.flower_source == 'obs'].index.values
    Xs, ms = X[obs], meta.loc[obs].reset_index(drop=True)
    describe(meta, log, '主データ')
    describe(ms, log, 'S 実測開花日のみ')
    Xd, Xsd = first_diff(X), first_diff(Xs)
    assert Xd.shape == (len(meta), 1, 59)

    precheck(Xd, meta, log)

    rows, fold_rows, seed_rows, split_rows = [], [], [], []
    log(f'\n[主データ](置換 {args.n_perm} 回)')
    r, fM, ps, st = run_minirocket('M', 'year_main', Xd, meta, args.n_perm, log)
    rows.append(r), fold_rows.append(fM), seed_rows.append(ps), split_rows.append(st)
    bl, fB = run_baselines('', 'year_main', X, meta, args.n_perm, log)
    rows += bl
    fold_rows += list(fB.values())

    log(f'\n[S 実測開花日のみ](置換 {args.n_perm_sens} 回)')
    r, fSM, ps, st = run_minirocket('S_M', 'year_obs', Xsd, ms, args.n_perm_sens, log)
    rows.append(r), fold_rows.append(fSM), seed_rows.append(ps), split_rows.append(st)
    bl, fSB = run_baselines('S_', 'year_obs', Xs, ms, args.n_perm_sens, log)
    rows += bl
    fold_rows += list(fSB.values())

    summary = pd.DataFrame(rows)
    for m in ['bacc', 'auc']:
        best = summary[summary.model != 'M'].groupby('dataset')[f'{m}_mean'].max()
        summary[f'{m}_best_baseline'] = summary.dataset.map(best)
        hit = (summary[f'{m}_mean'] > summary[f'{m}_null_q95']) & (summary[f'{m}_mean'] > summary[f'{m}_best_baseline'])
        summary[f'{m}_hit'] = hit.astype(object).where(summary.model == 'M', None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat(fold_rows)
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed = pd.concat(seed_rows)
    per_seed.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat(split_rows).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')

    pairs = pd.concat([paired_vs_baselines(fM, fB, ''), paired_vs_baselines(fSM, fSB, 'S_')])
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    pst = s7.paired_summary(pairs)
    pst.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, folds, per_seed)
    plot_null_M(summary)

    show = ['config', 'n', 'auc_mean', 'auc_seed_min', 'auc_seed_median', 'auc_seed_max', 'auc_null_mean',
            'auc_null_q95', 'auc_p', 'auc_best_baseline', 'auc_hit', 'bacc_mean', 'bacc_null_q95', 'bacc_p',
            'bacc_best_baseline', 'bacc_hit', 'alpha_at_max', 'n_auc_nan']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log('\n[fold ごとの差(同じ分割)]')
    log(pst.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
