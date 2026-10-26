"""
段階5: ROCKET 系の分類器で初めて評価する。

  実験A: 揃え方の比較     main_common vs dap_common(77件・同じ圃場・同じ分割)、生の値
                          MiniRocket が主、ROCKET を参考に1回
  実験B: 入力の形の比較   main(82件)で raw / znorm / diff / groupcenter
  実験C: 感度分析         実験Bで平均 AUC が最も高かった入力の形で main_obs / main_no_kakogawa / main_no_irrigation
  基準線                  各データセットで、窓内 VWC の要約統計 / 播種日の通日 / 灌水の有無 + LogisticRegression(C=1)

モデル: sktime 1.2.0 の MiniRocket(num_kernels=10000, random_state=0)→ StandardScaler(with_mean=False)
        → RidgeClassifierCV(alphas=logspace(-3, 3, 10))。AUC は decision_function の値で計算する。
評価:   src/evaluation.py(StratifiedGroupKFold(groups=muniCd) 5分割×20回、置換検定は市町村×年の中で並べ替え)。

計算の工夫: MiniRocket / ROCKET と StandardScaler の fit は学習データの X だけで決まり、ラベルを使わない。
置換検定ではラベルだけが変わり、分割も同じ(evaluation.py の説明を参照)なので、fold ごとの特徴量を一度だけ
計算して使い回し、RidgeClassifierCV だけを学習し直す。Pipeline をそのまま回した結果と一致することを
最初に確かめる(verify_cache_equivalence)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage5\\run_stage5.py [--n-perm 200]
出力: rocket/outputs/stage5/
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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')  # sktime・sklearn の将来の仕様変更の警告(結果には影響しない)

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
from src.evaluation import evaluate, make_splits, permute_within, split_table, summarize  # noqa: E402
from src.windows import first_diff, znorm  # noqa: E402

DATA_DIR = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage4')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage5')
N_SPLITS, N_REPEATS, SEED0 = 5, 20, 0
N_KERNELS, KERNEL_SEED = 10000, 0
ALPHAS = np.logspace(-3, 3, 10)

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_MAIN, C_ALT = '#2a78d6', '#eb6834'


# ---------------------------------------------------------------- データ
def load(name):
    X = np.load(os.path.join(DATA_DIR, f'{name}_X.npy'))
    m = pd.read_csv(os.path.join(DATA_DIR, f'{name}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    return X, m


def datasets():
    ds = {n: load(n) for n in ['main', 'dap', 'main_common', 'dap_common', 'main_obs', 'main_no_kakogawa']}
    X, m = ds['main']
    keep = (m['irrigated'] == 0).values
    ds['main_no_irrigation'] = (X[keep], m[keep].reset_index(drop=True))
    return ds


def groupcenter(X, meta):
    """同じ市町村×年の集団の、各日の VWC 平均を引く(ラベルは使わない。テスト集団の入力値は使う)。"""
    out = X.astype(float).copy()
    for g, idx in meta.groupby('muniyear').indices.items():
        out[idx] -= X[idx].mean(axis=0, keepdims=True)
    return out


FORMS = {'raw': lambda X, m: X.astype(float), 'znorm': lambda X, m: znorm(X),
         'diff': lambda X, m: first_diff(X), 'groupcenter': groupcenter}


# ---------------------------------------------------------------- モデル
class SktimePanel(BaseEstimator, TransformerMixin):
    """sktime の MiniRocket / Rocket を sklearn の Pipeline に入れるための薄い包み。"""

    def __init__(self, kind='minirocket', num_kernels=N_KERNELS, random_state=KERNEL_SEED):
        self.kind, self.num_kernels, self.random_state = kind, num_kernels, random_state

    def _make(self):
        from sktime.transformations.panel.rocket import MiniRocket, Rocket
        cls = MiniRocket if self.kind == 'minirocket' else Rocket
        return cls(num_kernels=self.num_kernels, random_state=self.random_state)

    def fit(self, X, y=None):
        self.t_ = self._make().fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X):
        return np.asarray(self.t_.transform(np.asarray(X, dtype=float)), dtype=float)


def rocket_pipeline(kind):
    return make_pipeline(SktimePanel(kind), StandardScaler(with_mean=False), RidgeClassifierCV(alphas=ALPHAS))


def fold_features(kind, X, splits):
    """fold ごとに、学習データだけで変換器と StandardScaler を fit し、学習・テストの特徴量を返す。"""
    feats = []
    for r, f, tr, te in splits:
        tf = make_pipeline(SktimePanel(kind), StandardScaler(with_mean=False)).fit(X[tr])
        feats.append((tf.transform(X[tr]), tf.transform(X[te])))
    return feats


def eval_cached(feats, y, splits):
    """evaluation.evaluate と同じ出力。分類器(RidgeClassifierCV)だけを fold ごとに学習する。"""
    rows = []
    for (r, f, tr, te), (Ftr, Fte) in zip(splits, feats):
        clf = RidgeClassifierCV(alphas=ALPHAS).fit(Ftr, y[tr])
        pred, score = clf.predict(Fte), clf.decision_function(Fte)
        auc = roc_auc_score(y[te], score) if len(np.unique(y[te])) == 2 else np.nan
        rows.append({'repeat': r, 'fold': f, 'n_test': len(te), 'alpha': clf.alpha_,
                     'bacc': balanced_accuracy_score(y[te], pred), 'auc': auc})
    return pd.DataFrame(rows)


def _perm_chunk(feats, y, splits, perm_groups, seed, idx):
    out = []
    for i in idx:
        yp = permute_within(y, perm_groups, np.random.default_rng([seed, i]))
        s = summarize(eval_cached(feats, yp, splits))
        out.append((s['bacc_mean'], s['auc_mean']))
    return out


def perm_cached(feats, y, splits, perm_groups, n_perm, seed=0, n_jobs=12):
    """置換 i の乱数は i だけで決まる(まとまりの分け方によらない)。特徴量は各ワーカーに1回だけ渡す。"""
    chunks = np.array_split(np.arange(n_perm), n_jobs)
    res = Parallel(n_jobs=n_jobs)(delayed(_perm_chunk)(feats, y, splits, perm_groups, seed, c) for c in chunks)
    return np.array([v for part in res for v in part])


def null_stats(observed, null):
    out = {}
    for j, m in enumerate(['bacc', 'auc']):
        out[f'{m}_null_mean'] = np.nanmean(null[:, j])
        out[f'{m}_null_q95'] = np.nanquantile(null[:, j], 0.95)
        out[f'{m}_p'] = (1 + np.sum(null[:, j] >= observed[f'{m}_mean'])) / (1 + len(null))
    return out


def verify_cache_equivalence(X, y, splits, kind='minirocket', n=10):
    """先頭 n fold で、Pipeline をそのまま回した evaluate() と、特徴量を使い回す eval_cached() を比べる。"""
    sp = splits[:n]
    a = evaluate(rocket_pipeline(kind), X, y, sp)
    b = eval_cached(fold_features(kind, X, sp), y, sp)
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    return ok, a, b


# ---------------------------------------------------------------- 基準線
def baseline_features(X, meta):
    v = X[:, 0, :]
    return {
        'base_vwc_stats': np.c_[v.mean(1), v.std(1, ddof=1), v.min(1), v.max(1)],
        'base_seed_doy': pd.to_datetime(meta['seed_date']).dt.dayofyear.values[:, None].astype(float),
        'base_irrigated': meta['irrigated'].values[:, None].astype(float),
    }


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def run_baseline(F, y, splits, perm_groups, n_perm):
    obs_folds = evaluate(lr(), F, y, splits)
    obs = summarize(obs_folds)

    def one(i):
        yp = permute_within(y, perm_groups, np.random.default_rng([0, i]))
        s = summarize(evaluate(lr(), F, yp, splits))
        return s['bacc_mean'], s['auc_mean']
    null = np.array(Parallel(n_jobs=12)(delayed(one)(i) for i in range(n_perm)))
    return obs_folds, obs, null


# ---------------------------------------------------------------- 図
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_boxes(summary, repeats, baselines):
    cfgs = summary[summary.kind != 'baseline'].reset_index(drop=True)
    fig, axs = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, m, lab in [(axs[0], 'bacc', 'balanced accuracy'), (axs[1], 'auc', 'ROC-AUC')]:
        data = [repeats[repeats.config == c][m].values for c in cfgs.config]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                        boxprops=dict(lw=0), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
        for patch, k in zip(bp['boxes'], cfgs.kind):
            patch.set_facecolor(C_MAIN if k == 'minirocket' else C_ALT)
            patch.set_alpha(0.55)
        for i, r in cfgs.iterrows():
            x = i + 1
            ax.hlines(r[f'{m}_null_q95'], x - 0.32, x + 0.32, color=INK, lw=2)
            b = baselines[(baselines.dataset == r.dataset)]
            best = b[f'{m}_mean'].max()
            ax.plot(x + 0.38, best, marker='D', ms=6, color='#e34948', ls='')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylabel(f'{lab}(20回の繰り返し平均)', color=INK)
        style(ax)
    axs[1].set_xticks(range(1, len(cfgs) + 1),
                      [f'{c}\n{d}' for c, d in zip(cfgs.label, cfgs.dataset)], fontsize=8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_MAIN, alpha=0.55), Patch(color=C_ALT, alpha=0.55), Line2D([], [], color=INK, lw=2),
                Line2D([], [], marker='D', color='#e34948', ls=''), Line2D([], [], color=MUTED, ls=':')],
               ['MiniRocket', 'ROCKET(参考)', '置換検定の帰無分布の95%点', '同じデータセットの最良の基準線', '0.5'],
               loc='upper right', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('設定ごとの性能(箱 = 20回の繰り返し平均の分布)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'boxplot_configs.png'), dpi=150)
    plt.close(fig)


def plot_paired(pairs):
    a = pairs[pairs.comparison.str.startswith('A_')]
    comps = list(a.comparison.unique())
    fig, axs = plt.subplots(len(comps), 2, figsize=(12, 3.6 * len(comps)), squeeze=False)
    for i, c in enumerate(comps):
        s = a[a.comparison == c]
        for j, m in enumerate(['bacc', 'auc']):
            ax = axs[i, j]
            d = s[f'd_{m}'].values
            rep = s.groupby('repeat')[f'd_{m}'].mean().values
            ax.hist(d, bins=np.linspace(-0.6, 0.6, 25), color=C_MAIN, alpha=0.35, label='fold ごとの差(100)')
            ax2 = ax.twinx()
            ax2.hist(rep, bins=np.linspace(-0.6, 0.6, 49), color=C_ALT, alpha=0.9, label='繰り返し平均の差(20)')
            ax2.set_yticks([])
            ax.axvline(0, color=INK, lw=1)
            ax.axvline(d.mean(), color=C_ALT, lw=1.5, ls='--')
            ax.set_title(f'{c}  {m}: 平均 {d.mean():+.3f}、繰り返し平均の 2.5〜97.5% '
                         f'[{np.percentile(rep, 2.5):+.3f}, {np.percentile(rep, 97.5):+.3f}]',
                         fontsize=9, color=INK, loc='left')
            ax.set_xlabel('main − dap', color=INK)
            style(ax)
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('実験A: 揃え方の差(main_common − dap_common、同じ分割)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'expA_paired_diff.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def paired(fa, fb, name):
    m = fa.merge(fb, on=['repeat', 'fold'], suffixes=('_a', '_b'))
    out = pd.DataFrame({'comparison': name, 'repeat': m.repeat, 'fold': m.fold,
                        'd_bacc': m.bacc_a - m.bacc_b, 'd_auc': m.auc_a - m.auc_b})
    return out


def paired_summary(p):
    rows = []
    for c, s in p.groupby('comparison', sort=False):
        r = {'comparison': c}
        for m in ['bacc', 'auc']:
            rep = s.groupby('repeat')[f'd_{m}'].mean()
            r.update({f'{m}_mean_diff': s[f'd_{m}'].mean(), f'{m}_sd_fold': s[f'd_{m}'].std(),
                      f'{m}_rep_q025': rep.quantile(0.025), f'{m}_rep_q975': rep.quantile(0.975),
                      f'{m}_rep_frac_pos': (rep > 0).mean(), f'{m}_rep_frac_neg': (rep < 0).mean(),
                      f'{m}_detected': bool((rep > 0).mean() >= 0.95 or (rep < 0).mean() >= 0.95)})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    ds = datasets()
    splits_cache, rows, fold_rows, split_rows, bl_rows = {}, [], [], [], []

    def splits_for(name):
        if name not in splits_cache:
            X, m = ds[name]
            splits_cache[name] = make_splits(m.y.values, m.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
            split_rows.append(split_table(splits_cache[name], m.y.values, m.muniCd.values).assign(dataset=name))
        return splits_cache[name]

    # main_common と dap_common は同じ件・同じ順番なので分割も同じになる(対の差を取るため確かめる)
    mc, dc = ds['main_common'][1], ds['dap_common'][1]
    assert (mc[['field_id', 'year']].values == dc[['field_id', 'year']].values).all()
    assert all((a[2] == b[2]).all() for a, b in zip(splits_for('main_common'), splits_for('dap_common')))

    # Pipeline と特徴量の使い回しが同じ結果になるか
    X, m = ds['main']
    ok, a, b = verify_cache_equivalence(X.astype(float), m.y.values, splits_for('main'))
    pd.concat([a.assign(method='pipeline'), b.drop(columns='alpha').assign(method='cached')]).to_csv(
        os.path.join(OUT_DIR, 'cache_equivalence_check.csv'), index=False, encoding='utf-8-sig')
    print(f'Pipeline と特徴量の使い回しの一致(先頭10 fold): {ok}')
    assert ok

    def run(config, label, exp, dataset, form, kind):
        X, m = ds[dataset]
        y, sp = m.y.values, splits_for(dataset)
        Xf = FORMS[form](X, m)
        t = time.time()
        feats = fold_features(kind, Xf, sp)
        folds = eval_cached(feats, y, sp)
        obs = summarize(folds)
        null = perm_cached(feats, y, sp, m.muniyear.values, args.n_perm)
        del feats
        r = {'config': config, 'label': label, 'experiment': exp, 'dataset': dataset, 'form': form, 'kind': kind,
             'n': len(y), 'n_perm': args.n_perm, **obs, **null_stats(obs, null),
             'alpha_median': folds.alpha.median(), 'alpha_at_min': (folds.alpha == ALPHAS[0]).mean(),
             'alpha_at_max': (folds.alpha == ALPHAS[-1]).mean(),
             'seconds': round(time.time() - t, 1)}
        if form == 'groupcenter':
            r['n_single_case_groups'] = int((m.groupby('muniyear').size() == 1).sum())
        rows.append(r)
        fold_rows.append(folds.assign(config=config))
        np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null)
        print(f'  {config:28s} n={len(y)} bacc {obs["bacc_mean"]:.3f}±{obs["bacc_sd_fold"]:.3f} '
              f'(q95 {r["bacc_null_q95"]:.3f}, p={r["bacc_p"]:.3f})  auc {obs["auc_mean"]:.3f}±{obs["auc_sd_fold"]:.3f} '
              f'(q95 {r["auc_null_q95"]:.3f}, p={r["auc_p"]:.3f})  [{r["seconds"]}s]')
        return folds

    print('\n[実験A] 揃え方の比較(77件、生の値)')
    fA = {}
    for kind, tag in [('minirocket', 'MR'), ('rocket', 'RK')]:
        for dset in ['main_common', 'dap_common']:
            fA[(kind, dset)] = run(f'A_{tag}_{dset}', f'A {tag} {dset}', 'A', dset, 'raw', kind)

    print('\n[実験B] 入力の形の比較(main 82件、MiniRocket)')
    fB = {form: run(f'B_MR_main_{form}', f'B {form}', 'B', 'main', form, 'minirocket') for form in FORMS}

    s = pd.DataFrame(rows)
    sb = s[s.experiment == 'B'].sort_values(['auc_mean', 'bacc_mean'], ascending=False)
    best = sb.iloc[0]['form']
    print(f'  → 平均 AUC が最も高い入力の形: {best}(bacc の順位: {list(sb.sort_values("bacc_mean", ascending=False).form)})')

    print(f'\n[実験C] 感度分析(入力の形 = {best})')
    for dset in ['main_obs', 'main_no_kakogawa', 'main_no_irrigation']:
        run(f'C_MR_{dset}_{best}', f'C {dset}', 'C', dset, best, 'minirocket')

    print('\n[基準線]')
    for dset in ['main_common', 'dap_common', 'main', 'main_obs', 'main_no_kakogawa', 'main_no_irrigation']:
        X, m = ds[dset]
        y, sp = m.y.values, splits_for(dset)
        for bname, F in baseline_features(X, m).items():
            folds, obs, null = run_baseline(F, y, sp, m.muniyear.values, args.n_perm)
            bl_rows.append({'config': f'{bname}_{dset}', 'label': bname, 'experiment': 'baseline', 'dataset': dset,
                            'form': '-', 'kind': 'baseline', 'n': len(y), 'n_perm': args.n_perm,
                            **obs, **null_stats(obs, null)})
            fold_rows.append(folds.assign(config=f'{bname}_{dset}'))
            print(f'  {dset:20s} {bname:16s} bacc {obs["bacc_mean"]:.3f}  auc {obs["auc_mean"]:.3f}')

    summary = pd.DataFrame(rows + bl_rows)
    baselines = pd.DataFrame(bl_rows)
    # 判断の目安: 帰無分布の95%点を超え、かつ同じデータセットの最良の基準線を上回る
    for mt in ['bacc', 'auc']:
        best_bl = baselines.groupby('dataset')[f'{mt}_mean'].max()
        summary[f'{mt}_best_baseline'] = summary.dataset.map(best_bl)
        hit = (summary[f'{mt}_mean'] > summary[f'{mt}_null_q95']) & \
              (summary[f'{mt}_mean'] > summary[f'{mt}_best_baseline'])
        summary[f'{mt}_hit'] = hit.astype(object).where(summary.kind != 'baseline', None)  # 基準線自身は空欄
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat(fold_rows)
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    repeats = folds.groupby(['config', 'repeat'])[['bacc', 'auc']].mean().reset_index()
    repeats.to_csv(os.path.join(OUT_DIR, 'repeats.csv'), index=False, encoding='utf-8-sig')
    pd.concat(split_rows).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')

    # fold ごとの差(同じ分割)
    pairs = [paired(fA[('minirocket', 'main_common')], fA[('minirocket', 'dap_common')], 'A_MR_main−dap'),
             paired(fA[('rocket', 'main_common')], fA[('rocket', 'dap_common')], 'A_RK_main−dap')]
    pairs += [paired(fB[f], fB['raw'], f'B_{f}−raw') for f in FORMS if f != 'raw']
    pairs = pd.concat(pairs)
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    plot_boxes(summary, repeats, baselines)
    plot_paired(pairs)

    show = ['config', 'n', 'bacc_mean', 'bacc_sd_fold', 'bacc_null_q95', 'bacc_p', 'bacc_best_baseline', 'bacc_hit',
            'auc_mean', 'auc_sd_fold', 'auc_null_q95', 'auc_p', 'auc_best_baseline', 'auc_hit', 'alpha_at_max']
    print('\n[まとめ]')
    print(summary[show].round(3).to_string(index=False))
    print('\n[fold ごとの差]')
    print(ps.round(3).to_string(index=False))
    print(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
