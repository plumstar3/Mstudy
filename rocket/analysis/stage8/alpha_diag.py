"""
段階8の補足: alpha が候補の上限に張り付く理由の観察(すべて探索的。評価の設定は変えず、判定にも使わない)

  段階6〜8 では、どの設定でもほぼ100%の fold で RidgeClassifierCV が alpha の上限(1000 × n_train)を選んだ。
  alpha の候補を logspace(−3, 6, 19) × n_train に広げ、fold ごとに alpha ごとの次の値を出す。
    1. GCV の値(LOO の2乗誤差の平均。±1 のラベルに対して RidgeClassifierCV が最小化するもの)
    2. LOO の正解率・AUC(LOO 予測値の符号・値から)。1 の最小と 2 の最大がずれていないかを見る
    3. 実効自由度 df(α) = Σ λ_i / (λ_i + α)。λ_i は中心化した学習データ Xc の XcXcᵀ の固有値(切片の1は含めない)
    4. テストの AUC・bacc(その alpha に固定したとき)
    5. α→∞ の参照分類器: w = Xcᵀ yc(クラス平均の差の方向)。テストの AUC と、学習データの平均を境にした bacc
       (alpha を大きくしたときの決定関数から切片を引いたものの向きの極限)
  ラベルを市町村×年の中で並べ替えた場合(置換 i は乱数 [0, i]、段階6〜8と同じ並べ方)も、同じ値を出す。

  計算は sklearn 1.5.1 の _RidgeGCV(eigen)と同じ式(run_stage6_local._batched_ridge と同じ)。
  LOO 予測値 = y − c / diag(G⁻¹)。最初の fold で次を確かめる(run_log.txt に出す):
    - GCV の値が RidgeClassifierCV(store_cv_results=True)の cv_results_ と一致する
    - 元の候補(logspace(−3, 3, 10) × n_train)で選ばれる alpha と決定関数が RidgeNTrain と一致する
    - LOO 予測値が、1件ずつ抜いて RidgeClassifier を学習し直した値と一致する(alpha 3通り)

  設定: MR_diff(段階6の実験D: MiniRocket・1次差分)、R0 / R1 / R2 の PPV のみ・PPV + max(段階7・8の Rocket)。
  特徴量(StandardScaler(with_mean=False) まで)・分割(5分割 × 20回)は段階6〜8と同じ。乱数(カーネル)は 0 のみ。

実行(別の PC で):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage8\\alpha_diag.py
  試運転: ... alpha_diag.py --configs MR_diff --n-perm 5 --n-folds 10
出力: rocket/outputs/stage8/alpha_diag/
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
from scipy import linalg
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage7'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, STAGE_DIR)
import run_stage6 as s6  # noqa: E402
import run_stage8 as s8  # noqa: E402
from src.evaluation import make_splits, permute_within  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage8', 'alpha_diag')
GRID = np.logspace(-3, 6, 19)  # alpha / n_train。0.5 刻み(対数)
OLD_MAX = s6.BASE_ALPHAS[-1]  # 評価で使っている候補の上限(1000)
CONFIGS = ['MR_diff', 'R0_ppv', 'R1_ppv', 'R2_ppv', 'R0_ppvmax', 'R1_ppvmax', 'R2_ppvmax']
METRICS = ['loo_mse', 'loo_acc', 'loo_auc', 'test_auc', 'test_bacc']
N_JOBS = os.cpu_count()
INK, MUTED, C_MAIN, C_ALT = s6.INK, s6.MUTED, s6.C_MAIN, s6.C_ALT
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 特徴量
def fold_features(cfg, seed, X, tr, te):
    """段階6〜8と同じ特徴量(StandardScaler(with_mean=False) まで、学習データだけで fit)。"""
    if cfg == 'MR_diff':
        return s6._one_fold('minirocket', seed, X, tr, te)
    inp, pool = cfg.split('_')
    if inp == 'R0':  # 段階7の Rocket(normalise=False)
        tf = make_pipeline(s8.RocketPool(pool, normalise=False, random_state=seed), StandardScaler(with_mean=False))
    else:
        tf = s8.make_transform(inp, pool, seed)
    tf.fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[te])


# ---------------------------------------------------------------- Ridge の経路
def ridge_path(Ftr, Fte, Ytr, alphas):
    """Ytr (P, n) の 0/1 ラベルそれぞれについて、alpha ごとの LOO 残差・LOO 予測値・テストの決定関数。

    戻り値: e, loo (n_alpha, n, P)、dec (n_alpha, n_test, P)、df (n_alpha,)、ref (n_test, P)
    """
    n = len(Ftr)
    X_offset = Ftr.mean(axis=0)
    Xc = Ftr - X_offset
    K = Xc @ Xc.T
    eigvals, Q = linalg.eigh(K + 1.0)
    intercept_dim = np.argmax(np.abs(np.full(n, 1 / np.sqrt(n)) @ Q))
    lam = np.clip(linalg.eigvalsh(K), 0, None)
    Kte = (Fte - X_offset) @ Xc.T  # (n_test, n)

    Y = (2.0 * np.atleast_2d(Ytr) - 1.0).T  # LabelBinarizer(pos_label=1, neg_label=-1)、(n, P)
    y_offset = Y.mean(axis=0)
    Yc = Y - y_offset
    QT_y = Q.T @ Yc
    E, L, D = [], [], []
    for a in alphas:
        w = 1.0 / (eigvals + float(a))
        w[intercept_dim] = 0
        c = Q @ (w[:, None] * QT_y)
        e = c / (w * Q ** 2).sum(axis=-1)[:, None]  # LOO 残差
        E.append(e)
        L.append(Y - e)
        D.append(Kte @ c + y_offset)  # = Fte @ coef + intercept(coef = cᵀ Xc)
    df = (lam[:, None] / (lam[:, None] + np.asarray(alphas))).sum(axis=0)
    ref = Kte @ Yc  # (x − x̄)ᵀ Xcᵀ yc
    return np.array(E), np.array(L), np.array(D), df, ref


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def run_fold(cfg, seed, X, y, Yp, r, f, tr, te):
    """1つの fold の、本当のラベル(perm = −1)と置換ごとの、alpha ごとの値。"""
    Ftr, Fte = fold_features(cfg, seed, X, tr, te)
    Ytr, Yte = np.vstack([y[tr], Yp[:, tr]]), np.vstack([y[te], Yp[:, te]])
    n = len(tr)
    E, L, D, df, ref = ridge_path(Ftr, Fte, Ytr, GRID * n)
    rows, ref_rows = [], []
    for p in range(len(Ytr)):
        for j, g in enumerate(GRID):
            rows.append({'repeat': r, 'fold': f, 'perm': p - 1, 'n_train': n, 'alpha_over_n': g, 'df': df[j],
                         'loo_mse': (E[j, :, p] ** 2).mean(),
                         'loo_acc': ((L[j, :, p] > 0) == (Ytr[p] == 1)).mean(),
                         'loo_auc': _auc(Ytr[p], L[j, :, p]),
                         'test_auc': _auc(Yte[p], D[j, :, p]),
                         'test_bacc': balanced_accuracy_score(Yte[p], (D[j, :, p] > 0).astype(int))})
        ref_rows.append({'repeat': r, 'fold': f, 'perm': p - 1, 'ref_test_auc': _auc(Yte[p], ref[:, p]),
                         'ref_test_bacc': balanced_accuracy_score(Yte[p], (ref[:, p] > 0).astype(int))})
    return rows, ref_rows


# ---------------------------------------------------------------- 確認
def check_against_sklearn(Ftr, Fte, ytr, log):
    n = len(Ftr)
    E, L, D, _, _ = ridge_path(Ftr, Fte, ytr, GRID * n)
    mse = (E[:, :, 0] ** 2).mean(axis=1)
    sk = RidgeClassifierCV(alphas=GRID * n, store_cv_results=True).fit(Ftr, ytr)
    sk_mse = sk.cv_results_.reshape(n, -1).mean(axis=0)
    ok1 = np.allclose(mse, sk_mse, rtol=1e-7) and np.isclose(sk.alpha_, GRID[np.argmin(mse)] * n)
    log(f'  GCV の値が cv_results_ と一致(広げた候補): {ok1}(最大の相対差 {np.max(np.abs(mse / sk_mse - 1)):.1e})')

    E0, _, D0, _, _ = ridge_path(Ftr, Fte, ytr, s6.BASE_ALPHAS * n)
    best = np.argmax(-(E0[:, :, 0] ** 2).mean(axis=1))  # 同点なら先の alpha(sklearn と同じ)
    ref = s6.RidgeNTrain().fit(Ftr, ytr)
    ok2 = np.isclose(ref.alpha_, s6.BASE_ALPHAS[best] * n) and np.allclose(
        D0[best, :, 0], ref.decision_function(Fte), rtol=1e-7, atol=1e-10)
    log(f'  元の候補で選ばれる alpha と決定関数が RidgeNTrain と一致: {ok2}(alpha / n = {ref.alpha_ / n:g})')

    ok3 = True
    for j in [0, int(np.argmin(np.abs(GRID - OLD_MAX))), len(GRID) - 1]:
        brute = np.empty(n)
        for i in range(n):
            m = np.arange(n) != i
            brute[i] = RidgeClassifier(alpha=GRID[j] * n).fit(Ftr[m], ytr[m]).decision_function(Ftr[i:i + 1])[0]
        diff = np.max(np.abs(brute - L[j, :, 0]))
        same_sign = np.mean((brute > 0) == (L[j, :, 0] > 0))
        ok = np.allclose(brute, L[j, :, 0], rtol=1e-6, atol=1e-8)
        ok3 &= ok
        log(f'  LOO 予測値が1件ずつ抜いた学習と一致(alpha / n = {GRID[j]:g}): {ok}'
            f'(最大の差 {diff:.1e}、符号の一致 {same_sign:.3f})')
    return ok1 and ok2, ok3


# ---------------------------------------------------------------- 集計・図
def summarize_by_alpha(cur):
    """fold で平均してから、本当のラベルと、置換の分布(平均・5%点・95%点)を alpha ごとに並べる。"""
    m = cur.groupby(['config', 'perm', 'alpha_over_n'])[METRICS + ['df']].mean().reset_index()
    real = m[m.perm == -1].drop(columns='perm').set_index(['config', 'alpha_over_n'])
    perm = m[m.perm >= 0].groupby(['config', 'alpha_over_n'])[METRICS]
    out = real.add_prefix('real_')
    for name, fn in [('mean', 'mean'), ('q05', lambda s: s.quantile(0.05)), ('q95', lambda s: s.quantile(0.95))]:
        out = out.join(perm.agg(fn).add_prefix('perm_').add_suffix(f'_{name}'))
    return out.reset_index()


def summarize_argmin(cur, ref):
    """fold ごとに、GCV が最小・LOO 正解率が最大になる alpha(同点なら小さい方)と、その位置での値。"""
    rows = []
    for (cfg, perm), g in cur.groupby(['config', 'perm']):
        for (r, f), h in g.groupby(['repeat', 'fold']):
            h = h.sort_values('alpha_over_n')
            i_mse, i_acc = h.loo_mse.values.argmin(), h.loo_acc.values.argmax()
            rows.append({'config': cfg, 'perm': perm, 'repeat': r, 'fold': f,
                         'gcv_argmin': h.alpha_over_n.values[i_mse], 'loo_acc_argmax': h.alpha_over_n.values[i_acc],
                         'test_auc_at_gcv_argmin': h.test_auc.values[i_mse],
                         'test_auc_at_old_max': h.test_auc.values[np.isclose(h.alpha_over_n.values, OLD_MAX)][0],
                         'test_auc_at_grid_max': h.test_auc.values[-1],
                         'df_at_old_max': h.df.values[np.isclose(h.alpha_over_n.values, OLD_MAX)][0]})
    am = pd.DataFrame(rows).merge(ref, on=['config', 'perm', 'repeat', 'fold'])
    s = []
    for (cfg, real), g in am.groupby(['config', am.perm == -1]):
        s.append({'config': cfg, 'labels': 'real' if real else 'perm', 'n_rows': len(g),
                  'gcv_argmin_at_grid_max': np.isclose(g.gcv_argmin, GRID[-1]).mean(),
                  'gcv_argmin_above_old_max': (g.gcv_argmin > OLD_MAX * 1.0001).mean(),
                  'gcv_argmin_interior': ((g.gcv_argmin > GRID[0] * 1.0001) & (g.gcv_argmin < GRID[-1] / 1.0001)).mean(),
                  'gcv_argmin_median': g.gcv_argmin.median(), 'loo_acc_argmax_median': g.loo_acc_argmax.median(),
                  'test_auc_at_old_max': g.test_auc_at_old_max.mean(),
                  'test_auc_at_grid_max': g.test_auc_at_grid_max.mean(),
                  'test_auc_at_gcv_argmin': g.test_auc_at_gcv_argmin.mean(),
                  'ref_test_auc': g.ref_test_auc.mean(), 'ref_test_bacc': g.ref_test_bacc.mean(),
                  'df_at_old_max_median': g.df_at_old_max.median()})
    return am, pd.DataFrame(s)


def plot_curves(sa, sm, cfg, fname):
    labels = {'loo_mse': 'GCV(LOO の2乗誤差の平均、小さいほど良い)', 'loo_acc': 'LOO の正解率',
              'loo_auc': 'LOO の AUC', 'test_auc': 'テストの AUC', 'test_bacc': 'テストの bacc'}
    d = sa[sa.config == cfg].sort_values('alpha_over_n')
    ref = sm[(sm.config == cfg) & (sm.labels == 'real')].iloc[0]
    x = d.alpha_over_n.values
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, m in zip(axs.flat, METRICS + ['df']):
        if m == 'df':
            ax.plot(x, d.real_df, color=C_MAIN, lw=2, marker='o', ms=4)
            ax.set_title('実効自由度 df(α)(切片を除く)', fontsize=11, color=INK)
        else:
            ax.fill_between(x, d[f'perm_{m}_q05'], d[f'perm_{m}_q95'], color=MUTED, alpha=0.18, lw=0,
                            label='置換したラベル(5〜95%点)')
            ax.plot(x, d[f'perm_{m}_mean'], color=MUTED, lw=1.5, ls='--', label='置換したラベル(平均)')
            ax.plot(x, d[f'real_{m}'], color=C_MAIN, lw=2, marker='o', ms=4, label='本当のラベル')
            ax.set_title(labels[m], fontsize=11, color=INK)
            if m in ('loo_auc', 'test_auc', 'loo_acc', 'test_bacc'):
                ax.axhline(0.5, color=MUTED, lw=0.8)
            if m == 'test_auc':
                ax.axhline(ref.ref_test_auc, color=C_ALT, lw=1.5, ls=':', label='α→∞ の参照分類器(本当のラベル)')
            if m == 'test_bacc':
                ax.axhline(ref.ref_test_bacc, color=C_ALT, lw=1.5, ls=':', label='α→∞ の参照分類器(本当のラベル)')
        ax.axvline(OLD_MAX, color=INK, lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel('alpha / n_train(点線 = 評価で使っている上限 1000)', fontsize=9, color=MUTED)
        s6.style(ax)
    h, lab = axs.flat[3].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f'{cfg}: alpha ごとの値(fold で平均、乱数0、5分割 × 20回)', fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close(fig)


def plot_gcv_folds(cur, cfg, fname):
    """fold ごとの GCV の曲線(最大の alpha での値との差)。どこかで最小になるのか、単調に下がるのかを見る。"""
    g = cur[(cur.config == cfg) & (cur.perm == -1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, h in g.groupby(['repeat', 'fold']):
        h = h.sort_values('alpha_over_n')
        ax.plot(h.alpha_over_n, h.loo_mse - h.loo_mse.values[-1], color=C_MAIN, lw=0.6, alpha=0.25)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(OLD_MAX, color=INK, lw=0.8, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('alpha / n_train(点線 = 評価で使っている上限 1000)', fontsize=9, color=MUTED)
    ax.set_ylabel('GCV − (alpha 最大での GCV)', fontsize=9, color=MUTED)
    ax.set_title(f'{cfg}: fold ごとの GCV の曲線(本当のラベル、{g.groupby(["repeat", "fold"]).ngroups} fold)',
                 fontsize=12, color=INK)
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=CONFIGS, choices=CONFIGS)
    ap.add_argument('--n-perm', type=int, default=50)
    ap.add_argument('--n-folds', type=int, default=None, help='試運転用。先頭から何 fold 使うか(既定はすべての100)')
    ap.add_argument('--seed', type=int, default=0, help='カーネルの乱数')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    import sklearn
    log(f'並列数 {N_JOBS}、設定 {args.configs}、乱数 {args.seed}、置換 {args.n_perm} 回、sklearn {sklearn.__version__}')
    log(f'alpha / n_train の候補: {", ".join(f"{g:g}" for g in GRID)}')

    Xm, mm = s6.s5.datasets()['main']
    y = mm.y.values
    X = first_diff(Xm)
    sp = make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)[:args.n_folds]
    Yp = np.array([permute_within(y, mm.muniyear.values, np.random.default_rng([0, i])) for i in range(args.n_perm)])
    Yp = Yp.reshape(args.n_perm, len(y))

    cur, refs = [], []
    for cfg in args.configs:
        r, f, tr, te = sp[0]
        Ftr, Fte = fold_features(cfg, args.seed, X, tr, te)
        log(f'[{cfg}] 確認(最初の fold、学習 {len(tr)} 件 × {Ftr.shape[1]} 列)')
        ok_sk, ok_brute = check_against_sklearn(Ftr, Fte, y[tr], log)
        assert ok_sk, 'sklearn と一致しない'
        if not ok_brute:
            log('  → 1件ずつ抜いた学習と一致しない alpha がある。LOO の正解率・AUC は参考扱いにする')

        res = Parallel(n_jobs=N_JOBS)(delayed(run_fold)(cfg, args.seed, X, y, Yp, r, f, tr, te)
                                      for r, f, tr, te in sp)
        c = pd.DataFrame([row for rows, _ in res for row in rows])
        c.insert(0, 'config', cfg)
        rf = pd.DataFrame([row for _, rows in res for row in rows])
        rf.insert(0, 'config', cfg)
        cur.append(c)
        refs.append(rf)
        log(f'  完了({time.time() - t0:.0f}秒)')

    cur, ref = pd.concat(cur, ignore_index=True), pd.concat(refs, ignore_index=True)
    cur.to_csv(os.path.join(OUT_DIR, 'curves.csv'), index=False, encoding='utf-8-sig')
    sa = summarize_by_alpha(cur)
    sa.to_csv(os.path.join(OUT_DIR, 'summary_by_alpha.csv'), index=False, encoding='utf-8-sig')
    am, sm = summarize_argmin(cur, ref)
    am.to_csv(os.path.join(OUT_DIR, 'argmin_folds.csv'), index=False, encoding='utf-8-sig')
    sm.to_csv(os.path.join(OUT_DIR, 'argmin_summary.csv'), index=False, encoding='utf-8-sig')

    log('\nfold ごとの GCV 最小・LOO 正解率最大の位置と、テストの AUC(fold の平均。perm は置換すべてをまとめたもの):')
    log(sm.round(3).to_string(index=False))
    log('\nalpha ごとの値(本当のラベル、fold の平均):')
    log(sa[['config', 'alpha_over_n', 'real_df'] + [f'real_{m}' for m in METRICS]].round(4).to_string(index=False))
    for cfg in args.configs:
        plot_curves(sa, sm, cfg, f'curves_{cfg}.png')
        plot_gcv_folds(cur, cfg, f'gcv_folds_{cfg}.png')
    log(f'\n出力: {OUT_DIR}(全体 {time.time() - t0:.0f}秒)')


if __name__ == '__main__':
    main()
