"""
段階6(run_stage6.py)を、この PC(i9-14900K、32スレッド)向けに速くしたもの。処理と出力は run_stage6.py と同じ。

run_stage6.py をそのまま import し、次の3点だけを差し替えて main() を呼ぶ。
  1. 並列数 N_JOBS を 20 → この PC の論理コア数(os.cpu_count())
  2. 置換検定(perm_cached)の計算方法
  3. 基準線の置換検定(s5.run_baseline)の並列数 12 → N_JOBS

2 の中身: RidgeClassifierCV(GCV)の計算のうち、K = Xc Xc^T + 1 の固有値分解と diag(G^-1) は特徴量だけで決まり、
ラベルによらない。そこで fold ごとに一度だけ計算し、置換 n_perm 回分のラベルを列に並べてまとめて解く
(sklearn 1.5.1 の _RidgeGCV の eigen モードと同じ式)。元は置換1回ごとに RidgeClassifierCV を学習し直していた。
置換 i の並べ方(乱数 [0, i])、alpha の選び方(同点なら先の alpha)、bacc・AUC・平均の計算は元と同じ関数を使う。

まとめて解くと、行列積の足し算の順番が変わり、最後の桁が元と違うことがある。結果が変わるのは、alpha の
GCV スコアがほぼ同点のとき、決定関数の値がほぼ 0 のとき、テスト件どうしの決定関数の値がほぼ同じとき
だけなので、該当する置換は元と同じ RidgeNTrain(sklearn)で学習し直す。さらに
  - fold ごとに、元のラベルでの alpha と決定関数を sklearn と比べる
  - 起動時に、元の perm_cached と置換20回分の null が完全に一致するかを確かめる
GPU は使わない: sktime の MiniRocket は CPU(numba)専用で、残る計算は 65×65 程度の小さな float64 の
行列演算になり、この GPU(RTX 4000 Ada)は float64 が遅いため。float32 にすると元と同じ結果にならない。

実行(引数は run_stage6.py と同じ):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage6\\run_stage6_local.py
出力: rocket/outputs/stage6/(run_stage6.py と同じ場所。上書きする)
"""
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE_DIR)
import run_stage6 as s6  # noqa: E402
from run_stage6 import BASE_ALPHAS, RidgeNTrain, permute_within, s5, summarize  # noqa: E402

N_JOBS = os.cpu_count()
TIE_RTOL = 1e-9  # これより近い値は、足し算の順番で大小が入れ替わりうるとみなし sklearn で学習し直す


# ---------------------------------------------------------------- 置換検定(まとめて解く)
def _batched_ridge(Ftr, Fte, Ytr):
    """RidgeNTrain を Ytr の各行(0/1 のラベル)で学習したときの alpha と、テストの決定関数。

    戻り値: alpha (P,), dec (P, n_test), unsure (P,) bool(sklearn で学習し直すべき行)
    """
    n = len(Ftr)
    alphas = np.asarray(BASE_ALPHAS) * n
    X_offset = Ftr.mean(axis=0)
    Xc = Ftr - X_offset
    eigvals, Q = linalg.eigh(Xc @ Xc.T + 1.0)
    intercept_dim = np.argmax(np.abs(np.full(n, 1 / np.sqrt(n)) @ Q))

    Y = (2.0 * Ytr - 1.0).T  # LabelBinarizer(pos_label=1, neg_label=-1)、(n, P)
    y_offset = Y.mean(axis=0)
    QT_y = Q.T @ (Y - y_offset)
    scores, cs = [], []
    for a in alphas:
        w = 1.0 / (eigvals + float(a))
        w[intercept_dim] = 0
        c = Q @ (w[:, None] * QT_y)
        G_inverse_diag = (w * Q ** 2).sum(axis=-1)
        scores.append(-((c / G_inverse_diag[:, None]) ** 2).mean(axis=0))
        cs.append(c)
    scores = np.array(scores)  # (n_alphas, P)
    best = np.argmax(scores, axis=0)  # 同点なら先の alpha(sklearn と同じ)
    P = Ytr.shape[0]
    c_best = np.stack(cs)[best, :, np.arange(P)]  # (P, n)
    coef = c_best @ Xc
    dec = Fte @ coef.T + (y_offset - coef @ X_offset)  # (n_test, P)

    s_best = scores[best, np.arange(P)]
    gap = np.abs(scores - s_best)
    gap[best, np.arange(P)] = np.inf
    unsure = gap.min(axis=0) <= TIE_RTOL * np.abs(s_best)
    scale = np.abs(dec).max(axis=0)
    unsure |= (np.abs(dec) <= TIE_RTOL * scale).any(axis=0)
    srt = np.sort(dec, axis=0)
    unsure |= (np.diff(srt, axis=0) <= TIE_RTOL * scale).any(axis=0)
    return alphas[best], dec.T, unsure


def _perm_fold(Ftr, Fte, y_tr, Yp_tr, Yp_te):
    """1つの fold で、元のラベルの確認と、置換ごとの bacc・AUC。"""
    alpha, dec, unsure = _batched_ridge(Ftr, Fte, np.vstack([y_tr, Yp_tr]))
    ref = RidgeNTrain().fit(Ftr, y_tr)
    ok = np.isclose(alpha[0], ref.alpha_) and np.allclose(dec[0], ref.decision_function(Fte), rtol=1e-7, atol=1e-10)
    alpha, dec, unsure = alpha[1:], dec[1:], unsure[1:]
    redo = np.flatnonzero(unsure)
    for i in redo:
        dec[i] = RidgeNTrain().fit(Ftr, Yp_tr[i]).decision_function(Fte)
    bacc, auc = np.empty(len(dec)), np.full(len(dec), np.nan)
    for i, (d, yte) in enumerate(zip(dec, Yp_te)):
        bacc[i] = balanced_accuracy_score(yte, (d > 0).astype(int))  # RidgeClassifierCV.predict と同じ
        if len(np.unique(yte)) == 2:
            auc[i] = roc_auc_score(yte, d)
    return bacc, auc, ok, len(redo)


def perm_cached(feats, y, splits, perm_groups, n_perm):
    """run_stage6.perm_cached と同じ戻り値((n_perm, 2) の bacc_mean, auc_mean)。"""
    Yp = np.array([permute_within(y, perm_groups, np.random.default_rng([0, i])) for i in range(n_perm)])
    res = Parallel(n_jobs=N_JOBS)(delayed(_perm_fold)(Ftr, Fte, y[tr], Yp[:, tr], Yp[:, te])
                                  for (r, f, tr, te), (Ftr, Fte) in zip(splits, feats))
    assert all(ok for _, _, ok, _ in res), '元のラベルでの alpha・決定関数が sklearn と一致しない fold がある'
    n_redo = sum(k for *_, k in res)
    print(f'    (置換 {n_perm} 回 × {len(splits)} fold のうち sklearn で学習し直した組: {n_redo})', flush=True)
    bacc = np.array([b for b, *_ in res])  # (fold, perm)
    auc = np.array([a for _, a, *_ in res])
    repeat = [r for r, *_ in splits]
    out = []
    for i in range(n_perm):
        s = summarize(s6.pd.DataFrame({'repeat': repeat, 'bacc': bacc[:, i], 'auc': auc[:, i]}))
        out.append((s['bacc_mean'], s['auc_mean']))
    return np.array(out)


# ---------------------------------------------------------------- 基準線(並列数だけ変える)
def run_baseline(F, y, splits, perm_groups, n_perm):
    obs_folds = s5.evaluate(s5.lr(), F, y, splits)
    obs = summarize(obs_folds)

    def one(i):
        yp = permute_within(y, perm_groups, np.random.default_rng([0, i]))
        s = summarize(s5.evaluate(s5.lr(), F, yp, splits))
        return s['bacc_mean'], s['auc_mean']
    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(i) for i in range(n_perm)))
    return obs_folds, obs, null


# ---------------------------------------------------------------- 元の perm_cached との一致
def check_against_original(n_perm=20):
    ds = s5.datasets()
    Xm, mm = ds['main']
    y = mm.y.values
    sp = s6.make_splits(y, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    feats = s6.fold_features('minirocket', 0, s6.first_diff(Xm), sp)
    t = time.time()
    a = s6.perm_cached(feats, y, sp, mm.muniyear.values, n_perm)
    ta = time.time() - t
    t = time.time()
    b = perm_cached(feats, y, sp, mm.muniyear.values, n_perm)
    tb = time.time() - t
    ok = np.array_equal(a, b, equal_nan=True)
    print(f'元の perm_cached との一致(main diff、乱数0、置換 {n_perm} 回): {ok}'
          f'  最大差 {np.nanmax(np.abs(a - b)):.1e}  元 {ta:.1f}s / 新 {tb:.1f}s', flush=True)
    assert ok


def main():
    s6.N_JOBS = N_JOBS
    print(f'並列数 {N_JOBS}', flush=True)
    check_against_original()
    s6.perm_cached = perm_cached
    s5.run_baseline = run_baseline
    s6.main()


if __name__ == '__main__':
    main()
