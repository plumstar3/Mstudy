r"""
長さの揃わない系列への対処法を ArrowHead で比べる(EXP-002)。

ArrowHead 公式分割(学習36 / テスト175、長さ251)の各系列を、長さ L ~ U[MIN_LEN, 251] に
先頭から切り詰めて不揃いにし、以下の対処法の Accuracy を N_REPEATS 回の平均±標準偏差で比べる。
繰り返しごとに、切り詰める長さ・カーネル・ノイズの乱数を変える。

  [基準] 切り詰める前(長さ251)
  ① ノイズパディング : 本来の長さで z 正規化 → 学習データの最長に合わせて低振幅ノイズでパディング
  ② 補間             : z 正規化 → 学習データの長さの中央値に線形補間で伸縮
  ③ 長さを変えず適用 : z 正規化 → 公式カーネル(学習データの最短長で生成)を1系列ずつ適用
  ④ ライブラリ       : sktime MiniRocketMultivariateVariable
  [参考] 最短に切り揃え: z 正規化 → 学習データの最短長に末尾を切る(それより短いテスト系列はノイズでパディング)

長さなどのパラメータは学習データだけから決める(テストデータの情報を使わない)。
分類器はすべて StandardScaler(with_mean=False) → RidgeClassifierCV(alphas=logspace(-3, 3, 10))。

使い方:
  rocket\venv_rocket\Scripts\python rocket\scripts\compare_unequal_handling.py
出力:
  rocket/outputs/compare_unequal_handling/results.csv(繰り返しごとの全結果)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_arrow_head
from sktime.transformations.rocket import MiniRocketMultivariateVariable

warnings.filterwarnings('ignore')

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROCKET_DIR)
from src.rocket_kernels import (apply_kernels, apply_kernels_variable, generate_kernels,  # noqa: E402
                                pad_with_noise, resample, seed_numba, znorm)

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'compare_unequal_handling')

N_REPEATS = 10
MIN_LEN = 100
NUM_KERNELS = 10000
NOISE_SCALE = 1e-3


def classifier():
    return make_pipeline(StandardScaler(with_mean=False),
                         RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))


def rocket_fixed(X_tr, X_te, seed):
    """長さを揃えた 2 次元配列に公式 ROCKET を適用する。"""
    seed_numba(seed)
    kernels = generate_kernels(X_tr.shape[1], NUM_KERNELS)
    return apply_kernels(X_tr, kernels), apply_kernels(X_te, kernels)


def main():
    X_tr, y_tr = load_arrow_head(split='train', return_X_y=True)
    X_te, y_te = load_arrow_head(split='test', return_X_y=True)
    tr = [np.asarray(s, dtype=np.float64) for s in X_tr.iloc[:, 0]]
    te = [np.asarray(s, dtype=np.float64) for s in X_te.iloc[:, 0]]
    y_tr, y_te = np.asarray(y_tr), np.asarray(y_te)
    print(f'ArrowHead 公式分割: 学習 {len(tr)} / テスト {len(te)}、元の長さ {len(tr[0])}、'
          f'繰り返し {N_REPEATS} 回\n')

    rows = []
    for rep in range(N_REPEATS):
        rng = np.random.default_rng(rep)
        tr_u = [s[:rng.integers(MIN_LEN, len(s) + 1)] for s in tr]
        te_u = [s[:rng.integers(MIN_LEN, len(s) + 1)] for s in te]
        tr_len = np.array([len(s) for s in tr_u])
        max_len, med_len, min_len = int(tr_len.max()), int(np.median(tr_len)), int(tr_len.min())

        # ③ では、学習の最短より短いテスト系列をノイズでパディングして適用する
        short = sum(len(s) < min_len for s in te_u)

        def baseline():
            return rocket_fixed(np.vstack([znorm(s) for s in tr]),
                                np.vstack([znorm(s) for s in te]), rep)

        def noise_pad():
            return rocket_fixed(pad_with_noise(tr_u, max_len, NOISE_SCALE, seed=rep),
                                pad_with_noise(te_u, max_len, NOISE_SCALE, seed=rep + 1000), rep)

        def interp():
            return rocket_fixed(resample(tr_u, med_len), resample(te_u, med_len), rep)

        def variable():
            seed_numba(rep)
            kernels = generate_kernels(min_len, NUM_KERNELS)
            F_tr, _ = apply_kernels_variable([znorm(s) for s in tr_u], kernels, min_len,
                                             NOISE_SCALE, seed=rep)
            F_te, _ = apply_kernels_variable([znorm(s) for s in te_u], kernels, min_len,
                                             NOISE_SCALE, seed=rep + 1000)
            return F_tr, F_te

        def minirocket_variable():
            mr = MiniRocketMultivariateVariable(num_kernels=NUM_KERNELS, random_state=rep,
                                                pad_value_short_series=None)
            nested = lambda ss: pd.DataFrame({'dim_0': [pd.Series(s) for s in ss]})  # noqa: E731
            return mr.fit_transform(nested(tr_u)).to_numpy(), mr.transform(nested(te_u)).to_numpy()

        def truncate_min():
            # 末尾を切って min_len に揃える。学習の最短より短いテスト系列はノイズでパディングする
            return rocket_fixed(pad_with_noise(tr_u, min_len, NOISE_SCALE, seed=rep),
                                pad_with_noise(te_u, min_len, NOISE_SCALE, seed=rep + 1000), rep)

        methods = {
            '基準(長さ251)': baseline,
            '① ノイズパディング': noise_pad,
            '② 補間': interp,
            '③ 長さを変えず適用': variable,
            '④ MiniRocketVariable': minirocket_variable,
            '[参考] 最短に切り揃え': truncate_min,
        }

        for name, transform in methods.items():
            t0 = time.time()
            F_tr, F_te = transform()
            model = classifier().fit(F_tr, y_tr)
            acc = accuracy_score(y_te, model.predict(F_te))
            rows.append(dict(repeat=rep, method=name, accuracy=acc, seconds=time.time() - t0,
                             train_max_len=max_len, train_median_len=med_len,
                             train_min_len=min_len, test_shorter_than_min=short))
        print(f'  繰り返し {rep + 1}/{N_REPEATS} 完了 '
              f'(学習の長さ 最短{min_len} / 中央{med_len} / 最長{max_len}、'
              f'学習の最短より短いテスト系列 {short} 本)')

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False, encoding='utf-8-sig')

    summary = (df.groupby('method', sort=False)
                 .agg(accuracy_mean=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std'),
                      seconds_mean=('seconds', 'mean')))
    print('\n' + summary.round(3).to_string())
    print(f'\n出力: {os.path.join(OUT_DIR, "results.csv")}')


if __name__ == '__main__':
    main()
