"""
長さの揃わない時系列を ROCKET 系の手法に入力できるかを確認する。

ArrowHead(公式分割: 学習36 / テスト175、全系列が長さ251)の各系列を、
乱数で決めた長さ L(100〜251)に先頭から切り詰めて不揃いにし、以下を試す。

  [基準] 長さを揃えたままの ArrowHead に Rocket + Ridge
  1. RocketClassifier                       (nested_univ)
  2. Rocket 変換器 + Ridge                   (nested_univ / pd-multiindex / df-list)
  3. MiniRocketMultivariateVariable + Ridge (nested_univ)
  4. 公式実装 external/rocket を1系列ずつ適用 + Ridge
  5. 全系列を最短の長さに切り揃えて Rocket + Ridge (長さを揃える案の例)

成功したものは Accuracy、失敗したものは例外の種類とメッセージを表示する。
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_arrow_head
from sktime.transformations.rocket import MiniRocketMultivariateVariable, Rocket

warnings.filterwarnings('ignore')

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'external', 'rocket', 'code'))

SEED = 42
MIN_LEN = 100
NUM_KERNELS = 10000


def ridge():
    return RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))


def to_arrays(X):
    """nested_univ の X を 1 次元配列のリストにする。"""
    return [np.asarray(s, dtype=np.float64) for s in X.iloc[:, 0]]


def make_unequal(series, rng):
    """各系列を長さ L ~ U[MIN_LEN, 元の長さ] に先頭から切り詰める。"""
    return [s[:rng.integers(MIN_LEN, len(s) + 1)] for s in series]


def to_nested(series):
    return pd.DataFrame({'dim_0': [pd.Series(s) for s in series]})


def to_multiindex(series):
    idx = pd.MultiIndex.from_tuples(
        [(i, t) for i, s in enumerate(series) for t in range(len(s))], names=['instance', 'time'])
    return pd.DataFrame({'dim_0': np.concatenate(series)}, index=idx)


def to_dflist(series):
    return [pd.DataFrame({'dim_0': s}) for s in series]


@njit
def generate_kernels(input_length, num_kernels):
    """公式実装 rocket_functions.generate_kernels と同じ処理(公式版は import 時に失敗するため写した)。"""
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)
    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]
        _weights = np.random.normal(0, 1, _length)
        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()
        biases[i] = np.random.uniform(-1, 1)
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        a1 = b1
    return weights, lengths, biases, dilations, paddings


@njit
def seed_numba(seed):
    # numba の乱数は NumPy とは別の状態を持つので、njit 関数の中で seed する
    np.random.seed(seed)


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    """公式実装 rocket_functions.apply_kernel と同じ処理。
    公式版は NumPy 2.0 で削除された np.NINF を使うため、-np.inf に置き換えただけ。"""
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0
    _max = -np.inf
    end = (input_length + padding) - ((length - 1) * dilation)
    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if index > -1 and index < input_length:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation
        if _max < _sum:
            _max = _sum
        if _sum > 0:
            _ppv += 1
    return _ppv / output_length, _max


def official_transform(series, kernels, kernel_fn):
    """公式実装のカーネルを1系列ずつ、長さを変えずに適用する。"""
    weights, lengths, biases, dilations, paddings = kernels
    out = np.zeros((len(series), len(lengths) * 2))
    for i, s in enumerate(series):
        a1 = 0
        for j in range(len(lengths)):
            b1 = a1 + lengths[j]
            out[i, 2 * j:2 * j + 2] = kernel_fn(
                s, weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])
            a1 = b1
    return out


def run(name, fn):
    t0 = time.time()
    try:
        acc = fn()
        print(f'  [成功] {name}: Accuracy {acc:.3f} ({time.time() - t0:.1f}秒)')
    except Exception as e:
        msg = str(e).splitlines()[0][:150]
        print(f'  [失敗] {name}: {type(e).__name__}: {msg}')


def main():
    X_tr, y_tr = load_arrow_head(split='train', return_X_y=True)
    X_te, y_te = load_arrow_head(split='test', return_X_y=True)
    tr, te = to_arrays(X_tr), to_arrays(X_te)

    rng = np.random.default_rng(SEED)
    tr_u, te_u = make_unequal(tr, rng), make_unequal(te, rng)
    lens = np.array([len(s) for s in tr_u + te_u])
    print(f'ArrowHead 公式分割: 学習 {len(tr)} / テスト {len(te)}、元の長さ {len(tr[0])}')
    print(f'不揃いにした後の長さ: 最小 {lens.min()} / 中央 {int(np.median(lens))} / 最大 {lens.max()}\n')

    def fit_eval(model, Xa, Xb):
        model.fit(Xa, y_tr)
        return accuracy_score(y_te, model.predict(Xb))

    def rocket_pipe():
        return make_pipeline(Rocket(num_kernels=NUM_KERNELS, random_state=SEED),
                             StandardScaler(with_mean=False), ridge())

    print('[基準] 長さが揃ったまま')
    run('Rocket + Ridge (nested_univ, 長さ251)',
        lambda: fit_eval(rocket_pipe(), to_nested(tr), to_nested(te)))

    print('\n[不揃いな長さ]')
    run('1. RocketClassifier (nested_univ)',
        lambda: fit_eval(RocketClassifier(rocket_transform='rocket', num_kernels=NUM_KERNELS,
                                          random_state=SEED),
                         to_nested(tr_u), to_nested(te_u)))
    for fmt, conv in (('nested_univ', to_nested), ('pd-multiindex', to_multiindex),
                      ('df-list', to_dflist)):
        run(f'2. Rocket 変換器 + Ridge ({fmt})',
            lambda conv=conv: fit_eval(rocket_pipe(), conv(tr_u), conv(te_u)))

    run('3. MiniRocketMultivariateVariable + Ridge (nested_univ)',
        lambda: fit_eval(make_pipeline(
            MiniRocketMultivariateVariable(num_kernels=NUM_KERNELS, random_state=SEED,
                                           pad_value_short_series=None),
            StandardScaler(with_mean=False), ridge()), to_nested(tr_u), to_nested(te_u)))

    # 4. 公式実装。dilation の上限は generate_kernels の input_length で決まる。
    #    最長に合わせると短い系列でカーネルがはみ出し output_length <= 0 になり得るので、最短の長さを渡す。
    def import_official():
        import rocket_functions  # noqa: F401  import 時に numba のコンパイルが走る
        return float('nan')

    run('4a. 公式 rocket_functions を import', import_official)

    seed_numba(SEED)
    kernels = generate_kernels(int(lens.min()), NUM_KERNELS)

    def official():
        model = make_pipeline(StandardScaler(with_mean=False), ridge())
        model.fit(official_transform(tr_u, kernels, apply_kernel), y_tr)
        return accuracy_score(y_te, model.predict(official_transform(te_u, kernels, apply_kernel)))

    run('4b. 公式実装(np.NINF → -np.inf)を1系列ずつ適用 + Ridge', official)

    m = int(lens.min())
    run(f'5. 最短の長さ {m} に切り揃えて Rocket + Ridge',
        lambda: fit_eval(rocket_pipe(), to_nested([s[:m] for s in tr_u]),
                         to_nested([s[:m] for s in te_u])))


if __name__ == '__main__':
    main()
