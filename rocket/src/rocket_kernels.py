"""
ROCKET のカーネル生成・適用と、長さの揃わない系列の前処理。

generate_kernels / apply_kernel / apply_kernels は公式実装
(external/rocket/code/rocket_functions.py, Dempster et al. 2020, GPL-3.0)の写し。
公式版は NumPy 2.0 で削除された np.NINF を使っていて import 時に失敗するため、
-np.inf に置き換えた点だけが異なる。

ROCKET のカーネルの重みとバイアスは、入力が z 正規化されている前提で決められている(論文 3.1 節)。
長さの揃わない系列は、本来の長さで z 正規化してから長さを揃えること。
"""
import numpy as np
from numba import njit, prange


@njit
def seed_numba(seed):
    """numba の乱数は NumPy とは別の状態を持つので、njit 関数の中で seed する。"""
    np.random.seed(seed)


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):

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


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf  # 公式版は np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))",
      parallel=True, fastmath=True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)  # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
                apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X


def apply_kernels_variable(series_list, kernels, min_len, noise_scale=1e-3, seed=0):
    """長さの揃わない系列のリストに、カーネルを1系列ずつ元の長さのまま適用する。

    kernels は generate_kernels(min_len, ...) で作ること(min_len は学習データの最短の長さ)。
    min_len より短い系列ではパディングなしのカーネルの出力長が 0 以下になり、ゼロ除算になる。
    そのような系列だけ、末尾を低振幅ノイズでパディングして min_len にする。
    series_list は z 正規化済みであること。戻り値は (特徴量, パディングした系列の本数)。
    """
    rng = np.random.default_rng(seed)
    out, n_padded = [], 0
    for s in series_list:
        s = np.asarray(s, dtype=np.float64)
        if len(s) < min_len:
            s = np.concatenate([s, rng.normal(0, noise_scale, min_len - len(s))])
            n_padded += 1
        out.append(apply_kernels(np.ascontiguousarray(s)[None, :], kernels))
    return np.vstack(out), n_padded


def znorm(s):
    s = np.asarray(s, dtype=np.float64)
    return (s - s.mean()) / (s.std() + 1e-8)


def pad_with_noise(series_list, max_len, noise_scale=1e-3, seed=0):
    """各系列を本来の長さで z 正規化し、末尾を低振幅ノイズでパディングして max_len に揃える。
    max_len より長い系列は末尾を切る。UCR 2018 の可変長データの調整版(論文 4.1.3 節)と同じ考え方。"""
    rng = np.random.default_rng(seed)
    X = np.empty((len(series_list), max_len), dtype=np.float64)
    for i, s in enumerate(series_list):
        s = znorm(s)[:max_len]
        X[i, :len(s)] = s
        X[i, len(s):] = rng.normal(0, noise_scale, max_len - len(s))
    return X


def resample(series_list, length):
    """各系列を z 正規化し、線形補間で length 点に伸縮する。時間軸の絶対的なスケールは失われる。"""
    grid = np.linspace(0, 1, length)
    return np.vstack([np.interp(grid, np.linspace(0, 1, len(s)), znorm(s)) for s in series_list])
