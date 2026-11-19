"""
段階17: MultiRocket のプーリング(PPV・MPV・MIPV・LSPV)による特徴量の比較(EXP-026)。

MiniRocket は畳み込みの出力から PPV(正の値の割合)しか取り出さない。MultiRocket(Tan et al., 2022)はさらに
MPV(正の値の平均)、MIPV(正の値の位置の平均)、LSPV(正の値が続いた最長の長さ)を取り出す。
反応の強さ・時期・長さの情報が使えるようになり、AUC が上がるかを確かめる。

  MR4_diff  主。1次差分(59点)だけに MultiRocket の4種類のプーリング。sktime の MultiRocket は「入力の系列」と
            「内部で作る1次差分」の両方を使うので、1次差分を入力として渡し、入力の系列の部分(4種類)だけを取り出す。
            num_kernels=2520 → 84 × 30 = 2520 組 × 4 = 10080 特徴量(M_diff の 9996 とほぼ同じ)
  MR_std    参考1。生の値(60点)に sktime の MultiRocket の既定値(元の系列 + 1次差分 × 4種類、約5万)。置換200回
  参考2     MR4_diff の特徴量の部分集合(PPV / PPV+MPV / PPV+MIPV / PPV+LSPV / 4種類)。置換なし、記述のみ
  M_diff    比べる相手。段階13の median・M(1次差分 → MiniRocket の PPV)

分類器・分割・置換は段階13〜16と同じ。判定(事前に固定): MR4_diff − M_diff の繰り返し平均の 2.5〜97.5%点が
0より上、かつ MR4_diff の全体の AUC が帰無分布の95%点を上回れば、MR4_diff を主の設定に採用する。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage17\\run_stage17.py
  試運転: ... run_stage17.py --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage17/
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13', 'stage14', 'stage15', 'stage16']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage12 as s12  # noqa: E402
import run_stage13 as s13  # noqa: E402
import run_stage14 as s14  # noqa: E402
import run_stage15 as s15  # noqa: E402
import run_stage16 as s16  # noqa: E402
from run_stage6 import BASE_ALPHAS, RidgeNTrain  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY = 'median'
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage17')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
TYPES = ['W', 'B', 'X']
K_MR4 = 2520            # MultiRocket の num_kernels(= 1つの系列表現・1種類のプーリングあたりの特徴量の数)
K_STD = 6250            # sktime の既定値
POOLS = ['PPV', 'LSPV', 'MPV', 'MIPV']  # sktime 1.2.0 の _transform での並び順(ソースで確認)
SUBSETS = {'PPV のみ': ['PPV'], 'PPV+MPV': ['PPV', 'MPV'], 'PPV+MIPV': ['PPV', 'MIPV'],
           'PPV+LSPV': ['PPV', 'LSPV'], '4種類すべて': ['PPV', 'LSPV', 'MPV', 'MIPV']}
CFG_LABEL = {'M_diff': 'M_diff MiniRocket PPV(段階13)', 'MR4_diff': 'MR4_diff 1次差分 × 4種類',
             'MR_std': 'MR_std 既定値(参考、置換200)'}
CFG_COLOR = {'M_diff': '#9a9890', 'MR4_diff': '#2a78d6', 'MR_std': '#eb6834'}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- 変換器
def n_per_block(num_kernels):
    """1つの系列表現・1種類のプーリングあたりの特徴量の数(84 × (num_kernels // 84))。"""
    return 84 * max(num_kernels // 84, 1)


class MR(BaseEstimator, TransformerMixin):
    """sktime 1.2.0 の MultiRocket。

    part='input': 入力の系列の部分(PPV | LSPV | MPV | MIPV、各 n_per_block)だけを返す。内部で作る差分の部分は捨てる。
    part='all'  : すべて(入力の系列 4種類 + 内部の1次差分 4種類)。
    """

    def __init__(self, num_kernels=K_MR4, part='input', random_state=0):
        self.num_kernels, self.part, self.random_state = num_kernels, part, random_state

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MultiRocket
        self.t_ = MultiRocket(num_kernels=self.num_kernels, random_state=self.random_state).fit(np.asarray(X, float))
        return self

    def transform(self, X):
        F = np.asarray(self.t_.transform(np.asarray(X, float)), dtype=float)
        nb = n_per_block(self.num_kernels)
        assert F.shape[1] == 8 * nb, (F.shape, nb)
        return F[:, :4 * nb] if self.part == 'input' else F


def mr_pipeline(cfg, seed):
    t = MR(K_MR4, 'input', seed) if cfg == 'MR4_diff' else MR(K_STD, 'all', seed)
    return make_pipeline(t, StandardScaler(with_mean=False), RidgeNTrain())


def block_cols(nb, pools):
    return np.concatenate([np.arange(POOLS.index(p) * nb, (POOLS.index(p) + 1) * nb) for p in pools])


# ---------------------------------------------------------------- 実装の確認
def implementation_check(Xd, log):
    from sktime.transformations.panel.rocket import MultiRocket
    from sktime.transformations.rocket._multirocket_numba import _transform
    lines = []

    def out(s):
        lines.append(s)
        log(s)

    out('[実装の確認] sktime 1.2.0 sktime/transformations/rocket/_multirocket.py・_multirocket_numba.py を読んだ結果')
    out('  1. 設計: _fit で入力 X のパラメータ(dilation・バイアス)と、np.diff(X, 1) のパラメータを別々に作る。'
        '_transform は _transform(X, X1 = np.diff(X, 1), parameter, parameter1, 4) を呼ぶ。'
        '論文どおり、入力の系列と内部の1次差分の両方を使う設計')
    out('  2. 並び順: features は (num_features + num_features1) × 4 列。前半(列 0〜4·nf−1)が入力の系列で、'
        '列 j(PPV)、j + nf(LSPV = max_stretch)、j + 2nf(MPV)、j + 3nf(MIPV)。後半は列 n_features_per_transform '
        '= 全列の半分 から同じ並びで内部の1次差分。nf = 84 × (num_kernels // 84) で、入力・差分とも同じ数')
    out('  3. 値: PPV = 正の数 / 長さ、LSPV = バイアス未満の値の間隔の最大(正の値が続いた最長)、'
        'MPV = Σ(C + bias) / 正の数(正が無ければ 0)、MIPV = 正の位置の平均(正が無ければ −1)。'
        '最後に np.nan_to_num をかける。MPV の和が C − bias ではなく C + bias なのは sktime の実装のまま(論文の式とは違う)')
    out('  4. 入力の系列のパラメータ(バイアス)は X だけから作られ、内部の差分のパラメータとは独立')
    # 4-1 形と列数
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(12, 1, Xd.shape[-1]))
    t = MultiRocket(num_kernels=K_MR4, random_state=0).fit(Xs)
    F = np.asarray(t.transform(Xs), float)
    nb = n_per_block(K_MR4)
    ok_shape = F.shape[1] == 8 * nb
    out(f'  確認 列数: MultiRocket(num_kernels={K_MR4}) の出力 {F.shape[1]} 列 = 8 × {nb}: {ok_shape}')
    # 4-2 差分の差分の部分だけを変えても、取り出した特徴量は変わらない
    X2 = Xs[:, 0].astype(np.float64)
    X1a = np.diff(X2, 1)
    X1b = rng.normal(size=X1a.shape)
    Fa = _transform(X2, X1a, t.parameter, t.parameter1, 4)
    Fb = _transform(X2, X1b, t.parameter, t.parameter1, 4)
    half = Fa.shape[1] // 2
    ok_in = np.array_equal(Fa[:, :4 * nb], Fb[:, :4 * nb])
    ch_diff = not np.array_equal(Fa[:, half:], Fb[:, half:])
    ok_half = half == 4 * nb
    out(f'  確認 内部の差分の入力だけを乱数に入れ替えると: 前半(入力の系列)は完全に同じ {ok_in}、後半は変わる {ch_diff}、'
        f'前半の列数 = 4 × nf {ok_half}')
    ok_same = np.array_equal(np.nan_to_num(Fa), F.astype(np.float32))
    out(f'  確認 numba の _transform を直接呼んだ結果 = sktime の transform の結果: {ok_same}')
    # 4-3 並び順(値の範囲)
    Fin = np.asarray(MultiRocket(num_kernels=K_MR4, random_state=0).fit(Xd).transform(Xd), float)[:, :4 * nb]
    B = {p: Fin[:, i * nb:(i + 1) * nb] for i, p in enumerate(POOLS)}
    L = Xd.shape[-1]
    r_ppv = bool(((B['PPV'] >= 0) & (B['PPV'] <= 1)).all())
    r_lspv = bool((np.mod(B['LSPV'], 1) == 0).all() and (B['LSPV'] >= 0).all() and (B['LSPV'] < L).all())
    r_mipv = bool(((B['MIPV'] >= -1) & (B['MIPV'] < L)).all())
    zero = B['PPV'] == 0
    r_zero = bool((B['MPV'][zero] == 0).all() and (B['MIPV'][zero] == -1).all())
    out(f'  確認 並び順(実データの1次差分 123件、乱数0): PPV は 0〜1 {r_ppv}、LSPV は 0 以上 {L} 未満の整数 {r_lspv}、'
        f'MIPV は −1 以上 {L} 未満 {r_mipv}、PPV = 0 の所で MPV = 0・MIPV = −1 {r_zero}')
    nan, inf = int(np.isnan(Fin).sum()), int(np.isinf(Fin).sum())
    out(f'  確認 値: NaN {nan}、inf {inf}。PPV = 0(正の値が無い)の割合 {zero.mean():.3f}'
        f'(このとき MPV は 0、MIPV は −1 で埋められる。0 での割り算は起きない)')
    ok = ok_shape and ok_in and ch_diff and ok_half and ok_same and r_ppv and r_lspv and r_mipv and r_zero
    out(f'  → 取り出し方: 1次差分(59点)を MultiRocket に入力し、出力の列 0〜{4 * nb - 1}(入力の系列 × 4種類、'
        f'{4 * nb} 特徴量)を使う。確認はすべて {ok}')
    with open(os.path.join(OUT_DIR, 'implementation_check.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    assert ok, '実装の確認が通らない。止めて報告する'
    assert nan == 0 and inf == 0, 'NaN・inf が残る。止めて報告する'
    return {'frac_ppv_zero': zero.mean()}


# ---------------------------------------------------------------- fold ごと(特徴量は worker の中だけで持つ)
def _fold(cfg, seed, X, tr, te, y, Yp, with_subsets):
    warnings.filterwarnings('ignore')
    tf = make_pipeline(mr_pipeline(cfg, seed).steps[0][1], StandardScaler(with_mean=False)).fit(X[tr])
    Ftr, Fte = tf.transform(X[tr]), tf.transform(X[te])
    dec, ok, n_redo = s12._m_fold(Ftr, Fte, y[tr], Yp[:, tr])
    clf = RidgeNTrain().fit(Ftr, y[tr])
    ref = {'n_test': len(te), 'n_train': len(tr), 'alpha_over_n': clf.alpha_ / len(tr),
           'bacc': balanced_accuracy_score(y[te], clf.predict(Fte)),
           'auc': roc_auc_score(y[te], clf.decision_function(Fte)) if len(np.unique(y[te])) == 2 else np.nan,
           'n_features': Ftr.shape[1]}
    sub = {}
    if with_subsets:
        nb = n_per_block(K_MR4)
        for name, pools in SUBSETS.items():
            c = block_cols(nb, pools)
            m = RidgeNTrain().fit(Ftr[:, c], y[tr])
            sub[name] = {'n_features': len(c), 'alpha_over_n': m.alpha_ / len(tr),
                         'bacc': balanced_accuracy_score(y[te], m.predict(Fte[:, c])),
                         'auc': roc_auc_score(y[te], m.decision_function(Fte[:, c]))
                         if len(np.unique(y[te])) == 2 else np.nan}
    return dec, ok, n_redo, ref, sub


def _fold_check(cfg, seed, X, tr, te, y, Yp):
    """確認3 用: まとめて解いた置換と sklearn で1回ずつ学習した置換。"""
    warnings.filterwarnings('ignore')
    tf = make_pipeline(mr_pipeline(cfg, seed).steps[0][1], StandardScaler(with_mean=False)).fit(X[tr])
    Ftr, Fte = tf.transform(X[tr]), tf.transform(X[te])
    dec, ok, n_redo = s12._m_fold(Ftr, Fte, y[tr], Yp[:, tr])
    return dec, s12._m_fold_sklearn(Ftr, Fte, Yp[:, tr]), n_redo


def prechecks(Xd, meta, sp, log, n_perm=20):
    y = meta.y.values
    # 2. Pipeline と特徴量の使い回し
    a = evaluate(mr_pipeline('MR4_diff', 0), Xd, y, sp[:10])
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold)('MR4_diff', 0, Xd, tr, te, y, np.zeros((0, len(y))), False)
                                  for r, f, tr, te in sp[:10])
    b = pd.DataFrame([x[3] for x in res])
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'  確認2 Pipeline と特徴量の使い回しの一致(MR4_diff、乱数0、先頭10 fold): {ok}')
    assert ok
    # 4. 置換しても分割が変わらない
    ps = s13.perm_sets(KEY, meta, n_perm)
    same = True
    for i in range(5):
        sp2 = make_splits(ps[0][i], meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
        same &= all(np.array_equal(t1, t2) and np.array_equal(e1, e2) for (_, _, t1, e1), (_, _, t2, e2) in zip(sp, sp2))
    log(f'  確認4 置換(市町村×年の中)5回分で分割が変わらない: {same}')
    assert same
    # 3. まとめて解いた置換 = sklearn で1回ずつ
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold_check)('MR4_diff', 0, Xd, tr, te, y, ps[2]) for r, f, tr, te in sp)
    _, na = s13.analyse([d for d, _, _ in res], meta, sp, ps)
    _, nb = s13.analyse([np.vstack([d[:1], rr]) for d, rr, _ in res], meta, sp, ps)
    ok = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'  確認3 置換{n_perm}回分の帰無分布(全体・W・B): まとめて解いた結果 = sklearn で1回ずつ: {ok}'
        f'(学習し直した組 {sum(k for *_, k in res)}、{time.time() - t:.1f}s)')
    assert ok


# ---------------------------------------------------------------- 1つの設定
def run_cfg(cfg, X, meta, sp, ps, seeds, P, log, with_subsets=False, tag=''):
    y = meta.y.values
    name = cfg + tag
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': [], 'sub': []}
    nulls = {'all': [], 'W': [], 'B': []}
    S = np.full((len(seeds), N_REPEATS, len(y)), np.nan)
    t0 = time.time()
    for k, seed in enumerate(seeds):
        res = Parallel(n_jobs=N_JOBS)(delayed(_fold)(cfg, seed, X, tr, te, y, ps[2], with_subsets)
                                      for r, f, tr, te in sp)
        assert all(ok for _, ok, _, _, _ in res), '観測ラベルでの alpha・決定関数が sklearn と一致しない fold がある'
        decs = [d for d, *_ in res]
        n_redo = sum(x[2] for x in res)
        ref = pd.DataFrame([dict(repeat=r, fold=f, **x[3]) for (r, f, _, _), x in zip(sp, res)])
        obs, null = s13.analyse(decs, meta, sp, ps)
        s13.check_obs(f'{name} 乱数{seed}', obs, ref, log)  # 確認5
        out['folds'].append(ref.assign(config=name, seed=seed))
        if with_subsets:
            for (r, f, _, _), x in zip(sp, res):
                for sn, v in x[4].items():
                    out['sub'].append({'subset': sn, 'seed': seed, 'repeat': r, 'fold': f, **v})
        for kk in nulls:
            nulls[kk].append(null[kk])
        for (r, f, tr, te), dec in zip(sp, decs):
            S[k, r, te] = dec[0]
            m = meta.iloc[te]
            out['scores'].append(pd.DataFrame({'config': name, 'seed': seed, 'repeat': r, 'fold': f,
                                               'field_id': m.field_id.values, 'year': m.year.values,
                                               'muniCd': m.muniCd.values, 'muniyear': m.muniyear.values,
                                               'y': m.y.values, 'score': dec[0]}))
        row = {'config': name, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
               'alpha_at_max': np.isclose(ref.alpha_over_n, BASE_ALPHAS[-1]).mean(), 'n_redo': n_redo,
               'n_features': int(ref.n_features.iloc[0])}
        for t in TYPES:
            a, n = obs[t]
            row[t] = np.nanmean(a)
            out['pair'].append(pd.DataFrame({'config': name, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
        out['seed'].append(row)
        log(f'    {name} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
            + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES)
            + f'(特徴量 {row["n_features"]}、alpha 上限 {row["alpha_at_max"]:.0%}、学習し直した組 {n_redo}、'
              f'{time.time() - t0:.0f}s)')
    per_seed = pd.DataFrame(out['seed'])
    pa = pd.concat(out['pair'])
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    np.save(os.path.join(OUT_DIR, f'null_{name}_all.npy'), null_all)
    base = {'config': name, 'n': len(y), 'n_seeds': len(per_seed), 'n_perm': P,
            'n_features': int(per_seed.n_features.iloc[0]), 'alpha_at_max': per_seed.alpha_at_max.mean(),
            'source': '段階17'}
    summ = []
    for met, j in [('all_auc', 1), ('all_bacc', 0)]:
        v, nl = per_seed[met], null_all[:, j]
        summ.append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                     'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                     'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
    for t in TYPES:
        v = per_seed[t]
        r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
             'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean()}
        if t in ('W', 'B'):
            nl = np.mean(np.stack(nulls[t]), axis=0)
            np.save(os.path.join(OUT_DIR, f'null_{name}_{t}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                      'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
        summ.append(r)
    s = pd.DataFrame(summ)
    log(f'  {name} ' + '  '.join(
        f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95')) else '')
        for _, r in s.iterrows()) + f'  [{time.time() - t0:.0f}s]')
    out['summary'], out['S'] = summ, S
    return out


# ---------------------------------------------------------------- 図
def plot(summary, abl):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [3, 2]})
    ax = axs[0]
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内'), ('B', 'B 地域間')]
    cfgs = [c for c in ['M_diff', 'MR4_diff', 'MR_std'] if c in set(summary.config)]
    sm = summary.set_index(['config', 'metric'])
    w = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        xs = np.arange(len(mets)) + (k - (len(cfgs) - 1) / 2) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.92, color=CFG_COLOR[c], alpha=0.85, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95']
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(0.4, max(0.75, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('設定ごとの全体・W・B', fontsize=11, color=INK, loc='left')
    s6.style(ax)
    ax = axs[1]
    names = list(SUBSETS)
    rep = abl.groupby(['subset', 'seed', 'repeat']).auc.mean().groupby(['subset', 'repeat']).mean()
    data = [rep.loc[n].values for n in names]
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.2), boxprops=dict(lw=0), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED), flierprops=dict(marker='o', ms=2.5, mfc=MUTED, mec='none'))
    for b in bp['boxes']:
        b.set_facecolor('#2a78d6')
        b.set_alpha(0.45)
    ax.plot(range(len(names)), [d.mean() for d in data], 'D', color=INK, ms=5)
    m = sm.loc[('M_diff', 'all_auc'), 'mean']
    ax.axhline(m, color=CFG_COLOR['M_diff'], lw=1.5, ls='--', label=f'M_diff {m:.3f}')
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.set_xticks(range(len(names)), names, fontsize=9)
    ax.set_ylabel('全体の AUC(繰り返しごと)', color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title('参考2: MR4_diff のプーリングの内訳(記述のみ)', fontsize=11, color=INK, loc='left')
    s6.style(ax)
    fig.suptitle('段階17: MultiRocket のプーリング(label_muniyear_median、123件)', fontsize=13, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'stage17_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-perm-std', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-seeds-std', type=int, default=None, help='MR_std の乱数の数(既定は --n-seeds と同じ)')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    seeds_std = list(range(args.n_seeds_std if args.n_seeds_std is not None else args.n_seeds))
    P = args.n_perm
    s6.N_JOBS = N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {N_JOBS}')
    X, meta = s14.load()
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    split_table(sp, y, meta.muniCd.values).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    s13sp = pd.read_csv(os.path.join(ROCKET_DIR, 'outputs', 'stage13', 'splits.csv'), encoding='utf-8-sig',
                        dtype={'muniCd': str})
    mine = split_table(sp, y, meta.muniCd.values)
    s13sp = s13sp[s13sp.label == KEY].drop(columns='label').reset_index(drop=True)
    ok = len(mine) == len(s13sp) and np.allclose(mine.values.astype(float), s13sp[mine.columns].values.astype(float))
    log(f'データ: {len(meta)} 件。分割の表が段階13(outputs/stage13/splits.csv の median)と同じ: {ok}'
        '(テスト件そのものの一致は、下の段階13の oof_scores.csv の確認で見る)')
    assert ok

    log('\n[起動時の確認]')
    implementation_check(Xd, log)  # 確認1
    mfolds, S_m, seeds13, summ_m, pa_m = s16.m_diff_from_stage13(meta, sp, Xd, log)  # 確認6
    prechecks(Xd, meta, sp, log)  # 確認2・4・3
    ps = s13.perm_sets(KEY, meta, P)
    ps_std = s13.perm_sets(KEY, meta, args.n_perm_std)

    res = {}
    log('\n[MR4_diff]')
    res['MR4_diff'] = run_cfg('MR4_diff', Xd, meta, sp, ps, seeds, P, log, with_subsets=True)
    log('\n[MR_std(参考1)]')
    log('  特徴量のキャッシュは持たず、fold ごとに worker の中で特徴量を作って決定関数だけを返す(メモリのため)')
    res['MR_std'] = run_cfg('MR_std', X.astype(float), meta, sp, ps_std, seeds_std, args.n_perm_std, log)

    summary = pd.DataFrame([dict(r, n_features=9996) for r in summ_m] + [r for c in res for r in res[c]['summary']])
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    folds = pd.concat([f for c in res for f in res[c]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for c in res for r in res[c]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([s for c in res for s in res[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([pa_m] + [p for c in res for p in res[c]['pair']]).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'),
                                                                        index=False, encoding='utf-8-sig')

    # 対ごとの差(同じ分割・同じ乱数)
    cols = ['seed', 'repeat', 'fold', 'auc', 'bacc']
    fM = mfolds[cols]
    g = lambda c: folds[folds.config == c][cols]  # noqa: E731
    pdiff = pd.concat([s7.paired(g('MR4_diff'), fM[fM.seed.isin(seeds)], 'MR4_diff − M_diff'),
                       s7.paired(g('MR_std'), fM[fM.seed.isin(seeds_std)], 'MR_std − M_diff')])
    pdiff.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    pds = s7.paired_summary(pdiff)
    pds['positive_ci'] = pds.rep_q025 > 0
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差(同じ分割・同じ乱数)]')
    for _, r in pds.iterrows():
        log(f'  {r.comparison} {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{r.rep_q025:+.3f}, '
            f'{r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}')

    # W の差のブートストラップ
    boot, pt = s16.w_diff_bootstrap(res['MR4_diff']['S'], S_m[[seeds13.index(s) for s in seeds]], meta, sp, args.n_boot)
    boot = boot.rename(columns={'W_GC_date': 'W_MR4_diff'})
    boot.to_csv(os.path.join(OUT_DIR, 'w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    bq = boot.W_diff.quantile([.025, .975]).values
    log(f'\n[W の差 MR4_diff − M_diff(市町村のブートストラップ {args.n_boot}回)] 点推定 {pt["a"] - pt["b"]:+.3f}'
        f'(MR4_diff {pt["a"]:.3f}、M_diff {pt["b"]:.3f})、95%区間 [{bq[0]:+.3f}, {bq[1]:+.3f}]、'
        f'正の割合 {(boot.W_diff > 0).mean():.3f}')
    summary.loc[(summary.config == 'MR4_diff') & (summary.metric == 'W'), 'w_diff_boot_low'] = bq[0]
    summary.loc[(summary.config == 'MR4_diff') & (summary.metric == 'W'), 'w_diff_boot_high'] = bq[1]

    # 参考2: プーリングの内訳
    abl = pd.DataFrame(res['MR4_diff']['sub'])
    ab = []
    for sn in SUBSETS:
        d = abl[abl.subset == sn]
        per_seed = d.groupby('seed')[['auc', 'bacc']].mean()
        ab.append({'subset': sn, 'n_features': int(d.n_features.iloc[0]), 'auc': per_seed.auc.mean(),
                   'auc_seed_min': per_seed.auc.min(), 'auc_seed_max': per_seed.auc.max(), 'bacc': per_seed.bacc.mean(),
                   'alpha_at_max': np.isclose(d.alpha_over_n, BASE_ALPHAS[-1]).mean()})
    base_ppv = abl[abl.subset == 'PPV のみ'][cols]
    pab = s7.paired_summary(pd.concat([s7.paired(abl[abl.subset == sn][cols], base_ppv, f'{sn} − PPV のみ')
                                       for sn in SUBSETS if sn != 'PPV のみ']))
    ab = pd.DataFrame(ab)
    pa_auc = pab[pab.metric == 'auc'].set_index('comparison')
    for c in ['mean_diff', 'rep_q025', 'rep_q975', 'rep_frac_pos']:
        ab[f'vs_ppv_{c}'] = [pa_auc.loc[f'{sn} − PPV のみ', c] if sn != 'PPV のみ' else np.nan for sn in ab.subset]
    ok_all = np.isclose(ab.set_index('subset').loc['4種類すべて', 'auc'],
                        summary[(summary.config == 'MR4_diff') & (summary.metric == 'all_auc')]['mean'].iloc[0])
    ab.to_csv(os.path.join(OUT_DIR, 'pooling_ablation.csv'), index=False, encoding='utf-8-sig')
    log(f'\n[参考2 プーリングの内訳(MR4_diff の特徴量の部分集合、置換なし、記述のみ)] 「4種類すべて」= MR4_diff: {ok_all}')
    for _, r in ab.iterrows():
        log(f'  {r.subset}({r.n_features} 特徴量): AUC {r.auc:.3f}(乱数 {r.auc_seed_min:.3f}〜{r.auc_seed_max:.3f})、'
            f'bacc {r.bacc:.3f}、alpha 上限 {r.alpha_at_max:.0%}'
            + ('' if pd.isna(r.vs_ppv_mean_diff) else
               f'、− PPV のみ {r.vs_ppv_mean_diff:+.3f} [{r.vs_ppv_rep_q025:+.3f}, {r.vs_ppv_rep_q975:+.3f}]'))

    # 判定
    r = summary[(summary.config == 'MR4_diff') & (summary.metric == 'all_auc')].iloc[0]
    d = pds[(pds.comparison == 'MR4_diff − M_diff') & (pds.metric == 'auc')].iloc[0]
    c1, c2 = bool(d.rep_q025 > 0), bool(r['mean'] > r.null_q95)
    adopt = c1 and c2
    log(f'\n[採用の判定] 1. MR4_diff − M_diff の繰り返し平均の2.5%点 {d.rep_q025:+.3f} > 0: {c1}  '
        f'2. MR4_diff の全体 AUC {r["mean"]:.3f} > 95%点 {r.null_q95:.3f}: {c2} → '
        + ('採用する(MR4_diff を主の設定にする)' if adopt else '採用しない(主の設定は M_diff のまま)'))

    # 採用した場合のみ: 段階15の部分集合
    if adopt:
        log('\n[採用した場合のみ: 段階15の部分集合(品種が集団の中で同じ19集団・76件)で MR4_diff を学習し直す、置換200回]')
        groups = s15.classify_groups(meta)
        keep = meta.muniyear.isin(groups[groups.included].muniyear).values
        ms = meta[keep].reset_index(drop=True)
        assert len(ms) == s15.EXPECT['n']
        sps = make_splits(ms.y.values, ms.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
        r15 = run_cfg('MR4_diff', Xd[keep], ms, sps, s13.perm_sets(KEY, ms, 200), seeds, 200, log, tag='_subset15')
        summary = pd.concat([summary, pd.DataFrame(r15['summary'])], ignore_index=True)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')

    plot(summary, abl)
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'n_features', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
