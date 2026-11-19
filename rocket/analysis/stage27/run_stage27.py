"""
段階27: 入力の形の切り分け: 開花起点の相対値と、正負に分けた1次差分(EXP-036)。

主ラベル label_muniyear_median(段階13の123件、開花起点 0〜+59 日の生の VWC θ(t))で、入力の形だけを変える。
長さはすべて59点。
  D   1次差分 Δθ(t) = θ(t) − θ(t−1)(基準、段階13の M。保存済みの結果を使い、乱数0の先頭10 fold を計算し直して確認)
  A   開花起点の相対値 θ(t) − θ(0)、t = 1…59(Q1: 水準だけ消し、累積の軌跡は残す)
  P   正の差分 max(Δθ, 0)(Q2: 増える動きだけ)
  N   負の差分 min(Δθ, 0)(Q2: 減る動きだけ)
  PN  [P, N] の2チャンネル(sktime 1.2.0 の MiniRocketMultivariate、num_kernels=10000)
1チャンネルは sktime 1.2.0 の MiniRocket(num_kernels=10000)。どれも乱数0〜9 → StandardScaler(with_mean=False)
→ RidgeClassifierCV(logspace(−3, 3, 10) × n_train)。分割は make_splits(y, muniCd, 5, 20, 0)(段階13と同じ)。
置換: 200回、市町村×年の中で並べ替え(置換 i は [0, i])。D は段階13の置換1000回の結果を使う(再計算しない)。
差の区間: 各入力 − D の AUC の差を、市町村単位のクラスターブートストラップ(段階14と同じ重み付け、2000回)で。
判定(事前に固定):
  Q1  A > 95%点 かつ D − A の区間が0を含む → 1次差分の効果の大部分は水準のズレの除去で説明できる
      D − A の区間が0より上 → 変化そのものに水準除去以上の情報がある
  Q2  N > 95%点 かつ P ≦ 95%点 → 減る動き(乾燥)に情報がある(段階26の推測を支持)
      P > 95%点 かつ N ≦ 95%点 → 増える動き(雨への応答)に情報がある
      PN − D の区間が0より上 → 正負を分ける形を次の段階の候補に(採用は別途相談)
  4入力(A, P, N, PN)の帰無分布との比較に Holm 補正後の p を併記する。
新しい変換はこのスクリプトの中に書く(src/windows.py は変更しない)。
注意: sktime の MiniRocket はゼロパディングのため平行移動に不変でない(EXP-009)。A と生の値の違いはこの性質の影響も受ける。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage27\\run_stage27.py
  試運転: ... run_stage27.py --n-seeds 1 --n-perm 20 --n-boot 50
入出力: rocket/outputs/stage27/、新しい入力は rocket/data/processed/stage27/。入力ごとの途中結果を
outputs/stage27/partial_<入力>.pkl に保存し、同じ引数で実行し直すと読み込んで再開する。
"""
import argparse
import os
import pickle
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
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage13'))
import run_stage13 as s13  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

s6, s12 = s13.s6, s13.s12
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage27')
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage27')
DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13', 'label_muniyear_median')
S13 = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
KEY = 'median'
N_EXPECT = 123
INPUTS = ['A', 'P', 'N', 'PN']
NAMES = {'D': '1次差分', 'A': '開花起点の相対値 θ(t)−θ(0)', 'P': '正の差分(増える動き)', 'N': '負の差分(減る動き)',
         'PN': '正負の2チャンネル'}
TYPES = ['W', 'B', 'X']
RAW_REF = {'auc': 0.506, 'p': 0.46, 'q95': 0.586, 'src': '段階16(EXP-025)の M_raw'}
INK, MUTED = s6.INK, s6.MUTED
COL = {'D': '#52514e', 'A': '#9b59b6', 'P': '#2a78d6', 'N': '#eb6834', 'PN': '#3a9a5b'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 入力の変換(このスクリプトの中だけで使う)
def anchored(X):
    """θ(t) − θ(0)、t = 1…59。X: (n, 1, 60)。"""
    return X[..., 1:] - X[..., :1]


def pos_diff(X):
    return np.maximum(first_diff(X), 0.0)


def neg_diff(X):
    return np.minimum(first_diff(X), 0.0)


def pn_channels(X):
    return np.concatenate([pos_diff(X), neg_diff(X)], axis=1)


# ---------------------------------------------------------------- 特徴量
class _MV(BaseEstimator, TransformerMixin):
    """MiniRocketMultivariate(num_kernels=10000)を sklearn の Pipeline で使うための薄い包み。"""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        self.t_ = MiniRocketMultivariate(num_kernels=s6.N_KERNELS, random_state=self.seed).fit(np.asarray(X, float))
        return self

    def transform(self, X):
        return np.asarray(self.t_.transform(np.asarray(X, float)), dtype=float)


def _one_fold_mv(seed, X, tr, te):
    tf = make_pipeline(_MV(seed), StandardScaler(with_mean=False)).fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[te])


def fold_features(name, seed, X, sp):
    if name == 'PN':
        return Parallel(n_jobs=s13.N_JOBS)(delayed(_one_fold_mv)(seed, X, tr, te) for r, f, tr, te in sp)
    return s6.fold_features('minirocket', seed, X, sp)


def pipeline_eval(name, seed, X, y, sp):
    """Pipeline をそのまま回した fold ごとの AUC・bacc(確認用)。"""
    if name == 'PN':
        est = make_pipeline(_MV(seed), StandardScaler(with_mean=False), s6.RidgeNTrain())
        return evaluate(est, X, y, sp)
    return evaluate(s6.pipeline('minirocket', seed), X, y, sp)


# ---------------------------------------------------------------- 1つの入力
def run_input(name, X, meta, sp, seeds, P, log):
    y = meta.y.values
    ps = s13.perm_sets(KEY, meta, P)
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': [], 'summary': []}
    nulls = {'all': [], 'W': [], 'B': []}
    S = np.full((len(seeds), s6.N_REPEATS, len(y)), np.nan)
    t0 = time.time()
    for si, seed in enumerate(seeds):
        feats = fold_features(name, seed, X, sp)
        if si == 0:
            F0 = np.vstack(feats[0])
            allf = np.vstack([np.vstack(f) for f in feats[:10]])
            log(f'    特徴量(乱数{seed}、fold 0 の {F0.shape[0]} 件 × {F0.shape[1]} 列): 全件で同じ値の列 '
                f'{int((F0.std(axis=0) == 0).sum())}、先頭10 fold の NaN {int(np.isnan(allf).sum())}・'
                f'inf {int(np.isinf(allf).sum())}')
            assert np.isfinite(allf).all()
        ref = s6.eval_cached(feats, y, sp)
        decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
        del feats
        obs, null = s13.analyse(decs, meta, sp, ps)
        s13.check_obs(f'{name} 乱数{seed}', obs, ref, log)
        out['folds'].append(ref.assign(config=name, seed=seed))
        for k in nulls:
            nulls[k].append(null[k])
        for (r, f, tr, te), dec in zip(sp, decs):
            S[si, r, te] = dec[0]
            m = meta.iloc[te]
            out['scores'].append(pd.DataFrame({'config': name, 'seed': seed, 'repeat': r, 'fold': f,
                                               'field_id': m.field_id.values, 'year': m.year.values,
                                               'muniyear': m.muniyear.values, 'y': m.y.values, 'score': dec[0]}))
        row = {'config': name, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
               'alpha_at_max': np.isclose(ref.alpha_over_n, s6.BASE_ALPHAS[-1]).mean()}
        for t in TYPES:
            a, n = obs[t]
            row[t] = np.nanmean(a)
            out['pair'].append(pd.DataFrame({'config': name, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
        out['seed'].append(row)
        log(f'    {name} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
            + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES) + f'  alpha 上限 {row["alpha_at_max"]:.2f}(学習し直した組 {n_redo})')
    per_seed = pd.DataFrame(out['seed'])
    pa = pd.concat(out['pair'])
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    np.save(os.path.join(OUT_DIR, f'null_{name}.npy'), null_all)
    base = {'config': name, 'n': len(y), 'n_seeds': len(per_seed), 'n_perm': P}
    for met, j in [('all_auc', 1), ('all_bacc', 0)]:
        v, nl = per_seed[met], null_all[:, j]
        out['summary'].append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                               'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                               'p': (1 + np.sum(nl >= v.mean())) / (1 + P), 'alpha_at_max': per_seed.alpha_at_max.mean()})
    for t in TYPES:
        v = per_seed[t]
        r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
             'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean(),
             'alpha_at_max': per_seed.alpha_at_max.mean()}
        if t in ('W', 'B'):
            nl = np.mean(np.stack(nulls[t]), axis=0)
            np.save(os.path.join(OUT_DIR, f'null_{name}_{t}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                      'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
        out['summary'].append(r)
    out['S'] = S
    log(f'  {name:3s} ' + '  '.join(
        f'{r["metric"]} {r["mean"]:.3f}' + (f'(q95 {r["null_q95"]:.3f}, p={r["p"]:.3f})' if 'null_q95' in r else '')
        for r in out['summary']) + f'  [{time.time() - t0:.0f}s]')
    return out


# ---------------------------------------------------------------- D(段階13の保存済み)
def load_d(meta, sp, seeds, log):
    sm = pd.read_csv(os.path.join(S13, 'summary.csv'), encoding='utf-8-sig')
    sm = sm[(sm.label == KEY) & (sm.config == 'M') & sm.metric.isin(['all_auc', 'all_bacc'] + TYPES)].copy()
    sm['config'] = 'D'
    folds = pd.read_csv(os.path.join(S13, 'folds.csv'), encoding='utf-8-sig')
    folds = folds[(folds.label == KEY) & (folds.config == 'M') & folds.seed.isin(seeds)].drop(columns='label')
    folds['config'] = 'D'
    pair = pd.read_csv(os.path.join(S13, 'pair_auc.csv'), encoding='utf-8-sig')
    pair = pair[(pair.label == KEY) & (pair.config == 'M') & pair.seed.isin(seeds)].drop(columns='label')
    pair['config'] = 'D'
    sc = pd.read_csv(os.path.join(S13, 'oof_scores.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    sc = sc[(sc.label == KEY) & (sc.config == 'M') & sc.seed.isin(seeds)]
    idx = {(f, yr): i for i, (f, yr) in enumerate(zip(meta.field_id, meta.year))}
    S = np.full((len(seeds), s6.N_REPEATS, len(meta)), np.nan)
    si = sc.seed.map({s: k for k, s in enumerate(seeds)}).astype(int).values
    S[si, sc.repeat.values, [idx[(f, yr)] for f, yr in zip(sc.field_id, sc.year)]] = sc.score.values
    assert not np.isnan(S).any(), '段階13の oof_scores から D のスコアを組み立てられない'
    ps = pd.read_csv(os.path.join(S13, 'per_seed.csv'), encoding='utf-8-sig')
    ps = ps[(ps.label == KEY) & (ps.config == 'M') & ps.seed.isin(seeds)].drop(columns='label')
    ps['config'] = 'D'
    if len(seeds) < 10:   # 試運転: 乱数を絞った D の平均に置き換える(帰無分布は段階13のまま)
        for met, col in [('all_auc', 'all_auc'), ('all_bacc', 'all_bacc')] + [(t, t) for t in TYPES]:
            sm.loc[sm.metric == met, 'mean'] = ps[col].mean()
    log(f'  D: 段階13の保存済み(乱数 {len(seeds)} 通り、置換 {int(sm.n_perm.iloc[0])} 回)を使う')
    return sm.drop(columns='label'), folds, pair, S, ps


# ---------------------------------------------------------------- ブートストラップ
def pair_sums(S, meta, sp):
    """全乱数・全繰り返しで合算した、ペア (i 多収, j 低収) ごとの一致の和 C と採点回数 N(同じ fold のテスト件どうし)。"""
    n = len(meta)
    y = meta.y.values.astype(float)
    C, N = np.zeros((n, n)), np.zeros((n, n))
    for r, f, tr, te in sp:
        d = S[:, r, te]
        cmp = ((d[:, :, None] > d[:, None, :]) + 0.5 * (d[:, :, None] == d[:, None, :])).sum(axis=0)
        pn = y[te][:, None] * (1 - y[te][None, :])
        C[np.ix_(te, te)] += cmp * pn
        N[np.ix_(te, te)] += S.shape[0] * pn
    return C, N


def boot_diff(Cs, N, meta, n_boot):
    """市町村を単位にしたクラスターブートストラップ(段階14の重み付け)。
    違う市町村のペアは k_ci × k_cj、同じ市町村のペアは k_c(選ばれた1回ごとに1組)。Cs: 名前 → C。
    戻り値: ブートストラップごとの、各入力のペアの AUC(全ペア合算)。"""
    cd = meta.muniCd.to_numpy(dtype=str)
    units, cidx = np.unique(cd, return_inverse=True)
    same = cd[:, None] == cd[None, :]
    rng = np.random.default_rng(0)
    rows = []
    Nw, Nb = (N * same).sum(axis=1), N * ~same
    pre = {k: ((C * same).sum(axis=1), C * ~same) for k, C in Cs.items()}
    for b in range(n_boot):
        kk = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = kk[cidx].astype(float)
        nn = w @ Nw + w @ Nb @ w
        rows.append({'boot': b, **{k: (w @ cw + w @ cb @ w) / nn for k, (cw, cb) in pre.items()}})
    return pd.DataFrame(rows), {k: C.sum() / N.sum() for k, C in Cs.items()}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------- 図
def plot(summary, diffs):
    cfgs = ['D'] + INPUTS
    sm = summary.set_index(['config', 'metric'])
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={'width_ratios': [1.3, 1, 1, 1.2]})
    for ax, (met, nm) in zip(axs[:3], [('all_auc', '全体 AUC'), ('W', '地域内 W(記述)'), ('B', '地域間 B(記述)')]):
        for i, c in enumerate(cfgs):
            r = sm.loc[(c, met)]
            ax.bar(i, r['mean'], color=COL[c], alpha=0.75)
            ax.text(i, r['mean'] + 0.006, f'{r["mean"]:.3f}', ha='center', fontsize=8.5, color=INK)
            if pd.notna(r.get('null_q95')):
                ax.hlines(r['null_q95'], i - 0.4, i + 0.4, color=s6.C_RED, lw=2, ls='--')
        if met == 'all_auc':
            ax.bar(len(cfgs), RAW_REF['auc'], color='#c9c7c2', alpha=0.9)
            ax.text(len(cfgs), RAW_REF['auc'] + 0.006, f'{RAW_REF["auc"]:.3f}', ha='center', fontsize=8.5, color=INK)
            ax.hlines(RAW_REF['q95'], len(cfgs) - 0.4, len(cfgs) + 0.4, color=s6.C_RED, lw=2, ls='--')
            ax.set_xticks(range(len(cfgs) + 1), cfgs + ['生(参考)'])
        else:
            ax.set_xticks(range(len(cfgs)), cfgs)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylim(0.35, 0.72)
        ax.set_title(nm, loc='left', fontsize=11, color=INK)
        s6.style(ax)
    axs[0].set_ylabel('AUC')
    ax = axs[3]
    d = diffs.set_index('comparison')
    labels = list(d.index)
    for i, c in enumerate(labels):
        r = d.loc[c]
        ax.errorbar(r.point_fold_mean, i, xerr=[[r.point_fold_mean - r.boot_q025], [r.boot_q975 - r.point_fold_mean]],
                    fmt='o', color=INK, capsize=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('AUC の差(点 = fold 平均の差、線 = ブートストラップ95%区間)')
    ax.set_title('D との差', loc='left', fontsize=11, color=INK)
    s6.style(ax)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=s6.C_RED, lw=2, ls='--')], ['帰無分布の95%点'], loc='upper right', frameon=False,
               fontsize=9)
    fig.suptitle('段階27: 入力の形の切り分け(D 1次差分 / A 開花起点の相対値 / P 正の差分 / N 負の差分 / PN 2チャンネル)',
                 fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'stage27_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--fresh', action='store_true', help='途中結果を使わずに計算し直す')
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    P = args.n_perm
    s6.N_JOBS = s13.N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(NEW_DATA, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {s13.N_JOBS}')

    # ---- データ・分割・入力
    X = np.load(DATA + '_X.npy')
    meta = pd.read_csv(DATA + '_meta.csv', encoding='utf-8-sig', dtype={'muniCd': str})
    y = meta.y.values
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 段階27・EXP-036 が未使用(docs に EXP-036 なし): {"EXP-036" not in docs}')
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    ref = pd.read_csv(os.path.join(S13, 'splits.csv'), encoding='utf-8-sig')
    ref = ref[ref.label == KEY].drop(columns='label').reset_index(drop=True)
    mine = split_table(sp, y, meta.muniCd.values)
    ok = len(meta) == N_EXPECT and X.shape[-1] == 60 and mine.shape == ref.shape and \
        np.allclose(mine.values.astype(float), ref.values.astype(float))
    log(f'確認 件数 {len(meta)}(多収 {int(y.sum())} / 低収 {int((1 - y).sum())})・窓 60点・分割が段階13と同じ: {ok}')
    assert ok
    Xd = first_diff(X)
    inputs = {'A': anchored(X), 'P': pos_diff(X), 'N': neg_diff(X), 'PN': pn_channels(X)}
    ok_shape = all(v.shape[-1] == 59 for v in inputs.values()) and inputs['PN'].shape[1] == 2
    ok_def = (np.allclose(inputs['P'] + inputs['N'], Xd) and (inputs['P'] >= 0).all() and (inputs['N'] <= 0).all()
              and np.allclose(np.diff(np.concatenate([np.zeros_like(X[..., :1]), inputs['A']], axis=-1), axis=-1), Xd)
              and np.allclose(inputs['PN'][:, 0], inputs['P'][:, 0]) and np.allclose(inputs['PN'][:, 1], inputs['N'][:, 0]))
    log(f'確認 入力の長さがすべて59点・PN は2チャンネル: {ok_shape}。定義の確認(P + N = Δθ、P ≧ 0、N ≦ 0、'
        f'A の1次差分 = Δθ、PN = [P, N]): {ok_def}')
    log(f'  ゼロの割合: P {np.mean(inputs["P"] == 0):.3f}、N {np.mean(inputs["N"] == 0):.3f}、'
        f'Δθ がちょうど0 {np.mean(Xd == 0):.3f}')
    assert ok_shape and ok_def
    for k, v in inputs.items():
        np.save(os.path.join(NEW_DATA, f'label_muniyear_median_{k}_X.npy'), v)
    meta.to_csv(os.path.join(NEW_DATA, 'label_muniyear_median_meta.csv'), index=False, encoding='utf-8-sig')
    log(f'  新しい入力を保存: {NEW_DATA}(元の stage13 の _X.npy は変更していない)')

    # ---- 起動時の確認
    f13 = pd.read_csv(os.path.join(S13, 'folds.csv'), encoding='utf-8-sig')
    f13 = f13[(f13.label == KEY) & (f13.config == 'M') & (f13.seed == 0)].reset_index(drop=True)
    d10 = s6.eval_cached(s6.fold_features('minirocket', 0, Xd, sp[:10]), y, sp[:10])
    ok_d = np.allclose(d10.auc, f13.auc[:10], rtol=0, atol=1e-12, equal_nan=True) and \
        np.allclose(d10.bacc, f13.bacc[:10], rtol=0, atol=1e-12)
    log(f'確認 D の再計算(乱数0、先頭10 fold)が段階13と一致: {ok_d}')
    assert ok_d, 'D が段階13と一致しない。止めて報告する'
    for name in ['A', 'PN']:
        a = pipeline_eval(name, 0, inputs[name], y, sp[:10])
        b = s6.eval_cached(fold_features(name, 0, inputs[name], sp[:10]), y, sp[:10])
        okp = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
        log(f'確認 Pipeline と特徴量の使い回しの一致({name}、乱数0、先頭10 fold): {okp}')
        assert okp
    # まとめて解いた置換 = sklearn(PN、置換20回)。1チャンネルは段階13の prechecks(P で)
    log('確認(P): 段階13の prechecks')
    s13.prechecks(KEY, inputs['P'], meta, sp, log)
    psq = s13.perm_sets(KEY, meta, 20)
    feats = fold_features('PN', 0, inputs['PN'], sp)
    decs, _ = s12.m_decisions(feats, y, sp, psq[2])
    refd = Parallel(n_jobs=s13.N_JOBS)(delayed(s12._m_fold_sklearn)(Ftr, Fte, psq[2][:, tr])
                                       for (r, f, tr, te), (Ftr, Fte) in zip(sp, feats))
    _, na = s13.analyse(decs, meta, sp, psq)
    _, nb = s13.analyse([np.vstack([d[:1], rr]) for d, rr in zip(decs, refd)], meta, sp, psq)
    okb = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'確認(PN): 置換20回分の帰無分布(全体・W・B)、まとめて解いた結果 = sklearn で1回ずつ: {okb}')
    assert okb
    del feats

    # ---- 入力ごと(途中結果を保存・再開)
    res = {}
    for name in INPUTS:
        part = os.path.join(OUT_DIR, f'partial_{name}.pkl')
        tag = {'seeds': seeds, 'P': P}
        if os.path.exists(part) and not args.fresh:
            with open(part, 'rb') as fh:
                saved = pickle.load(fh)
            if saved['tag'] == tag:
                res[name] = saved['out']
                log(f'\n[{name}: {NAMES[name]}] 途中結果を読み込んだ({part})')
                continue
        log(f'\n[{name}: {NAMES[name]}]')
        res[name] = run_input(name, inputs[name], meta, sp, seeds, P, log)
        with open(part, 'wb') as fh:
            pickle.dump({'tag': tag, 'out': res[name]}, fh)

    d_sm, d_folds, d_pair, S_d, d_ps = load_d(meta, sp, seeds, log)

    # ---- まとめの表
    summary = pd.concat([d_sm] + [pd.DataFrame(res[n]['summary']) for n in INPUTS], ignore_index=True)
    m = summary.metric == 'all_auc'
    pin = summary[m & summary.config.isin(INPUTS)].set_index('config').p
    adj = pd.Series(holm(pin.values), index=pin.index)
    summary.loc[m & summary.config.isin(INPUTS), 'p_holm'] = summary.loc[m & summary.config.isin(INPUTS), 'config'].map(adj)
    has = summary.null_q95.notna()
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(has, None)
    keep = ['config', 'metric', 'n', 'n_seeds', 'n_perm', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p',
            'p_holm', 'above_q95', 'n_pairs_per_repeat', 'alpha_at_max']
    summary = summary.reindex(columns=keep)

    # ---- D との差(fold 平均の差、繰り返し平均の区間、クラスターブートストラップ)
    Cs, N0 = {}, None
    for name, S in [('D', S_d)] + [(n, res[n]['S']) for n in INPUTS]:
        C, N = pair_sums(S, meta, sp)
        if N0 is None:
            N0 = N
        assert np.array_equal(N, N0)
        Cs[name] = C
    boot, pooled = boot_diff(Cs, N0, meta, args.n_boot)
    boot.to_csv(os.path.join(OUT_DIR, 'bootstrap_auc.csv'), index=False, encoding='utf-8-sig')
    rows = []
    for name in INPUTS:
        a = pd.concat(res[name]['folds'])[['seed', 'repeat', 'fold', 'auc']]
        j = d_folds[['seed', 'repeat', 'fold', 'auc']].merge(a, on=['seed', 'repeat', 'fold'], suffixes=('_D', '_X'))
        assert len(j) == len(a)
        diff = j.auc_X - j.auc_D
        rep = j.assign(d=diff).groupby('repeat').d.mean()
        bd = boot[name] - boot['D']
        sign = 'D − ' + name if name == 'A' else name + ' − D'
        s = -1.0 if name == 'A' else 1.0   # A は D − A の向きで判定するので符号を反転
        rows.append({'comparison': sign, 'input': name, 'point_fold_mean': s * diff.mean(),
                     'point_pooled_pairs': s * (pooled[name] - pooled['D']),
                     'boot_q025': np.quantile(s * bd, .025), 'boot_q975': np.quantile(s * bd, .975),
                     'boot_frac_pos': (s * bd > 0).mean(), 'repeat_q025': np.quantile(s * rep, .025),
                     'repeat_q975': np.quantile(s * rep, .975), 'n_boot': args.n_boot})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(os.path.join(OUT_DIR, 'diff_vs_D.csv'), index=False, encoding='utf-8-sig')
    for name in INPUTS:
        r = diffs.set_index('input').loc[name]
        summary.loc[(summary.config == name) & m, 'diff_label'] = r.comparison
        summary.loc[(summary.config == name) & m, 'diff_point'] = r.point_fold_mean
        summary.loc[(summary.config == name) & m, 'diff_boot_q025'] = r.boot_q025
        summary.loc[(summary.config == name) & m, 'diff_boot_q975'] = r.boot_q975
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat([d_folds] + [f for n in INPUTS for f in res[n]['folds']]).to_csv(
        os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.concat([d_ps] + [pd.DataFrame(res[n]['seed']) for n in INPUTS]).to_csv(
        os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat([d_pair] + [p for n in INPUTS for p in res[n]['pair']]).to_csv(
        os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')
    pd.concat([s for n in INPUTS for s in res[n]['scores']]).to_csv(
        os.path.join(OUT_DIR, 'oof_scores.csv'), index=False, encoding='utf-8-sig')
    plot(summary, diffs)

    # ---- 判定
    sm = summary.set_index(['config', 'metric'])
    above = {n: bool(sm.loc[(n, 'all_auc'), 'mean'] > sm.loc[(n, 'all_auc'), 'null_q95']) for n in INPUTS}
    dd = diffs.set_index('input')
    trial = '' if len(seeds) == 10 and P == 200 and args.n_boot == 2000 else '(試運転)'
    log('\n[まとめ]')
    log(summary.round(3).to_string(index=False))
    log('\n[D との差]')
    log(diffs.round(4).to_string(index=False))
    log(f'\n参考: 生の VWC {RAW_REF["auc"]}(p = {RAW_REF["p"]}、{RAW_REF["src"]}。回し直していない)')
    lo, hi = dd.loc['A', 'boot_q025'], dd.loc['A', 'boot_q975']
    if lo > 0:
        q1 = '変化そのものに、水準除去以上の情報がある(D − A の区間が0より上)'
    elif above['A'] and lo <= 0 <= hi:
        q1 = '1次差分の効果の大部分は水準のズレの除去で説明できる(A > 95%点、D − A の区間が0を含む)'
    else:
        q1 = 'どちらの基準も満たさない(A が95%点を上回らず、D − A の区間が0を含む)' if lo <= 0 else '判定外'
    log(f'\n[判定 Q1] A {sm.loc[("A", "all_auc"), "mean"]:.3f}(95%点 {sm.loc[("A", "all_auc"), "null_q95"]:.3f})、'
        f'D − A {dd.loc["A", "point_fold_mean"]:+.3f} [{lo:+.3f}, {hi:+.3f}] → {q1}{trial}')
    if above['N'] and not above['P']:
        q2 = '減る動き(乾燥)に情報がある(段階26の推測を支持)'
    elif above['P'] and not above['N']:
        q2 = '増える動き(雨への応答)に情報がある'
    elif above['P'] and above['N']:
        q2 = 'P・N とも95%点を上回る(どちらか一方とは言えない)'
    else:
        q2 = 'P・N とも95%点を上回らない(どちらに情報があるとも言えない)'
    log(f'[判定 Q2] P {sm.loc[("P", "all_auc"), "mean"]:.3f}(95%点 {sm.loc[("P", "all_auc"), "null_q95"]:.3f})、'
        f'N {sm.loc[("N", "all_auc"), "mean"]:.3f}(95%点 {sm.loc[("N", "all_auc"), "null_q95"]:.3f}) → {q2}{trial}')
    pnlo = dd.loc['PN', 'boot_q025']
    log(f'[判定 Q2] PN − D {dd.loc["PN", "point_fold_mean"]:+.3f} [{pnlo:+.3f}, {dd.loc["PN", "boot_q975"]:+.3f}] → '
        + ('区間が0より上: 正負を分ける形を次の段階の候補にする' if pnlo > 0 else '区間が0を含む(または0より下): 候補にしない')
        + trial)
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
