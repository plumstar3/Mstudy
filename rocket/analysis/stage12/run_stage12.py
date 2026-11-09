"""
段階12: 段階11の M の性能を「地域内」と「地域間」に分ける(EXP-021)。

label_year は年の中での順位なので、段階11の AUC には地域内の差と地域間の差が混ざっている。
同じ(乱数, 繰り返し, fold)のテスト件どうしの「多収 × 低収」のペアを3種類に分け、それぞれの AUC を出す。
  W  地域内: 同じ市町村×年
  B  地域間: 同じ年で、違う市町村
  X  年をまたぐ: 違う年(参考)
AUC = スコアの大小がラベルと合うペアの割合(同点は 0.5)。乱数・繰り返しごとに fold をまたいでペアを合算して
AUC を出し、20回の繰り返しで平均し、乱数10通りでも平均する。

データ・モデル・分割は段階11と同じ(data/processed/stage11/ の162件、M・B1・B2、make_splits(y, muniCd, 5, 20, 0))。
置換検定(分割は段階11と同じく観測ラベルで作ったものを固定):
  W  ラベルを市町村×年の中で並べ替える(集団のクラス構成を保つ)
  B  ラベルを年の中で並べ替える(段階11と同じ)
  置換 i の乱数は [0, i]。M は run_stage6_local._batched_ridge で置換をまとめて解き、判定があいまいな行は
  RidgeNTrain で学習し直す(段階6の高速版と同じ)。帰無分布は置換 i ごとに乱数10通りで平均する。
判定(事前に固定): 主 = M の W の AUC が W の帰無分布の95%点を上回るか。副 = M の B と B の帰無分布。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage12\\run_stage12.py
  試運転: ... run_stage12.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage12/(段階11の outputs/stage11/folds.csv が必要)
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
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage11'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage11 as s11  # noqa: E402
from run_stage6 import RidgeNTrain, s5  # noqa: E402
from src.evaluation import make_splits, permute_within  # noqa: E402
from src.windows import first_diff  # noqa: E402

DATA_DIR = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage11')
S11_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage11')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage12')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
SEEDS = list(range(10))
TYPES = ['W', 'B', 'X']
TYPE_LABEL = {'W': 'W 地域内(同じ市町村×年)', 'B': 'B 地域間(同じ年・違う市町村)', 'X': 'X 年をまたぐ(参考)'}
NULL_GROUPS = {'W': 'muniyear', 'B': 'year'}  # 置換で並べ替える単位

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED
TYPE_COLOR = {'W': '#2a78d6', 'B': '#eb6834', 'X': '#9a9890'}


# ---------------------------------------------------------------- データ
def load():
    X = np.load(os.path.join(DATA_DIR, 'year_main_X.npy'))
    m = pd.read_csv(os.path.join(DATA_DIR, 'year_main_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    return X, m


# ---------------------------------------------------------------- ペアの集計
def pair_masks(meta, te):
    """テスト件 i・j の組の種類。(n_te, n_te) の bool を W・B・X ごとに。"""
    my = meta.muniyear.to_numpy(dtype=str)[te]
    yr = meta.year.values[te]
    same_my = my[:, None] == my[None, :]
    same_yr = yr[:, None] == yr[None, :]
    return {'W': same_my, 'B': same_yr & ~same_my, 'X': ~same_yr}


def pair_counts(dec, yte, masks):
    """dec, yte: (P, n_te)。種類ごとに、ラベルと合うペアの数(同点 0.5)と、多収 × 低収のペア数(各 (P,))。"""
    d = np.atleast_2d(dec)
    yy = np.atleast_2d(yte).astype(float)
    cmp = (d[:, :, None] > d[:, None, :]) + 0.5 * (d[:, :, None] == d[:, None, :])  # i のスコア > j
    pn = yy[:, :, None] * (1 - yy[:, None, :])  # i が多収、j が低収
    out = {}
    for t, T in masks.items():
        w = pn * T
        out[t] = ((cmp * w).sum(axis=(1, 2)), w.sum(axis=(1, 2)))
    return out


def repeat_auc(counts, splits, t):
    """fold ごとの (conc, n) を繰り返しごとに合算して AUC にする。戻り値 (n_repeats, P)、ペア数 (n_repeats, P)。"""
    reps = sorted({r for r, *_ in splits})
    conc = {r: 0.0 for r in reps}
    npair = {r: 0.0 for r in reps}
    for (r, f, tr, te), c in zip(splits, counts):
        conc[r] = conc[r] + c[t][0]
        npair[r] = npair[r] + c[t][1]
    C = np.array([conc[r] for r in reps], dtype=float)
    N = np.array([npair[r] for r in reps], dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(N > 0, C / N, np.nan), N


# ---------------------------------------------------------------- 決定関数
def _m_fold(Ftr, Fte, y_tr, Yp_tr):
    """M の1つの fold。行0 = 観測ラベル(sklearn)、行 1.. = 置換(まとめて解き、あいまいな行は学習し直す)。"""
    alpha, dec, unsure = fast._batched_ridge(Ftr, Fte, np.vstack([y_tr, Yp_tr]))
    ref = RidgeNTrain().fit(Ftr, y_tr)
    ref_dec = ref.decision_function(Fte)
    ok = np.isclose(alpha[0], ref.alpha_) and np.allclose(dec[0], ref_dec, rtol=1e-7, atol=1e-10)
    dec[0] = ref_dec
    redo = np.flatnonzero(unsure[1:]) + 1
    for i in redo:
        dec[i] = RidgeNTrain().fit(Ftr, Yp_tr[i - 1]).decision_function(Fte)
    return dec, ok, len(redo)


def _m_fold_sklearn(Ftr, Fte, Yp_tr):
    return np.array([RidgeNTrain().fit(Ftr, yp).decision_function(Fte) for yp in Yp_tr])


def _lr_fold(F, tr, te, Y):
    warnings.filterwarnings('ignore')
    return np.array([clone(s5.lr()).fit(F[tr], yy[tr]).decision_function(F[te]) for yy in Y])


def m_decisions(feats, y, splits, Yp):
    """各 fold の決定関数 (1 + P, n_te)。行0 は観測ラベル。"""
    res = Parallel(n_jobs=N_JOBS)(delayed(_m_fold)(Ftr, Fte, y[tr], Yp[:, tr])
                                  for (r, f, tr, te), (Ftr, Fte) in zip(splits, feats))
    assert all(ok for _, ok, _ in res), '観測ラベルでの alpha・決定関数が sklearn と一致しない fold がある'
    return [d for d, _, _ in res], sum(k for *_, k in res)


def lr_decisions(F, y, splits, Yp):
    Y = np.vstack([y, Yp])
    return Parallel(n_jobs=N_JOBS)(delayed(_lr_fold)(F, tr, te, Y) for r, f, tr, te in splits)


# ---------------------------------------------------------------- 1つの設定(1つの乱数)
def analyse(decs, meta, splits, YpW, YpB, n_perm):
    """観測の繰り返しごとの AUC(種類ごと)、置換の帰無分布(W・B)、fold ごとの観測スコア。"""
    y = meta.y.values
    obs_counts, null_counts = [], {'W': [], 'B': []}
    for (r, f, tr, te), dec in zip(splits, decs):
        masks = pair_masks(meta, te)
        obs_counts.append(pair_counts(dec[0], y[te], masks))
        null_counts['W'].append(pair_counts(dec[1:1 + n_perm], YpW[:, te], {'W': masks['W']}))
        null_counts['B'].append(pair_counts(dec[1 + n_perm:], YpB[:, te], {'B': masks['B']}))
    obs = {}
    for t in TYPES:
        a, n = repeat_auc(obs_counts, splits, t)
        obs[t] = (a[:, 0], n[:, 0])
    null = {}
    for t in ['W', 'B']:
        a, _ = repeat_auc(null_counts[t], splits, t)
        null[t] = np.nanmean(a, axis=0) if n_perm else np.empty(0)  # 置換ごとに20回の繰り返しで平均
    return obs, null, obs_counts


def check_folds(config, seed, decs, meta, splits, s11_folds, obs_counts, log):
    """スコアから計算し直した fold の AUC が段階11と一致し、W・B・X を合わせると fold の AUC になること。"""
    y = meta.y.values
    ref = s11_folds[s11_folds.config == config]
    if seed is not None:
        ref = ref[ref.seed == seed]
    ref = ref.set_index(['repeat', 'fold']).auc
    auc, pooled = [], []
    for (r, f, tr, te), dec, c in zip(splits, decs, obs_counts):
        auc.append(roc_auc_score(y[te], dec[0]) if len(np.unique(y[te])) == 2 else np.nan)
        n = sum(c[t][1][0] for t in TYPES)
        pooled.append(sum(c[t][0][0] for t in TYPES) / n if n else np.nan)
    auc, pooled = np.array(auc), np.array(pooled)
    expect = np.array([ref.loc[(r, f)] for r, f, *_ in splits])
    ok1 = np.allclose(auc, expect, equal_nan=True, rtol=0, atol=1e-12)
    ok2 = np.allclose(auc, pooled, equal_nan=True, rtol=0, atol=1e-12)
    log(f'  確認 {config} 乱数{seed}: fold の AUC が段階11の folds.csv と一致 {ok1}、'
        f'W+B+X をペア数で合わせると fold の AUC と一致 {ok2}(最大差 {np.nanmax(np.abs(auc - expect)):.1e})')
    assert ok1, f'{config} 乱数{seed}: fold の AUC が outputs/stage11/folds.csv と一致しない'
    assert ok2, f'{config} 乱数{seed}: W・B・X を合わせた AUC が fold の AUC と一致しない'


def score_rows(config, seed, decs, meta, splits):
    rows = []
    for (r, f, tr, te), dec in zip(splits, decs):
        m = meta.iloc[te]
        rows.append(pd.DataFrame({'config': config, 'seed': seed, 'repeat': r, 'fold': f,
                                  'field_id': m.field_id.values, 'year': m.year.values, 'muniCd': m.muniCd.values,
                                  'muniyear': m.muniyear.values, 'y': m.y.values, 'score': dec[0]}))
    return rows


def within_group_rows(seed, decs, meta, splits):
    """M の W を、市町村×年の集団ごとに(乱数・繰り返しごと)。"""
    y = meta.y.values
    rows = []
    for (r, f, tr, te), dec in zip(splits, decs):
        my = meta.muniyear.to_numpy(dtype=str)[te]
        for g in np.unique(my):
            k = my == g
            yg = y[te][k]
            if 0 < yg.sum() < len(yg):
                c = pair_counts(dec[0][k], yg, {'W': np.ones((k.sum(), k.sum()), bool)})['W']
                rows.append({'seed': seed, 'repeat': r, 'muniyear': g, 'n': int(k.sum()), 'n_high': int(yg.sum()),
                             'n_pairs': c[1][0], 'auc': c[0][0] / c[1][0]})
    return rows


# ---------------------------------------------------------------- 起動時の確認
def precheck(Xd, meta, splits, log, n_perm=20):
    """置換20回分で、まとめて解いた W・B の帰無分布が、sklearn で1回ずつ学習した結果と一致すること(M、乱数0)。"""
    y = meta.y.values
    YpW = perm_labels(meta, 'W', n_perm)
    YpB = perm_labels(meta, 'B', n_perm)
    Yp = np.vstack([YpW, YpB])
    feats = s6.fold_features('minirocket', 0, Xd, splits)
    t = time.time()
    decs, n_redo = m_decisions(feats, y, splits, Yp)
    ta = time.time() - t
    t = time.time()
    ref = Parallel(n_jobs=N_JOBS)(delayed(_m_fold_sklearn)(Ftr, Fte, Yp[:, tr]) for (r, f, tr, te), (Ftr, Fte)
                                  in zip(splits, feats))
    tb = time.time() - t
    _, na, _ = analyse(decs, meta, splits, YpW, YpB, n_perm)
    decs_ref = [np.vstack([d[:1], rr]) for d, rr in zip(decs, ref)]
    _, nb, _ = analyse(decs_ref, meta, splits, YpW, YpB, n_perm)
    ok = all(np.array_equal(na[t], nb[t], equal_nan=True) for t in ['W', 'B'])
    log(f'確認 置換{n_perm}回分の W・B の帰無分布: まとめて解いた結果 = sklearn で1回ずつ学習した結果: {ok}'
        f'(学習し直した組 {n_redo}、まとめて {ta:.1f}s / 1回ずつ {tb:.1f}s)')
    assert ok


def perm_labels(meta, t, n_perm):
    g = meta[NULL_GROUPS[t]].values
    return np.array([permute_within(meta.y.values, g, np.random.default_rng([0, i])) for i in range(n_perm)]) \
        .reshape(n_perm, len(meta))


# ---------------------------------------------------------------- 図
def plot(pair_auc, summary):
    cfgs = ['M', 'B1', 'B2']
    names = {'M': 'M MiniRocket・1次差分', 'B1': 'B1 VWC の要約統計', 'B2': 'B2 緯度・経度・播種日'}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sm = summary.set_index(['config', 'type'])
    pos, ticks = 0, []
    for c in cfgs:
        for k, t in enumerate(TYPES):
            x = pos + k
            d = pair_auc[(pair_auc.config == c) & (pair_auc.type == t)].groupby('repeat').auc.mean().dropna().values
            bp = ax.boxplot([d], positions=[x], widths=0.6, patch_artist=True, medianprops=dict(color=INK, lw=1.5),
                            boxprops=dict(lw=0), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', ms=3, mfc=MUTED, mec='none'))
            bp['boxes'][0].set_facecolor(TYPE_COLOR[t])
            bp['boxes'][0].set_alpha(0.55)
            ax.plot(x, sm.loc[(c, t), 'auc_mean'], marker='D', ms=5, color=INK, zorder=3)
            if t in ('W', 'B') and np.isfinite(sm.loc[(c, t), 'null_q95']):
                ax.hlines(sm.loc[(c, t), 'null_q95'], x - 0.38, x + 0.38, color=s6.C_RED, lw=2, ls='--')
        ticks.append((pos + 1, names[c]))
        pos += 4
    for p in [3, 7]:
        ax.axvline(p, color=MUTED, lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.set_xticks([p for p, _ in ticks], [n for _, n in ticks], fontsize=10)
    ax.set_ylabel('ペアの AUC(繰り返しごと)', color=INK)
    s6.style(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend([Patch(color=TYPE_COLOR[t], alpha=0.55) for t in TYPES] +
              [Line2D([], [], marker='D', color=INK, ls='', ms=5), Line2D([], [], color=s6.C_RED, lw=2, ls='--')],
              [TYPE_LABEL[t] for t in TYPES] + ['平均(乱数で平均)', '帰無分布の95%点(W: 市町村×年、B: 年の中で並べ替え)'],
              loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('段階12: ペアの種類ごとの AUC(162件、label_year。箱 = 20回の繰り返し、M は乱数で平均)',
                 fontsize=12, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pair_auc.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
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
    X, meta = load()
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    s11_folds = pd.read_csv(os.path.join(S11_DIR, 'folds.csv'))
    log(f'データ: {len(meta)} 件(低収 {int((1 - y).sum())} / 多収 {int(y.sum())})、地域 {meta.muniCd.nunique()}、'
        f'市町村×年 {meta.muniyear.nunique()}')

    # W に寄与する集団と、種類ごとのペア数
    g = meta.groupby('muniyear').y.agg(['size', 'sum'])
    both = g[(g['sum'] > 0) & (g['sum'] < g['size'])]
    log(f'両方のクラスがそろう市町村×年: {len(both)} 集団({int(both["size"].sum())} 件)、'
        f'W のペア {int((both["sum"] * (both["size"] - both["sum"])).sum())} 組(集団は fold をまたがないので、どの繰り返しでも同じ)')
    npairs = pd.DataFrame([{t: c[t][1][0] for t in TYPES} | {'repeat': r} for (r, *_), c in
                           zip(sp, [pair_counts(np.zeros(len(te)), y[te], pair_masks(meta, te)) for *_, te in sp])])
    per_rep = npairs.groupby('repeat')[TYPES].sum()
    log('繰り返しごとのペア数(平均 / 最小 / 最大): ' + '、'.join(
        f'{t} {per_rep[t].mean():.0f} / {per_rep[t].min():.0f} / {per_rep[t].max():.0f}' for t in TYPES))

    precheck(Xd, meta, sp, log)

    YpW, YpB = perm_labels(meta, 'W', P), perm_labels(meta, 'B', P)
    Yp = np.vstack([YpW, YpB])
    pa_rows, score_dfs, wg_rows, rows = [], [], [], []
    feats_bl = s11.baseline_features(X, meta)

    for config in ['M', 'B1', 'B2']:
        t = time.time()
        nulls = {'W': [], 'B': []}
        seed_list = seeds if config == 'M' else [None]
        for seed in seed_list:
            if config == 'M':
                feats = s6.fold_features('minirocket', seed, Xd, sp)
                decs, n_redo = m_decisions(feats, y, sp, Yp)
                del feats
            else:
                decs, n_redo = lr_decisions(feats_bl[config], y, sp, Yp), 0
            obs, null, obs_counts = analyse(decs, meta, sp, YpW, YpB, P)
            check_folds(config, seed, decs, meta, sp, s11_folds, obs_counts, log)
            score_dfs += score_rows(config, seed, decs, meta, sp)
            if config == 'M':
                wg_rows += within_group_rows(seed, decs, meta, sp)
            for tt in TYPES:
                a, n = obs[tt]
                pa_rows.append(pd.DataFrame({'config': config, 'type': tt, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
            for tt in ['W', 'B']:
                nulls[tt].append(null[tt])
            if config == 'M':
                log(f'  M 乱数{seed}: ' + '  '.join(f'{tt} {np.nanmean(obs[tt][0]):.3f}' for tt in TYPES) +
                    f'(学習し直した組 {n_redo})')
        pa = pd.concat([p for p in pa_rows if p.config.iloc[0] == config])
        for tt in TYPES:
            s = pa[pa.type == tt]
            per_seed = s.groupby('seed', dropna=False).auc.mean()  # 20回の繰り返しで平均
            r = {'config': config, 'type': tt, 'auc_mean': per_seed.mean(), 'auc_seed_min': per_seed.min(),
                 'auc_seed_max': per_seed.max(), 'auc_sd_repeat': s.groupby('repeat').auc.mean().std(),
                 'n_pairs_per_repeat': s.groupby('repeat').n_pairs.first().mean(),
                 'n_repeats_with_pairs': int(s.groupby('repeat').n_pairs.first().gt(0).sum()),
                 'n_seeds': len(seed_list), 'n_perm': P if tt in ('W', 'B') else 0}
            if tt in ('W', 'B'):
                nl = np.mean(np.stack(nulls[tt]), axis=0)  # 置換 i ごとに乱数で平均
                np.save(os.path.join(OUT_DIR, f'null_{config}_{tt}.npy'), nl)
                r.update({'null_group': NULL_GROUPS[tt], 'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, 0.95),
                          'p': (1 + np.sum(nl >= r['auc_mean'])) / (1 + P), 'n_null_nan': int(np.isnan(nl).sum())})
                r['above_q95'] = bool(r['auc_mean'] > r['null_q95'])
            rows.append(r)
            log(f'  {config:2s} {tt}: AUC {r["auc_mean"]:.3f}(乱数 {r["auc_seed_min"]:.3f}〜{r["auc_seed_max"]:.3f})'
                + (f'  帰無分布 平均 {r["null_mean"]:.3f} / q95 {r["null_q95"]:.3f}、p={r["p"]:.3f}' if tt != 'X' else '')
                + f'  ペア {r["n_pairs_per_repeat"]:.0f}/繰り返し')
        log(f'  [{config} {time.time() - t:.0f}s]')

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    pair_auc = pd.concat(pa_rows)
    pair_auc.to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')
    pd.concat(score_dfs).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'), index=False, encoding='utf-8-sig')
    wg = pd.DataFrame(wg_rows)
    wg = wg.groupby('muniyear').agg(n=('n', 'first'), n_high=('n_high', 'first'), n_pairs=('n_pairs', 'first'),
                                    auc_mean=('auc', 'mean'), auc_sd=('auc', 'std')).reset_index()
    wg = wg.merge(meta.groupby('muniyear')[['muniCd', 'year']].first().reset_index(), on='muniyear') \
        .sort_values('auc_mean', ascending=False)
    wg.to_csv(os.path.join(OUT_DIR, 'within_groups.csv'), index=False, encoding='utf-8-sig')
    plot(pair_auc, summary)

    log('\n[まとめ]')
    log(summary.round(3).to_string(index=False))
    log('\n[M の W 集団ごと(乱数・繰り返しで平均)]')
    log(wg.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
