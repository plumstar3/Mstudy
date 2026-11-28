"""
段階32: 収量に関連する水分動態の直接検定(EXP-041)。事前登録は docs/decisions.md の D-011。

分類器を学習させずに、系列そのものを使った検定で次の3つの動態を確かめる。
  D       1次差分 Δθ(t)(t = 1 … 59)
  N       減る動き min(Δθ, 0)
  N_post  雨の後の減少(雨後の応答曲線、ラグ 1〜7)
  P       増える動き max(Δθ, 0)(N の対照。主の族に入れない)
データから示唆された仮説(段階27・29・31 の MiniRocket の結果)の、分類器を介さない確認である(double dipping の可能性あり)。

データ(D-008〜D-010): v2 日平均・24時間そろう日だけ・主ラベル label_muniyear_median・開花起点 0〜+59 日の119件。
  data/processed/stage31/ の Dy_raw_X・meta・rain_days(段階29の雨の日の定義 Δθ > 0 を v2 に当てたもの)を読むだけ。
雨のイベント: 連続する雨の日のかたまり(開始 s 〜 最後の雨の日 e)。0日目 = e。s ≥ 1 かつ e + 7 ≤ 59 のものだけ。
応答曲線: c(l) = θ(e + l) − θ(e)、l = 0 … 7。次の雨の日の前日で打ち切り、それより後は欠測。圃場ごとにラグ別の平均。
  l = 0 は定義上0なので検定はラグ 1〜7。振幅で正規化した版: c(l) / a、a = θ(e) − θ(s−1)、a < 0.002 のイベントは除く。
置換(全検定で共通): 市町村×年の中でラベルを並べ替える(src.evaluation.permute_within)。置換 i の乱数は default_rng([0, i])。
  p = (1 + #{帰無 ≥ 観測}) / (1 + P)。
検定A(主: A-D・A-N、対照 A-P): MiniRocket(10000、乱数 k)をラベルなしで全件 fit・変換 → 全件で標準化(SD = 0 の列は除く)
  → ユークリッド距離 → energy 統計量。上側 p。z = (E − 帰無の平均) / 帰無の SD。k = 0 が主、k = 1〜4 は参考。
  記述: 同じ市町村×年の中のペアだけの E_W(集団ごとの E を多収×低収のペア数で重み付け平均。両クラスが2件以上の集団だけ)。
検定B(主: B = 生の応答曲線の max-T): ラグ 1〜7 で Welch の t(多収 − 低収)。T = max|t| の置換分布で p。
  補助: クラスタ質量(|t| ≥ 2.0、隣り合う同符号のラグ、質量 = Σ|t|、帰無は最大質量)。振幅で正規化した版も補助。
検定C(対照): D・N の開花起点の時点(t = 1 … 59)ごとに、B と同じ max-T とクラスタ質量。
検定D(補助): D の DTW 距離(sktime DtwDist、Sakoe–Chiba 半径6日)で PERMANOVA の擬似 F。
多重比較: {A-D(k=0), A-N(k=0), B} に Holm。Holm 後 p < 0.05 を「差がある」とする。ほかはすべて記述。
年ごと(記述): 主の族の3つを 2016・2017・2018(参考)で。同じ置換の、その年の列だけを使う(集団は年の中)。

試運転(--part trial、置換200回)はコードが動くかの確認だけ(D-011)。結果は outputs/stage32/trial/ に出し、レポートに載せない。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage32\\run_stage32.py --part trial
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage32\\run_stage32.py --part main
出力: rocket/outputs/stage32/
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
from scipy import stats

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
from src.evaluation import permute_within  # noqa: E402

D31 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage31')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage32')
LABEL = 'label_muniyear_median'
LAGS = 7
AMP_MIN = 0.002
CLUSTER_THR = 2.0
DTW_WINDOW = 6
N_KERNELS = 10000
SEEDS = [0, 1, 2, 3, 4]
MAIN_SEED = 0
ALPHA = 0.05
INK, MUTED, GRID = '#2b2a28', '#8a8984', '#e4e2dd'
COL = {1: '#2a78d6', 0: '#eb6834'}
LS = {1: '-', 0: '--'}
GNAME = {1: '多収', 0: '低収'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データ
def load():
    X = np.load(os.path.join(D31, f'{LABEL}_Dy_raw_X.npy'))[:, 0, :]        # (n, 60) θ(0 … 59)
    meta = pd.read_csv(os.path.join(D31, f'{LABEL}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    rain = np.load(os.path.join(D31, 'rain_days.npy')).astype(bool)          # (n, 60) 日 0 … 59
    return X, meta, rain


def series(X):
    d = X[:, 1:] - X[:, :-1]                                                # t = 1 … 59
    return {'D': d, 'N': np.minimum(d, 0.0), 'P': np.maximum(d, 0.0)}


def events(rain_i):
    """(s, e) のリスト。窓に完全に収まるもの(s ≥ 1、e + LAGS ≤ 59)だけ。"""
    out = []
    T = len(rain_i)
    for e in range(T):
        if rain_i[e] and (e == T - 1 or not rain_i[e + 1]):
            s = e
            while s - 1 >= 0 and rain_i[s - 1]:
                s -= 1
            if s >= 1 and e + LAGS <= T - 1:
                out.append((s, e))
    return out


def response_curves(X, rain):
    """生の応答曲線と振幅で正規化した版(各 (n, LAGS+1)、欠測は NaN)と、件数の表。"""
    n = len(X)
    raw = np.full((n, LAGS + 1), np.nan)
    nrm = np.full((n, LAGS + 1), np.nan)
    rows = []
    for i in range(n):
        ev = events(rain[i])
        cr, cn = [], []
        for s, e in ev:
            nxt = np.flatnonzero(rain[i, e + 1:e + LAGS + 1])
            stop = LAGS if len(nxt) == 0 else nxt[0]          # 有効なラグは 0 … stop
            c = np.full(LAGS + 1, np.nan)
            c[:stop + 1] = X[i, e:e + stop + 1] - X[i, e]
            cr.append(c)
            a = X[i, e] - X[i, s - 1]
            if a >= AMP_MIN:
                cn.append(c / a)
        cr = np.array(cr) if cr else np.full((0, LAGS + 1), np.nan)
        cn = np.array(cn) if cn else np.full((0, LAGS + 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if len(cr):
                raw[i] = np.nanmean(cr, axis=0)
            if len(cn):
                nrm[i] = np.nanmean(cn, axis=0)
        r = {'i': i, 'n_events': len(cr), 'n_events_norm': len(cn)}
        for l in range(LAGS + 1):
            r[f'n_ev_lag{l}'] = int(np.sum(~np.isnan(cr[:, l]))) if len(cr) else 0
        rows.append(r)
    return raw, nrm, pd.DataFrame(rows)


# ---------------------------------------------------------------- 置換
def make_perms(y, groups, P):
    Ys = np.empty((P + 1, len(y)), dtype=np.int8)
    Ys[0] = y
    for i in range(P):
        Ys[i + 1] = permute_within(y, groups, np.random.default_rng([0, i]))
    return Ys


def p_upper(null, obs):
    return (1 + np.sum(null >= obs - 1e-12)) / (1 + len(null))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, j in enumerate(order):
        run = max(run, (m - r) * ps[j])
        adj[j] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- 検定A
def minirocket_features(Z, seed):
    from sktime.transformations.panel.rocket import MiniRocket
    X3 = Z[:, None, :]
    F = MiniRocket(num_kernels=N_KERNELS, random_state=seed).fit(X3).transform(X3)
    F = np.asarray(F, dtype=float)
    sd = F.std(axis=0)
    keep = sd > 1e-12
    F = (F[:, keep] - F[:, keep].mean(axis=0)) / sd[keep]
    return F, int(keep.sum()), int(len(sd))


def euclid(F):
    g = F @ F.T
    sq = np.diag(g)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * g, 0.0)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def energy(Dm, Ys):
    """各行のラベルでの energy 統計量(U 統計量の形)。Ys (B, n)、1 = 多収。"""
    H = Ys.astype(float)
    L = 1.0 - H
    n1, n0 = H.sum(1), L.sum(1)
    HD, LD = H @ Dm, L @ Dm
    cross = (HD * L).sum(1) / (n1 * n0)
    w1 = (HD * H).sum(1) / (n1 * (n1 - 1))
    w0 = (LD * L).sum(1) / (n0 * (n0 - 1))
    return 2 * cross - w1 - w0


def energy_within(Dm, Ys, groups):
    """同じ市町村×年の中のペアだけの E_W。両クラスが2件以上の集団だけ。重み = 集団内の多収×低収のペア数。"""
    g = np.asarray(groups)
    y0 = Ys[0]
    num = np.zeros(len(Ys))
    den = 0.0
    used = []
    for u in np.unique(g):
        idx = np.flatnonzero(g == u)
        n1 = int(y0[idx].sum())
        n0 = len(idx) - n1
        if n1 < 2 or n0 < 2:
            continue
        Eg = energy(Dm[np.ix_(idx, idx)], Ys[:, idx])
        w = n1 * n0
        num += w * Eg
        den += w
        used.append((u, len(idx)))
    return num / den, used


def test_A(S, Ys, groups, years, log, out_dir):
    rows, nulls, yrows = [], {}, []
    Dms = {}
    for nm in ['D', 'N', 'P']:
        for k in SEEDS:
            t0 = time.time()
            F, nkeep, ntot = minirocket_features(S[nm], k)
            Dm = euclid(F)
            Dms[(nm, k)] = Dm
            E = energy(Dm, Ys)
            Ew, used = energy_within(Dm, Ys, groups)
            null, nullw = E[1:], Ew[1:]
            r = {'series': nm, 'seed': k, 'role': ('主' if nm in ('D', 'N') and k == MAIN_SEED else
                                                    ('対照' if nm == 'P' and k == MAIN_SEED else '参考(乱数)')),
                 'n_features': ntot, 'n_features_used': nkeep,
                 'E': E[0], 'null_mean': null.mean(), 'null_sd': null.std(ddof=1),
                 'z': (E[0] - null.mean()) / null.std(ddof=1), 'p': p_upper(null, E[0]),
                 'E_W': Ew[0], 'z_W': (Ew[0] - nullw.mean()) / nullw.std(ddof=1), 'p_W': p_upper(nullw, Ew[0]),
                 'n_strata_W': len(used), 'n_cases_W': int(sum(c for _, c in used))}
            rows.append(r)
            if k == MAIN_SEED:
                nulls[nm] = null
                np.save(os.path.join(out_dir, f'null_A_{nm}.npy'), null)
                for yr in sorted(np.unique(years)):
                    idx = np.flatnonzero(years == yr)
                    Ey = energy(Dm[np.ix_(idx, idx)], Ys[:, idx])
                    ny = Ey[1:]
                    yrows.append({'test': f'A-{nm}', 'year': yr, 'n': len(idx), 'n_high': int(Ys[0, idx].sum()),
                                  'stat': Ey[0], 'z': (Ey[0] - ny.mean()) / ny.std(ddof=1) if ny.std() > 0 else np.nan,
                                  'p': p_upper(ny, Ey[0]), 'note': 'energy(向きなし)'})
            log(f'  A-{nm} 乱数{k}: 特徴量 {nkeep}/{ntot}、E = {r["E"]:.4f}、z = {r["z"]:.2f}、p = {r["p"]:.4f}、'
                f'E_W z = {r["z_W"]:.2f}、p_W = {r["p_W"]:.4f}({r["n_strata_W"]}集団・{r["n_cases_W"]}件)、'
                f'{time.time() - t0:.1f}秒')
    return pd.DataFrame(rows), nulls, yrows


# ---------------------------------------------------------------- 検定B・C(max-T とクラスタ質量)
def welch_t(C, Ys):
    """C (n, L)(NaN は欠測)、Ys (B, n) → t (B, L)、多収 − 低収。"""
    V = (~np.isnan(C)).astype(float)
    C0 = np.nan_to_num(C)
    H = Ys.astype(float)
    L = 1.0 - H
    out = []
    for G in (H, L):
        n = G @ V
        s = G @ C0
        ss = G @ (C0 ** 2)
        m = s / n
        v = (ss - n * m ** 2) / (n - 1)
        out.append((n, m, v))
    (n1, m1, v1), (n0, m0, v0) = out
    return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


def max_cluster(T, thr):
    """各行の最大クラスタ質量(|t| ≥ thr、隣り合う同符号)。T (B, L)。"""
    B, L = T.shape
    best = np.zeros(B)
    cur = np.zeros(B)
    sgn = np.zeros(B)
    for j in range(L):
        t = T[:, j]
        s = np.where(t >= thr, 1, np.where(t <= -thr, -1, 0))
        cont = (s != 0) & (s == sgn)
        cur = np.where(s == 0, 0.0, np.where(cont, cur + np.abs(t), np.abs(t)))
        sgn = s
        best = np.maximum(best, cur)
    return best


def clusters_obs(t, thr, offset):
    out = []
    j = 0
    while j < len(t):
        if abs(t[j]) >= thr:
            s = np.sign(t[j])
            k = j
            while k + 1 < len(t) and abs(t[k + 1]) >= thr and np.sign(t[k + 1]) == s:
                k += 1
            out.append((j + offset, k + offset, int(s), float(np.abs(t[j:k + 1]).sum())))
            j = k + 1
        else:
            j += 1
    return out


def pointwise(C, Ys, name, offset, unit, out_dir, save_null):
    """max-T とクラスタ質量。C は検定する点だけの (n, L)。offset は先頭の点のラグ/日。"""
    T = welch_t(C, Ys)
    t_obs, Tnull = T[0], T[1:]
    mx = np.nanmax(np.abs(Tnull), axis=1)
    Tobs = np.nanmax(np.abs(t_obs))
    thr95 = np.quantile(mx, 0.95)
    cm_null = max_cluster(np.nan_to_num(Tnull), CLUSTER_THR)
    if save_null:
        np.save(os.path.join(out_dir, f'null_maxT_{name}.npy'), mx)
        np.save(os.path.join(out_dir, f'null_cluster_{name}.npy'), cm_null)
    y0 = Ys[0]
    pts = []
    for j in range(C.shape[1]):
        c = C[:, j]
        h, l = c[(y0 == 1) & ~np.isnan(c)], c[(y0 == 0) & ~np.isnan(c)]
        pts.append({'test': name, unit: j + offset, 'n_high': len(h), 'n_low': len(l),
                    'mean_high': h.mean(), 'mean_low': l.mean(), 'diff': h.mean() - l.mean(),
                    't': t_obs[j], 'p_uncorrected': 2 * stats.t.sf(abs(t_obs[j]), welch_df(h, l)),
                    'p_maxT_adj': (1 + np.sum(mx >= abs(t_obs[j]) - 1e-12)) / (1 + len(mx))})
    cl = clusters_obs(t_obs, CLUSTER_THR, offset)
    crow = [{'test': name, 'start': a, 'end': b, 'sign': s, 'mass': m, 'p': p_upper(cm_null, m)} for a, b, s, m in cl]
    summ = {'test': name, 'n_points': C.shape[1], 'T_max': Tobs, 'arg': int(np.nanargmax(np.abs(t_obs))) + offset,
            't_at_arg': t_obs[int(np.nanargmax(np.abs(t_obs)))], 'thr95': thr95, 'p_maxT': p_upper(mx, Tobs),
            'n_clusters': len(cl), 'max_cluster_mass': max([m for *_, m in cl], default=0.0),
            'p_cluster': p_upper(cm_null, max([m for *_, m in cl], default=0.0)) if cl else 1.0}
    return summ, pd.DataFrame(pts), crow, t_obs, thr95


def welch_df(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


# ---------------------------------------------------------------- 検定D
def permanova_F(D2, Ys):
    H = Ys.astype(float)
    L = 1.0 - H
    N = Ys.shape[1]
    sst = D2.sum() / 2 / N
    ssw = (((H @ D2) * H).sum(1) / 2) / H.sum(1) + (((L @ D2) * L).sum(1) / 2) / L.sum(1)
    return (sst - ssw) / 1 / (ssw / (N - 2))


def test_D(S, Ys, log, out_dir):
    from sktime.dists_kernels.dtw import DtwDist
    t0 = time.time()
    from sktime.dists_kernels._numba_distances.lower_bounding import LowerBounding
    X3 = S['D'][:, None, :]
    # sktime 1.2.0 の内部は window を系列長に対する割合(float)で受け取り、半径 = window × 長さ。
    # 半径6日(D-011)は window = 6 / 59。docstring の int 指定は内部で ValueError になる。
    w = DTW_WINDOW / X3.shape[2]
    fin = np.isfinite(LowerBounding.SAKOE_CHIBA.create_bounding_matrix(X3[0], X3[0], sakoe_chiba_window_radius=w))
    ii, jj = np.nonzero(fin)
    log(f'  Sakoe–Chiba: window = {w:.5f}(割合)→ 帯の半径 max|i − j| = {np.abs(ii - jj).max()} 日')
    Dm = np.asarray(DtwDist(window=w).transform(X3, X3), dtype=float)
    asym = np.abs(Dm - Dm.T).max()
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm, 0.0)
    F = permanova_F(Dm ** 2, Ys)
    np.save(os.path.join(out_dir, 'null_D_permanova.npy'), F[1:])
    np.save(os.path.join(out_dir, 'dtw_D.npy'), Dm)
    r = {'test': 'D (DTW PERMANOVA)', 'window': DTW_WINDOW, 'pseudo_F': F[0], 'null_mean': F[1:].mean(),
         'null_sd': F[1:].std(ddof=1), 'z': (F[0] - F[1:].mean()) / F[1:].std(ddof=1), 'p': p_upper(F[1:], F[0])}
    log(f'  DTW 距離 {time.time() - t0:.1f}秒(非対称の最大 {asym:.2e})。擬似 F = {r["pseudo_F"]:.3f}、z = {r["z"]:.2f}、'
        f'p = {r["p"]:.4f}')
    return pd.DataFrame([r])


# ---------------------------------------------------------------- 図
def plot_curves(raw, nrm, y, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    lags = np.arange(LAGS + 1)
    for ax, C, ttl, yl in [(axes[0], raw, '雨後の応答曲線(生)', 'θ(e + l) − θ(e)'),
                           (axes[1], nrm, '振幅で正規化した版(補助)', '(θ(e + l) − θ(e)) / 上昇幅')]:
        for g in (1, 0):
            Cg = C[y == g]
            n = np.sum(~np.isnan(Cg), axis=0)
            m = np.nanmean(Cg, axis=0)
            se = np.nanstd(Cg, axis=0, ddof=1) / np.sqrt(n)
            h = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * se
            h[0] = 0
            ax.fill_between(lags, m - h, m + h, color=COL[g], alpha=0.15, lw=0)
            ax.plot(lags, m, color=COL[g], ls=LS[g], lw=2, marker='o', ms=4, label=f'{GNAME[g]}(n = {len(Cg)})')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(ttl, loc='left', fontsize=11, color=INK)
        ax.set_xlabel('雨のイベント後の日数 l(0 = 最後の雨の日)', color=INK)
        ax.set_ylabel(yl, color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.text(0.01, 0.01, '帯は圃場単位の平均の95%区間。l = 0 は定義上0。次の雨の日の前日で打ち切り(l = 5〜7 は件数が減る)',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out_dir, 'fig_response_curves.png'), dpi=150)
    plt.close(fig)


def plot_t(tcurves, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, (name, (x, t, thr95, xl)) in zip(axes.ravel(), tcurves.items()):
        ax.plot(x, t, color=INK, lw=2, marker='o' if len(x) < 15 else None, ms=4, label='Welch の t(多収 − 低収)')
        for s in (1, -1):
            ax.axhline(s * thr95, color='#2a78d6', ls='--', lw=1.2, label='max-T の95%点' if s == 1 else None)
            ax.axhline(s * CLUSTER_THR, color=MUTED, ls=':', lw=1, label=f'クラスタのしきい値 ±{CLUSTER_THR}' if s == 1 else None)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(name, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel(xl, color=INK)
        ax.set_ylabel('t', color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_t_curves.png'), dpi=150)
    plt.close(fig)


def plot_energy(dfA, nulls, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, nm in zip(axes, ['D', 'N', 'P']):
        r = dfA[(dfA.series == nm) & (dfA.seed == MAIN_SEED)].iloc[0]
        ax.hist(nulls[nm], bins=50, color=MUTED, alpha=0.6, label='帰無分布(層内置換)')
        ax.axvline(r.E, color='#2a78d6', lw=2, label='観測値')
        role = '主' if nm != 'P' else '対照'
        ax.set_title(f'A-{nm}({role}) z = {r.z:.2f}、p = {r.p:.4f}', loc='left', fontsize=10, color=INK)
        ax.set_xlabel('energy 統計量 E', color=INK)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_energy_null.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本体
def run(P, out_dir, log):
    t_all = time.time()
    X, meta, rain = load()
    y = meta[LABEL].astype(int).to_numpy()
    groups = meta.muniyear.astype(str).to_numpy()
    years = meta.year.to_numpy()
    S = series(X)
    log(f'件数 {len(y)}(多収 {y.sum()} / 低収 {len(y) - y.sum()})、集団 {len(np.unique(groups))}、置換 {P} 回')
    ok = np.array_equal(rain[:, 1:], S['D'] > 0)
    log(f'確認: rain_days(日 1 … 59)= (Δθ > 0): {ok}')
    if not ok:
        raise RuntimeError('雨の日のフラグが Δθ > 0 と一致しない')

    raw, nrm, cnt = response_curves(X, rain)
    cnt.insert(0, 'field_id', meta.field_id.values)
    cnt.insert(1, 'year', years)
    cnt.insert(2, 'label', y)
    cnt.to_csv(os.path.join(out_dir, 'event_counts.csv'), index=False, encoding='utf-8-sig')
    rc = []
    for kind, C in [('raw', raw), ('norm', nrm)]:
        d = pd.DataFrame(C, columns=[f'lag{l}' for l in range(LAGS + 1)])
        d.insert(0, 'kind', kind)
        d.insert(0, 'label', y)
        d.insert(0, 'year', years)
        d.insert(0, 'field_id', meta.field_id.values)
        rc.append(d)
    pd.concat(rc).to_csv(os.path.join(out_dir, 'response_curves.csv'), index=False, encoding='utf-8-sig')
    nofield = int((cnt.n_events == 0).sum())
    log(f'雨のイベント(窓に収まるもの): 1圃場 {cnt.n_events.min()}〜{cnt.n_events.max()} 回(平均 {cnt.n_events.mean():.2f})、'
        f'イベントの無い圃場 {nofield}(除外)。正規化版(上昇幅 ≥ {AMP_MIN}): 合計 {cnt.n_events_norm.sum()} / '
        f'{cnt.n_events.sum()} イベント、使えるイベントの無い圃場 {int((cnt.n_events_norm == 0).sum())}')
    log('ラグごとに値のある圃場(生): ' + ', '.join(f'l{l} {int(np.sum(~np.isnan(raw[:, l])))}' for l in range(LAGS + 1)))
    log('ラグごとのイベント数(生): ' + ', '.join(f'l{l} {int(cnt[f"n_ev_lag{l}"].sum())}' for l in range(LAGS + 1)))
    log('ラグごとに値のある圃場(正規化): ' + ', '.join(f'l{l} {int(np.sum(~np.isnan(nrm[:, l])))}' for l in range(LAGS + 1)))

    t0 = time.time()
    Ys = make_perms(y, groups, P)
    log(f'置換の作成 {time.time() - t0:.1f}秒。各置換で集団ごとのクラス数が保たれる: '
        f'{all(np.array_equal(pd.Series(Ys[i]).groupby(groups).sum().values, pd.Series(y).groupby(groups).sum().values) for i in range(0, P + 1, max(1, P // 50)))}')

    log('\n==== 検定A: MiniRocket 特徴の energy 検定 ====')
    dfA, nullsA, yrows = test_A(S, Ys, groups, years, log, out_dir)
    dfA.to_csv(os.path.join(out_dir, 'testA_energy.csv'), index=False, encoding='utf-8-sig')

    log('\n==== 検定B・C: max-T とクラスタ質量 ====')
    summ, pts_all, cl_all, tcurves = [], [], [], {}
    specs = [('B_raw', raw[:, 1:], 1, 'lag', '検定B(主): 雨後の応答曲線(生)', '雨のイベント後の日数 l'),
             ('B_norm', nrm[:, 1:], 1, 'lag', '検定B(補助): 振幅で正規化', '雨のイベント後の日数 l'),
             ('C_D', S['D'], 1, 'day', '検定C(対照): D(開花起点)', '開花後の日数 t'),
             ('C_N', S['N'], 1, 'day', '検定C(対照): N(開花起点)', '開花後の日数 t')]
    for name, C, off, unit, ttl, xl in specs:
        s, pts, cl, t, thr95 = pointwise(C, Ys, name, off, unit, out_dir, True)
        summ.append(s)
        pts_all.append(pts)
        cl_all += cl
        tcurves[ttl] = (np.arange(off, off + C.shape[1]), t, thr95, xl)
        log(f'  {name}: T_max = {s["T_max"]:.3f}({unit} {s["arg"]}、t = {s["t_at_arg"]:+.3f})、95%点 {thr95:.3f}、'
            f'p_maxT = {s["p_maxT"]:.4f}、クラスタ {s["n_clusters"]} 個(最大質量 {s["max_cluster_mass"]:.2f}、p = {s["p_cluster"]:.4f})')
        if name.startswith('B'):
            log('    t: ' + ', '.join(f'l{j + off} {v:+.2f}' for j, v in enumerate(t)))
    dfS = pd.DataFrame(summ)
    dfS.to_csv(os.path.join(out_dir, 'maxT_summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat([p for p in pts_all if p.test.iloc[0].startswith('B')]).to_csv(
        os.path.join(out_dir, 'testB_points.csv'), index=False, encoding='utf-8-sig')
    pd.concat([p for p in pts_all if p.test.iloc[0].startswith('C')]).to_csv(
        os.path.join(out_dir, 'testC_points.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(cl_all, columns=['test', 'start', 'end', 'sign', 'mass', 'p']).to_csv(
        os.path.join(out_dir, 'clusters.csv'), index=False, encoding='utf-8-sig')

    log('\n==== 検定D: DTW の PERMANOVA ====')
    dfD = test_D(S, Ys, log, out_dir)
    dfD.to_csv(os.path.join(out_dir, 'testD_permanova.csv'), index=False, encoding='utf-8-sig')

    log('\n==== 主の族の Holm 補正 ====')
    fam = [('A-D', dfA[(dfA.series == 'D') & (dfA.seed == MAIN_SEED)].p.iloc[0]),
           ('A-N', dfA[(dfA.series == 'N') & (dfA.seed == MAIN_SEED)].p.iloc[0]),
           ('B', dfS[dfS.test == 'B_raw'].p_maxT.iloc[0])]
    adj = holm([p for _, p in fam])
    dfH = pd.DataFrame({'test': [f for f, _ in fam], 'p': [p for _, p in fam], 'p_holm': adj,
                        'decision': ['差がある' if a < ALPHA else '差を検出できない' for a in adj]})
    dfH.to_csv(os.path.join(out_dir, 'holm.csv'), index=False, encoding='utf-8-sig')
    for r in dfH.itertuples():
        log(f'  {r.test}: p = {r.p:.4f} → Holm 後 {r.p_holm:.4f}({r.decision})')

    log('\n==== 年ごと(記述) ====')
    for yr in sorted(np.unique(years)):
        idx = np.flatnonzero(years == yr)
        s, pts, _, t, thr95 = pointwise(raw[idx, 1:], Ys[:, idx], f'B_raw_{yr}', 1, 'lag', out_dir, False)
        yrows.append({'test': 'B', 'year': yr, 'n': len(idx), 'n_high': int(y[idx].sum()), 'stat': s['T_max'],
                      'z': np.nan, 'p': s['p_maxT'],
                      'note': 't(l1…l7) = ' + ' '.join(f'{v:+.2f}' for v in t)})
    dfY = pd.DataFrame(yrows)
    dfY.to_csv(os.path.join(out_dir, 'by_year.csv'), index=False, encoding='utf-8-sig')
    for r in dfY.itertuples():
        log(f'  {r.test} {r.year}(n = {r.n}、多収 {r.n_high}): 統計量 {r.stat:.4f}、z = {r.z:.2f}、p = {r.p:.4f}  {r.note}')

    plot_curves(raw, nrm, y, out_dir)
    plot_t(tcurves, out_dir)
    plot_energy(dfA, nullsA, out_dir)
    log(f'\n合計 {time.time() - t_all:.1f}秒')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['trial', 'main'], required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    P = 200 if a.part == 'trial' else 10000
    out_dir = os.path.join(OUT_MAIN, 'trial') if a.part == 'trial' else OUT_MAIN
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'段階32 --part {a.part}({time.strftime("%Y-%m-%d %H:%M:%S")})')
    if a.part == 'trial':
        log('試運転: コードが動くかの確認だけ(D-011)。この結果でパラメータや検定の族を変えない。p 値はレポートに載せない')
    run(P, out_dir, log)
    logf.close()


if __name__ == '__main__':
    main()
