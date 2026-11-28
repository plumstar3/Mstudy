"""
段階33: 同じ市町村×年の中での多収と低収の直接比較(分類器を使わない、探索)(EXP-042)。設定の記録は docs/decisions.md の D-012。

段階32の統計量は「全部の多収 vs 全部の低収」だった。ここでは同じ市町村×年(同じ気象・同じ雨の日)の中だけで差を計算し、
集団を合わせて検定する。探索なので判定の基準は置かず、試した設定をすべて記録して全 p 値の表(補正前と全検定の Holm 後)を出す。

データ: v2 日平均・24時間そろう日だけ(D-008〜D-010)。label_muniyear_median の119件・28集団のうち、両クラスがある24集団・113件を使う。
  段階31の Dy_raw(開花起点)・rain_days・meta と、暦の日付の窓のために soilmoisture_daily_v2.csv(n_hours = 24、328 を除く)を読む。
置換: 市町村×年の中でラベルを並べ替える、10,000回、default_rng([0, i])(段階32と同じ並べ替え)。集団の重みは各集団1票が主、件数の重みは参考。

検定1(中心): 集団の開花日の中央値(日の途中なら切り下げ)を0日目とした暦の 0〜+59 日。D・N・P(t = 1 … 59)と水準 θ(参考)。
  日ごとに d_g(t) = 多収の平均 − 低収の平均、t = 平均 / (SD / √G)。max-T(両側)とクラスタ質量(|t| ≥ 2.0)。参考: 開花起点で同じ計算。
検定2: 集団の中のペアだけの energy(V 統計量)。E_g の平均。(a) 暦の日付の D・N・P(ユークリッド距離)/
  (b) 段階32と同じ MiniRocket 特徴量(開花起点の D・N・P、乱数0、全件で標準化)。上側の p と z。
検定3: 集団の雨の日 = 集団の件数の半分以上の圃場で rain_days(段階29の定義、暦の日付に置き直したもの。開花窓の外は雨の日に当たらない)。
  集団のイベント(最後の雨の日 e、s ≥ 1、e + 7 ≤ 59)ごとに θ(e) − θ(e + h)、h = min(k, 集団の次の雨の日までの日数 − 1)。
  イベントの値 = 多収の平均 − 低収の平均(正 = 多収のほうが多く減る)、集団の値 = イベントの平均、統計量 = 集団の値の平均(両側)。
  主は k = 3。補助: 減少率(/ 上昇幅、上昇幅 < 0.002 の圃場を除く)、k = 1・2・4・5。記述: 符号検定、集団ごとのイベント数。
開花日の幅で絞った版(記述): 集団の中の開花日の幅が14日以下の集団だけで、検定1(D・N)と検定3を計算し直す。
年ごと(記述): 検定1の D・N、検定2(a)(b)、検定3(k = 3)。

試運転(--part trial、置換200回)はコードの確認だけ。結果は outputs/stage33/trial/ に出し、レポートに載せない。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage33\\run_stage33.py --part trial
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage33\\run_stage33.py --part main
出力: rocket/outputs/stage33/
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
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage32'))
import run_stage32 as s32  # noqa: E402

D31 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage31')
DAILY = os.path.join(ROCKET_DIR, '大豆元データ', 'soilmoisture_daily_v2.csv')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage33')
LABEL = 'label_muniyear_median'
T_LEN = 60
LAGS = 7
K_MAIN = 3
K_ALL = [1, 2, 3, 4, 5]
AMP_MIN = 0.002
CLUSTER_THR = 2.0
SPREAD_MAX = 14
INK, MUTED, GRID = s32.INK, s32.MUTED, s32.GRID
COL, LS, GNAME = s32.COL, s32.LS, s32.GNAME
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データ
def load():
    Xf = np.load(os.path.join(D31, f'{LABEL}_Dy_raw_X.npy'))[:, 0, :]
    meta = pd.read_csv(os.path.join(D31, f'{LABEL}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    meta['flower_date'] = pd.to_datetime(meta.flower_date)
    rain_f = np.load(os.path.join(D31, 'rain_days.npy')).astype(bool)
    d = pd.read_csv(DAILY, encoding='utf-8-sig')
    d = d[(d.n_hours == 24) & (d.field_id != 328)].copy()
    d['date'] = pd.to_datetime(d.date)
    return Xf, meta, rain_f, d


def group_table(meta):
    rows = []
    for g, gg in meta.groupby('muniyear', sort=True):
        med = gg.flower_date.median().floor('D')
        n1 = int(gg[LABEL].sum())
        rows.append({'group': g, 'year': int(gg.year.iloc[0]), 'muniCd': gg.muniCd.iloc[0], 'n': len(gg), 'n_high': n1,
                     'n_low': len(gg) - n1, 'flower_median': med.date(),
                     'flower_spread': int((gg.flower_date.max() - gg.flower_date.min()).days),
                     'used': (n1 > 0) and (n1 < len(gg))})
    t = pd.DataFrame(rows)
    t['in_spread14'] = t.used & (t.flower_spread <= SPREAD_MAX)
    return t


def calendar(meta, gt, d, rain_f):
    """暦の日付の窓: θ(n, 60)、rain_days を置き直した雨の日(n, 60)、開花窓の外の日(n, 60)。"""
    med = dict(zip(gt.group, pd.to_datetime(gt.flower_median)))
    Xc = np.full((len(meta), T_LEN), np.nan)
    Rc = np.zeros((len(meta), T_LEN), bool)
    outside = np.zeros((len(meta), T_LEN), bool)
    for i, r in meta.iterrows():
        dates = pd.date_range(med[r.muniyear], periods=T_LEN)
        s = d[(d.field_id == r.field_id) & (d.year == r.year)].set_index('date').VWC
        Xc[i] = s.reindex(dates).to_numpy()
        off = (dates - r.flower_date).days.to_numpy()
        ok = (off >= 0) & (off < T_LEN)
        Rc[i, ok] = rain_f[i, off[ok]]
        outside[i] = ~ok
    return Xc, Rc, outside


def diff_series(X):
    dl = X[:, 1:] - X[:, :-1]
    return {'D': dl, 'N': np.minimum(dl, 0.0), 'P': np.maximum(dl, 0.0), 'level': X}


# ---------------------------------------------------------------- 検定1
def group_diffs(S, Ys, gidx):
    """集団ごとの差の曲線。S (n, T)(NaN 可)、Ys (B, n) → (G, B, T)。片方のクラスに値が無い日は NaN。"""
    V = (~np.isnan(S)).astype(float)
    S0 = np.nan_to_num(S)
    out = []
    for idx in gidx:
        H = Ys[:, idx].astype(float)
        L = 1.0 - H
        nh, nl = H @ V[idx], L @ V[idx]
        mh = (H @ S0[idx]) / nh
        ml = (L @ S0[idx]) / nl
        dg = mh - ml
        dg[(nh == 0) | (nl == 0)] = np.nan
        out.append(dg)
    return np.stack(out)


def t_across(Dg, w=None):
    """集団をまたいだ t。Dg (G, B, T)。w は集団の重み(None なら各集団1票)。"""
    valid = ~np.isnan(Dg)
    G = valid.sum(0)
    if w is None:
        m = np.nanmean(Dg, 0)
        sd = np.nanstd(Dg, 0, ddof=1)
        return m / (sd / np.sqrt(G)), m, G
    W = np.where(valid, np.asarray(w, float)[:, None, None], 0.0)
    X0 = np.nan_to_num(Dg)
    sw = W.sum(0)
    m = (W * X0).sum(0) / sw
    se = np.sqrt(G / (G - 1)) * np.sqrt((W ** 2 * (X0 - m) ** 2).sum(0)) / sw
    return m / se, m, G


def maxT(Tm, offset):
    """Tm (B, T)。max-T とクラスタ質量の結果。"""
    t_obs, Tn = Tm[0], Tm[1:]
    mx = np.nanmax(np.abs(Tn), axis=1)
    Tobs = np.nanmax(np.abs(t_obs))
    arg = int(np.nanargmax(np.abs(t_obs)))
    cm_null = s32.max_cluster(np.nan_to_num(Tn), CLUSTER_THR)
    cl = s32.clusters_obs(np.nan_to_num(t_obs), CLUSTER_THR, offset)
    mmax = max([m for *_, m in cl], default=0.0)
    padj = np.array([(1 + np.sum(mx >= abs(v) - 1e-12)) / (1 + len(mx)) if not np.isnan(v) else np.nan for v in t_obs])
    return {'T_max': Tobs, 'arg': arg + offset, 't_at_arg': t_obs[arg], 'thr95': np.quantile(mx, 0.95),
            'p_maxT': s32.p_upper(mx, Tobs), 'n_clusters': len(cl), 'max_cluster_mass': mmax,
            'p_cluster': s32.p_upper(cm_null, mmax) if cl else 1.0}, cl, cm_null, mx, padj


# ---------------------------------------------------------------- 検定2
def energy_V(Dm, Ys, gidx, w=None):
    """集団の中の V 統計量の energy の平均。Dm (n, n)、Ys (B, n) → (B,)。"""
    Es = []
    for idx in gidx:
        Dg = Dm[np.ix_(idx, idx)]
        H = Ys[:, idx].astype(float)
        L = 1.0 - H
        n1, n0 = H.sum(1), L.sum(1)
        HD, LD = H @ Dg, L @ Dg
        Es.append(2 * (HD * L).sum(1) / (n1 * n0) - (HD * H).sum(1) / n1 ** 2 - (LD * L).sum(1) / n0 ** 2)
    Es = np.stack(Es)
    if w is None:
        return Es.mean(0), Es[:, 0]
    w = np.asarray(w, float)
    return (w[:, None] * Es).sum(0) / w.sum(), Es[:, 0]


def euclid_within_complete(S, gidx):
    """集団の中で全圃場に値がある日だけを使うユークリッド距離(集団の外は0、使わない)。"""
    n = len(S)
    Dm = np.zeros((n, n))
    for idx in gidx:
        sub = S[idx]
        cols = ~np.isnan(sub).any(0)
        Dm[np.ix_(idx, idx)] = s32.euclid(sub[:, cols])
    return Dm


# ---------------------------------------------------------------- 検定3
def group_events(Rc, gt, gidx_all, meta):
    """集団の雨の日と集団のイベント。"""
    out = {}
    for g, idx in gidx_all.items():
        n = len(idx)
        grain = Rc[idx].sum(0) >= n / 2.0
        out[g] = (grain, s32.events(grain))
    return out


def test3_stat(Xc, Ys, gidx, gnames, gev, k, rate=False):
    """集団の値 (G, B) と、イベントの記録。正 = 多収のほうが多く減る。"""
    vals, rec = [], []
    for g, idx in zip(gnames, gidx):
        grain, evs = gev[g]
        H = Ys[:, idx].astype(float)
        L = 1.0 - H
        ev_vals = []
        for s, e in evs:
            nxt = np.flatnonzero(grain[e + 1:e + LAGS + 1])
            h = k if len(nxt) == 0 else min(k, nxt[0])          # 集団の次の雨の日が lag j なら有効は j−1 まで
            if h == 0:
                rec.append({'group': g, 's': s, 'e': e, 'k': k, 'h': 0, 'used': False})
                continue
            dec = Xc[idx, e] - Xc[idx, e + h]
            v = np.ones(len(idx))
            if rate:
                rise = Xc[idx, e] - Xc[idx, s - 1]
                v = (rise >= AMP_MIN).astype(float)
                dec = np.where(v > 0, dec / np.where(rise == 0, 1, rise), 0.0)
            nh, nl = H @ v, L @ v
            with np.errstate(invalid='ignore', divide='ignore'):
                ev = (H @ (dec * v)) / nh - (L @ (dec * v)) / nl
            ev[(nh == 0) | (nl == 0)] = np.nan
            ev_vals.append(ev)
            rec.append({'group': g, 's': s, 'e': e, 'k': k, 'h': h, 'used': True, 'n_valid': int(v.sum()),
                        'value': ev[0]})
        if ev_vals:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                vals.append(np.nanmean(np.stack(ev_vals), 0))
        else:
            vals.append(np.full(len(Ys), np.nan))
    return np.stack(vals), rec


def combine(V, w=None):
    """集団の値 (G, B) → 統計量 (B,)。"""
    if w is None:
        return np.nanmean(V, 0)
    W = np.where(np.isnan(V), 0.0, np.asarray(w, float)[:, None])
    return (W * np.nan_to_num(V)).sum(0) / W.sum(0)


def p_two(stat):
    return s32.p_upper(np.abs(stat[1:]), abs(stat[0]))


# ---------------------------------------------------------------- 図
def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


def plot_diff_curves(curves, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    names = {'D': 'D = Δθ', 'N': 'N = min(Δθ, 0)', 'P': 'P = max(Δθ, 0)(対照)', 'level': 'θ の水準(参考)'}
    for ax, nm in zip(axes.ravel(), ['D', 'N', 'P', 'level']):
        Dg = curves[('cal', nm)]
        x = np.arange(1, 60) if nm != 'level' else np.arange(0, 60)
        G = np.sum(~np.isnan(Dg), 0)
        m = np.nanmean(Dg, 0)
        h = stats.t.ppf(0.975, np.maximum(G - 1, 1)) * np.nanstd(Dg, 0, ddof=1) / np.sqrt(G)
        ax.fill_between(x, m - h, m + h, color='#2a78d6', alpha=0.15, lw=0)
        ax.plot(x, m, color='#2a78d6', lw=2, label='集団の差の平均(多収 − 低収)')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(names[nm], loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('集団の開花日の中央値からの日数(暦の日付)', color=INK)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, '帯は集団間の95%区間(各集団1票、24集団)', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out_dir, 'fig_diff_curves.png'), dpi=150)
    plt.close(fig)


def plot_t(tc, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, nm in zip(axes.ravel(), ['D', 'N', 'P', 'level']):
        x = np.arange(1, 60) if nm != 'level' else np.arange(0, 60)
        t, thr = tc[('cal', nm)]
        tf, _ = tc[('flw', nm)]
        ax.plot(x, tf, color=MUTED, lw=1.2, ls='--', label='開花起点で揃えた t(参考)')
        ax.plot(x, t, color=INK, lw=2, label='暦の日付で揃えた t')
        for s in (1, -1):
            ax.axhline(s * thr, color='#2a78d6', ls='--', lw=1.2, label='max-T の95%点(暦)' if s == 1 else None)
            ax.axhline(s * CLUSTER_THR, color=MUTED, ls=':', lw=1, label=f'±{CLUSTER_THR}' if s == 1 else None)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title({'D': 'D', 'N': 'N', 'P': 'P(対照)', 'level': '水準(参考)'}[nm], loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('0日目からの日数', color=INK)
        ax.set_ylabel('t(集団の差の平均 / 標準誤差)', color=INK)
        style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_t_curves.png'), dpi=150)
    plt.close(fig)


def plot_panels(Xc, meta, gt, gev, out_dir):
    fig, axes = plt.subplots(7, 4, figsize=(14, 18), sharex=True)
    y = meta[LABEL].astype(int).to_numpy()
    for ax, r in zip(axes.ravel(), gt.itertuples()):
        idx = np.flatnonzero(meta.muniyear.to_numpy() == r.group)
        grain, _ = gev[r.group]
        for t in np.flatnonzero(grain):
            ax.axvline(t, color='#9b59b6', alpha=0.18, lw=2)
        for i in idx:
            ax.plot(np.arange(T_LEN), Xc[i], color=COL[y[i]], ls=LS[y[i]], lw=1.1)
        mark = '' if r.used else '(差の計算から除外)'
        sp = '' if r.flower_spread <= SPREAD_MAX else '、幅>14日'
        ax.set_title(f'{r.group}  多収{r.n_high}/低収{r.n_low}  開花の幅{r.flower_spread}日{sp}{mark}',
                     loc='left', fontsize=7.5, color=INK)
        style(ax)
        ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=COL[1], ls=LS[1]), Line2D([], [], color=COL[0], ls=LS[0]),
                Line2D([], [], color='#9b59b6', alpha=0.4, lw=4)], ['多収', '低収', '集団の雨の日'],
               loc='upper right', frameon=False, fontsize=9, ncol=3)
    fig.supxlabel('集団の開花日の中央値からの日数(暦の日付)', fontsize=10, color=INK)
    fig.supylabel('VWC', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(out_dir, 'fig_group_panels.png'), dpi=130)
    plt.close(fig)


def plot_test3(gvals, gnames, out_dir):
    o = np.argsort(gvals)
    fig, ax = plt.subplots(figsize=(8, 6))
    v = gvals[o]
    ax.scatter(v, np.arange(len(v)), color=[('#2a78d6' if x > 0 else '#eb6834') for x in v], s=36, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(np.nanmean(gvals), color=INK, ls='--', lw=1.2, label=f'平均 {np.nanmean(gvals):+.4f}')
    ax.set_yticks(np.arange(len(v)))
    ax.set_yticklabels([gnames[i] for i in o], fontsize=8)
    ax.set_xlabel('多収の減少量 − 低収の減少量(k = 3、正 = 多収のほうが多く減る)', color=INK)
    ax.set_title('検定3: 集団ごとの値(イベントの平均)', loc='left', fontsize=10.5, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_test3_groups.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本体
def run(P, out_dir, log):
    t_all = time.time()
    Xf, meta, rain_f, daily = load()
    y = meta[LABEL].astype(int).to_numpy()
    groups = meta.muniyear.astype(str).to_numpy()
    gt = group_table(meta)
    gt.to_csv(os.path.join(out_dir, 'groups.csv'), index=False, encoding='utf-8-sig')
    used = gt[gt.used]
    log(f'件数 {len(y)}(多収 {y.sum()} / 低収 {len(y) - y.sum()})、集団 {len(gt)}、置換 {P} 回')
    log(f'使う集団: {len(used)} 集団・{used.n.sum()} 件。除く(片方のクラスだけ): '
        + ', '.join(f'{r.group}({r.n}件)' for r in gt[~gt.used].itertuples()))
    sub = gt[gt.in_spread14]
    log(f'開花日の幅 ≤ {SPREAD_MAX}日の版: {len(sub)} 集団・{sub.n.sum()} 件。除く: '
        + ', '.join(f'{r.group}(幅{r.flower_spread}日、{r.n}件)' for r in gt[gt.used & ~gt.in_spread14].itertuples()))

    Xc, Rc, outside = calendar(meta, gt, daily, rain_f)
    uidx = np.isin(groups, used.group)
    log(f'暦の窓: 欠測 {int(np.isnan(Xc).sum())} 圃場日(119件)。rain_days の範囲外(雨の日に当たらないとする): '
        f'{int(outside[uidx].sum())} / {int(uidx.sum()) * T_LEN} 圃場日(使う24集団)')
    # 確認: 開花窓と重なる日では、置き直した rain_days = 暦の Δθ > 0
    dl = np.diff(Xc, axis=1) > 0
    inside = ~outside[:, 1:] & ~outside[:, :-1]
    agree = (Rc[:, 1:][inside] == dl[inside]).mean()
    log(f'確認: 開花窓の中(前日も中)で、置き直した rain_days と暦の Δθ > 0 の一致率 {agree:.4f}')

    Ys = s32.make_perms(y, groups, P)
    ok_perm = all(np.array_equal(pd.Series(Ys[i]).groupby(groups).sum().values,
                                 pd.Series(y).groupby(groups).sum().values) for i in range(0, P + 1, max(1, P // 50)))
    log(f'置換: 段階32と同じ並べ替え、集団ごとのクラス数が保たれる: {ok_perm}')

    gidx_all = {g: np.flatnonzero(groups == g) for g in gt.group}
    gn_used = list(used.group)
    w_used = used.n.to_numpy()
    gev = group_events(Rc, gt, gidx_all, meta)
    pv = []   # 全 p 値の表

    def add_p(test, variant, weight, subset, year, stat, p, note=''):
        pv.append({'test': test, 'variant': variant, 'weight': weight, 'subset': subset, 'year': year,
                   'stat': stat, 'p': p, 'note': note})

    # ------------------------------------------------ 検定1
    log('\n==== 検定1: 集団の中の差の曲線(max-T・クラスタ質量) ====')
    Sc, Sf = diff_series(Xc), diff_series(Xf)
    curves, tc, summ, pts, cls, contrib = {}, {}, [], [], [], []
    gi_used = [gidx_all[g] for g in gn_used]
    for align, S in [('cal', Sc), ('flw', Sf)]:
        for nm in ['D', 'N', 'P', 'level']:
            off = 0 if nm == 'level' else 1
            Dg = group_diffs(S[nm], Ys, gi_used)
            curves[(align, nm)] = Dg[:, 0, :]
            for wname, w in [('集団1票', None), ('件数', w_used)]:
                Tm, m, G = t_across(Dg, w)
                r, cl, cmn, mx, padj = maxT(Tm, off)
                r.update({'align': align, 'series': nm, 'weight': wname})
                summ.append(r)
                for a, b, sg, ms in cl:
                    cls.append({'align': align, 'series': nm, 'weight': wname, 'start': a, 'end': b, 'sign': sg,
                                'mass': ms, 'p': s32.p_upper(cmn, ms)})
                if wname == '集団1票':
                    tc[(align, nm)] = (Tm[0], r['thr95'])
                    if P >= 1000:
                        np.save(os.path.join(out_dir, f'null_maxT_{align}_{nm}.npy'), mx)
                    for j in range(Tm.shape[1]):
                        pts.append({'align': align, 'series': nm, 'day': j + off, 'G': int(G[0, j]), 'mean_diff': m[0, j],
                                    't': Tm[0, j], 'p_maxT_adj': padj[j]})
                    contrib.append(pd.Series(G[0], index=np.arange(off, off + len(G[0])), name=f'{align}_{nm}'))
                add_p('検定1', f'{"暦" if align == "cal" else "開花起点"}・{nm}・max-T', wname, '24集団', '全年',
                      r['T_max'], r['p_maxT'], f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}')
                add_p('検定1', f'{"暦" if align == "cal" else "開花起点"}・{nm}・クラスタ質量', wname, '24集団', '全年',
                      r['max_cluster_mass'], r['p_cluster'], f'クラスタ {r["n_clusters"]} 個')
                log(f'  {align} {nm} [{wname}]: T_max = {r["T_max"]:.3f}(日 {r["arg"]}、t = {r["t_at_arg"]:+.2f})、'
                    f'95%点 {r["thr95"]:.3f}、p_maxT = {r["p_maxT"]:.4f}、クラスタ {r["n_clusters"]} 個'
                    f'(最大質量 {r["max_cluster_mass"]:.2f}、p = {r["p_cluster"]:.4f})')
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, 'test1_maxT.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(pts).to_csv(os.path.join(out_dir, 'test1_points.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(cls, columns=['align', 'series', 'weight', 'start', 'end', 'sign', 'mass', 'p']).to_csv(
        os.path.join(out_dir, 'clusters.csv'), index=False, encoding='utf-8-sig')
    pd.concat(contrib, axis=1).rename_axis('day').to_csv(os.path.join(out_dir, 'contrib_groups.csv'), encoding='utf-8-sig')
    dc = []
    for (align, nm), C in curves.items():
        off = 0 if nm == 'level' else 1
        d_ = pd.DataFrame(C, columns=[f'd{j + off}' for j in range(C.shape[1])])
        d_.insert(0, 'series', nm)
        d_.insert(0, 'align', align)
        d_.insert(0, 'group', gn_used)
        dc.append(d_)
    pd.concat(dc).to_csv(os.path.join(out_dir, 'diff_curves.csv'), index=False, encoding='utf-8-sig')
    log(f'  寄与集団数(暦・D): 最小 {int(np.min(np.sum(~np.isnan(curves[("cal", "D")]), 0)))}、'
        f'最大 {int(np.max(np.sum(~np.isnan(curves[("cal", "D")]), 0)))}')

    # ------------------------------------------------ 検定2
    log('\n==== 検定2: 集団の中のペアだけの energy(V 統計量) ====')
    t2, Dms = [], {}
    for nm in ['D', 'N', 'P']:
        Dms[('a', nm)] = euclid_within_complete(Sc[nm], gi_used)
        F, nkeep, ntot = s32.minirocket_features(Sf[nm], 0)
        Dms[('b', nm)] = s32.euclid(F)
    for (inp, nm), Dm in Dms.items():
        for wname, w in [('集団1票', None), ('件数', w_used)]:
            E, Eg = energy_V(Dm, Ys, gi_used, w)
            nl = E[1:]
            r = {'input': inp, 'series': nm, 'weight': wname, 'E': E[0], 'null_mean': nl.mean(), 'null_sd': nl.std(ddof=1),
                 'z': (E[0] - nl.mean()) / nl.std(ddof=1), 'p': s32.p_upper(nl, E[0]),
                 'n_groups_Eg_pos': int((Eg > 0).sum())}
            t2.append(r)
            if wname == '集団1票' and P >= 1000:
                np.save(os.path.join(out_dir, f'null_E_{inp}_{nm}.npy'), nl)
            lab = '(a) 暦の系列' if inp == 'a' else '(b) MiniRocket'
            add_p('検定2', f'{lab}・{nm}', wname, '24集団', '全年', E[0], r['p'], f'z = {r["z"]:.2f}')
            log(f'  {lab} {nm} [{wname}]: E = {E[0]:.5f}、z = {r["z"]:.2f}、p = {r["p"]:.4f}、E_g > 0 の集団 {r["n_groups_Eg_pos"]}/24')
    pd.DataFrame(t2).to_csv(os.path.join(out_dir, 'test2_energy.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------------ 検定3
    log('\n==== 検定3: 集団の中で共通の雨のイベントごとの対比較 ====')
    evc = pd.DataFrame([{'group': g, 'n_group_rain_days': int(gev[g][0].sum()), 'n_events': len(gev[g][1])}
                        for g in gn_used])
    log('  集団の雨の日(60日中): ' + f'平均 {evc.n_group_rain_days.mean():.1f}(最小 {evc.n_group_rain_days.min()}、'
        f'最大 {evc.n_group_rain_days.max()})。窓に収まるイベント: 1集団 {evc.n_events.min()}〜{evc.n_events.max()}'
        f'(平均 {evc.n_events.mean():.2f})、イベントの無い集団 {int((evc.n_events == 0).sum())}')
    t3, recs, gv_main = [], [], None

    def run3(gnames, label_subset, year='全年', record=True):
        nonlocal gv_main
        gidx = [gidx_all[g] for g in gnames]
        w = gt.set_index('group').loc[gnames, 'n'].to_numpy()
        res = []
        for k, rate in [(K_MAIN, False), (K_MAIN, True)] + [(k, False) for k in K_ALL if k != K_MAIN]:
            V, rec = test3_stat(Xc, Ys, gidx, gnames, gev, k, rate)
            if record and label_subset == '24集団' and year == '全年':
                for r_ in rec:
                    r_['rate'] = rate
                recs.extend(rec)
            for wname, ww in [('集団1票', None), ('件数', w)]:
                st = combine(V, ww)
                gobs = V[:, 0]
                ok = ~np.isnan(gobs)
                npos = int((gobs[ok] > 0).sum())
                nval = int(ok.sum())
                psign = stats.binomtest(npos, nval, 0.5).pvalue if nval else np.nan
                vname = f'k = {k}' + ('・減少率' if rate else '・減少量')
                r = {'subset': label_subset, 'year': year, 'k': k, 'version': '減少率' if rate else '減少量',
                     'weight': wname, 'stat': st[0], 'null_sd': np.nanstd(st[1:], ddof=1), 'p': p_two(st),
                     'n_groups': nval, 'n_high_more': npos, 'p_sign': psign}
                res.append(r)
                if (label_subset, year, k, rate, wname) == ('24集団', '全年', K_MAIN, False, '集団1票'):
                    gv_main = gobs
                    if P >= 1000:
                        np.save(os.path.join(out_dir, 'null_test3_k3.npy'), st[1:])
                add_p('検定3', vname, wname, label_subset, year, st[0], r['p'],
                      f'多収が多く減る集団 {npos}/{nval}(符号検定 p = {psign:.3f})')
                if wname == '集団1票' or year == '全年':
                    log(f'  [{label_subset}・{year}] {vname} [{wname}]: 統計量 {st[0]:+.5f}、p = {r["p"]:.4f}、'
                        f'多収が多く減る集団 {npos}/{nval}(符号検定 p = {psign:.3f})')
        return res

    t3 += run3(gn_used, '24集団')
    pd.DataFrame(recs).to_csv(os.path.join(out_dir, 'test3_events.csv'), index=False, encoding='utf-8-sig')
    evc['value_k3'] = gv_main
    evc.to_csv(os.path.join(out_dir, 'test3_groups.csv'), index=False, encoding='utf-8-sig')
    hz = pd.DataFrame([r for r in recs if r['k'] == K_MAIN and not r['rate']])
    log(f'  k = 3 のイベント: {len(hz)} 個、h = 0(翌日に集団の雨)で使わない {int((~hz.used).sum())} 個、'
        f'h の内訳 ' + str(hz[hz.used].h.value_counts().sort_index().to_dict()))

    # ------------------------------------------------ 開花日の幅で絞った版
    log(f'\n==== 開花日の幅 ≤ {SPREAD_MAX}日の集団だけ(記述) ====')
    gn_s = list(sub.group)
    gi_s = [gidx_all[g] for g in gn_s]
    w_s = sub.n.to_numpy()
    sub_rows = []
    for nm in ['D', 'N']:
        Dg = group_diffs(Sc[nm], Ys, gi_s)
        for wname, w in [('集団1票', None), ('件数', w_s)]:
            Tm, m, G = t_across(Dg, w)
            r, cl, cmn, mx, padj = maxT(Tm, 1)
            r.update({'align': 'cal', 'series': nm, 'weight': wname, 'subset': f'幅≤{SPREAD_MAX}日'})
            sub_rows.append(r)
            add_p('検定1', f'暦・{nm}・max-T', wname, f'幅≤{SPREAD_MAX}日({len(gn_s)}集団)', '全年', r['T_max'],
                  r['p_maxT'], f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}')
            add_p('検定1', f'暦・{nm}・クラスタ質量', wname, f'幅≤{SPREAD_MAX}日({len(gn_s)}集団)', '全年',
                  r['max_cluster_mass'], r['p_cluster'], f'クラスタ {r["n_clusters"]} 個')
            log(f'  検定1 暦 {nm} [{wname}]: T_max = {r["T_max"]:.3f}(日 {r["arg"]}、t = {r["t_at_arg"]:+.2f})、'
                f'p_maxT = {r["p_maxT"]:.4f}、クラスタ {r["n_clusters"]} 個(p = {r["p_cluster"]:.4f})')
    pd.DataFrame(sub_rows).to_csv(os.path.join(out_dir, 'spread14_test1.csv'), index=False, encoding='utf-8-sig')
    t3 += run3(gn_s, f'幅≤{SPREAD_MAX}日({len(gn_s)}集団)')

    # ------------------------------------------------ 年ごと
    log('\n==== 年ごと(記述) ====')
    yrows = []
    for yr in sorted(used.year.unique()):
        gy = list(used[used.year == yr].group)
        gi_y = [gidx_all[g] for g in gy]
        ny = int(used[used.year == yr].n.sum())
        for nm in ['D', 'N']:
            Dg = group_diffs(Sc[nm], Ys, gi_y)
            Tm, m, G = t_across(Dg)
            if len(gy) >= 3:
                r, *_ = maxT(Tm, 1)
                yrows.append({'test': f'検定1 暦・{nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': r['T_max'],
                              'p': r['p_maxT'], 'note': f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}'})
                add_p('検定1', f'暦・{nm}・max-T', '集団1票', '24集団', yr, r['T_max'], r['p_maxT'])
            else:
                yrows.append({'test': f'検定1 暦・{nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': np.nan,
                              'p': np.nan, 'note': '集団が2つで集団間の t を計算できない'})
        for (inp, nm), Dm in Dms.items():
            E, Eg = energy_V(Dm, Ys, gi_y)
            nl = E[1:]
            z = (E[0] - nl.mean()) / nl.std(ddof=1)
            yrows.append({'test': f'検定2 ({inp}) {nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': E[0],
                          'p': s32.p_upper(nl, E[0]), 'note': f'z = {z:.2f}'})
            add_p('検定2', f'{"(a) 暦の系列" if inp == "a" else "(b) MiniRocket"}・{nm}', '集団1票', '24集団', yr,
                  E[0], s32.p_upper(nl, E[0]), f'z = {z:.2f}')
        V, _ = test3_stat(Xc, Ys, gi_y, gy, gev, K_MAIN, False)
        st = combine(V)
        gobs = V[:, 0]
        ok = ~np.isnan(gobs)
        yrows.append({'test': '検定3 k = 3・減少量', 'year': yr, 'n_groups': int(ok.sum()), 'n': ny, 'stat': st[0],
                      'p': p_two(st), 'note': f'多収が多く減る集団 {int((gobs[ok] > 0).sum())}/{int(ok.sum())}'})
        add_p('検定3', 'k = 3・減少量', '集団1票', '24集団', yr, st[0], p_two(st))
    dfY = pd.DataFrame(yrows)
    dfY.to_csv(os.path.join(out_dir, 'by_year.csv'), index=False, encoding='utf-8-sig')
    for r in dfY.itertuples():
        log(f'  {r.test} {r.year}({r.n_groups}集団・{r.n}件): 統計量 {r.stat:.4f}、p = {r.p:.4f}  {r.note}')

    pd.DataFrame(t3).to_csv(os.path.join(out_dir, 'test3_summary.csv'), index=False, encoding='utf-8-sig')

    # ------------------------------------------------ 全 p 値
    dfP = pd.DataFrame(pv)
    dfP['p_holm_all'] = s32.holm(dfP.p.to_numpy())
    dfP.to_csv(os.path.join(out_dir, 'all_pvalues.csv'), index=False, encoding='utf-8-sig')
    log(f'\n==== 全 p 値: {len(dfP)} 個。補正前 p < 0.05: {int((dfP.p < 0.05).sum())} 個、'
        f'全検定の Holm 後 p < 0.05: {int((dfP.p_holm_all < 0.05).sum())} 個 ====')
    for r in dfP.sort_values('p').head(10).itertuples():
        log(f'  {r.test} {r.variant} [{r.weight}・{r.subset}・{r.year}]: p = {r.p:.4f}、Holm 後 {r.p_holm_all:.4f}  {r.note}')

    plot_diff_curves(curves, out_dir)
    plot_t(tc, out_dir)
    plot_panels(Xc, meta, gt, gev, out_dir)
    plot_test3(gv_main, gn_used, out_dir)
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

    log(f'段階33 --part {a.part}({time.strftime("%Y-%m-%d %H:%M:%S")})')
    if a.part == 'trial':
        log('試運転: コードが動くかの確認だけ(D-012)。この結果で設定を変えない。p 値はレポートに載せない')
    run(P, out_dir, log)
    logf.close()


if __name__ == '__main__':
    main()
