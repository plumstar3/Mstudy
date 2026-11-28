"""
段階36: 選択込みの置換検定(Westfall–Young の single-step max-T)で水分動態と収量の関連を直接検定する(EXP-045、D-016)。

分類器は使わない。雨のイベントに揃えた圃場ごとのスカラー特徴量(22個 + 対照 P)と、
  (Y1) 相対収量 y_rel = log(収量 / 市町村×年の平均)(123件・27集団、主の判定)
  (Y2) label_muniyear_median(120件、比較用)
の関連を、市町村×年の層の中だけで比べる統計量で検定する。
  S1(Y1): 層ごとの Spearman ρ を件数で重み付けして平均(3件以上の層だけ)
  S2(Y1): 層ごとの正規化した順位 u = (rank − 0.5)/n を特徴量と y_rel の両方に作り、全件で Pearson 相関
  S3(Y2): 層内 AUC(同じ層の多収・低収のペアだけ)
層内置換10,000回(置換 i は default_rng([0, i]))で圃場と (y_rel, ラベル) の対応を同時に並べ替え、
各組を帰無分布の平均・標準偏差で z にし、族(Y1 44組 / Y2 22組)ごとに置換の各回の最大 |z| から選択込みの p を出す。
対照 P(S1・S2・S3)は族の外で、補正前の p だけ出す。設定はすべて D-016(docs/decisions.md)で事前に固定。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage36\\run_stage36.py --part trial   (置換100回、コードの確認だけ)
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage36\\run_stage36.py --part main    (置換10,000回)
出力: rocket/outputs/stage36/(試運転は outputs/stage36/trial/)、入力は rocket/data/processed/stage36/
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
from src.evaluation import permute_within  # noqa: E402

TS = os.path.join(ROCKET_DIR, 'data', 'processed', 'soilmoisture_timeseries_fixed.csv')
LAB = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_labeled.csv')
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage36')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage36')
LABEL = 'label_muniyear_median'
T_LEN = 60
RAIN_DEFS = [('θ0', 0.0), ('θ2', 0.002), ('θ5', 0.005)]   # θ0 は Δθ > 0、ほかは Δθ ≥ しきい値
K_POST = [1, 2, 3, 5]
K_RATE = 3
K_DRY = 3
AMP_MIN = s32.AMP_MIN          # 0.002
MIN_S1 = 3                     # S1 は3件以上の層だけ
N_BOOT = 2000
INK, MUTED, GRID = s32.INK, s32.MUTED, s32.GRID
COL_Y1, COL_Y2, COL_P = '#2a78d6', '#1f9e74', '#eb6834'
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データ
def build_inputs(log):
    ts = pd.read_csv(TS, encoding='utf-8-sig')
    ts = ts[ts.field_id != 515].copy()          # 使わない圃場(D-013)
    ts['date'] = pd.to_datetime(ts.date)
    lab = pd.read_csv(LAB, encoding='utf-8-sig', dtype={'muniCd': str})
    for c in ['flower_date', 'harvest_date']:
        lab[c] = pd.to_datetime(lab[c])
    lab['muniyear'] = lab.muniCd + '_' + lab.year.astype(str)
    lab['group_mean_yield'] = lab.groupby('muniyear')['yield'].transform('mean')   # 集団の全件(窓の外も含む)
    lab['y_rel'] = np.log(lab['yield'] / lab['group_mean_yield'])
    groups = set(lab.loc[lab[LABEL].notna(), 'muniyear'])
    cand = lab[lab.muniyear.isin(groups)].copy()
    tsh = ts.merge(lab[['field_id', 'year', 'harvest_date']], on=['field_id', 'year'])
    tsh = tsh[tsh.date <= tsh.harvest_date]
    series = {k: g.set_index('date').VWC for k, g in tsh.groupby(['field_id', 'year'])}
    TH, keep, excl = [], [], []
    for i, r in cand.iterrows():
        s = series.get((r.field_id, r.year))
        if s is None:
            excl.append((r.field_id, r.year, r.muniyear, r[LABEL], '系列なし'))
            continue
        w = s.reindex(pd.date_range(r.flower_date, periods=T_LEN))
        if w.isna().any():
            excl.append((r.field_id, r.year, r.muniyear, r[LABEL], f'窓の中に観測の無い日 {int(w.isna().sum())}'))
            continue
        prev = s.get(r.flower_date - pd.Timedelta(days=1), np.nan)
        TH.append(np.concatenate([[prev], w.to_numpy(float)]))      # θ(−1), θ(0) … θ(59)
        keep.append(i)
    meta = cand.loc[keep].reset_index(drop=True)
    TH = np.array(TH)
    log(f'集団(`{LABEL}` が付く市町村×年) {len(groups)} 個、候補 {len(cand)} 件 → 窓の60日がそろう {len(meta)} 件・'
        f'{meta.muniyear.nunique()} 集団(ラベル付き {int(meta[LABEL].notna().sum())}: 多収 {int((meta[LABEL] == 1).sum())} / '
        f'低収 {int((meta[LABEL] == 0).sum())}、中央の空欄 {int(meta[LABEL].isna().sum())})、外した {len(excl)} 件')
    log(f'  年ごと {meta.year.value_counts().sort_index().to_dict()}、集団の大きさ '
        f'{meta.muniyear.value_counts().value_counts().sort_index().to_dict()}、開花日の前日の観測が無い件 '
        f'{int(np.isnan(TH[:, 0]).sum())}')
    os.makedirs(NEW_DATA, exist_ok=True)
    np.save(os.path.join(NEW_DATA, 'theta.npy'), TH)
    meta.to_csv(os.path.join(NEW_DATA, 'meta.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(excl, columns=['field_id', 'year', 'muniyear', LABEL, 'reason']).to_csv(
        os.path.join(NEW_DATA, 'excluded.csv'), index=False, encoding='utf-8-sig')
    return TH, meta


# ---------------------------------------------------------------- 特徴量
def runs(rain):
    """雨の日の区間 (s, e) のリスト。"""
    out, t, T = [], 0, len(rain)
    while t < T:
        if rain[t]:
            s = t
            while t + 1 < T and rain[t + 1]:
                t += 1
            out.append((s, t))
        t += 1
    return out


def features(TH):
    """特徴量の表 (n, 23) と名前・属性の表、イベント数などの記録。"""
    th = TH[:, 1:]                                   # θ(0) … θ(59)
    dfull = TH[:, 1:] - TH[:, :-1]                   # Δθ(0) … Δθ(59)。Δθ(0) は前日が無ければ NaN
    d = dfull[:, 1:]                                 # t = 1 … 59
    n = len(TH)
    cols, spec, info = [], [], []
    for rn, thr in RAIN_DEFS:
        dz = np.nan_to_num(dfull, nan=0.0)
        rain = dz > 0 if thr == 0 else dz >= thr     # t = 0 … 59
        post = {k: np.full(n, np.nan) for k in K_POST}
        rate = np.full(n, np.nan)
        dry = np.full(n, np.nan)
        nev = np.zeros(n)
        n_used_rate = np.zeros(n)
        for i in range(n):
            ev = runs(rain[i])
            nev[i] = len(ev)
            vals = {k: [] for k in K_POST}
            rv = []
            for s, e in ev:
                if e >= T_LEN - 1:
                    continue
                nxt = np.flatnonzero(rain[i, e + 1:])
                stop = T_LEN - 1 if len(nxt) == 0 else e + nxt[0]          # 最後に使える日(次の雨の前日)
                for k in K_POST:
                    last = min(e + k, stop)
                    vals[k].append(np.minimum(dfull[i, e + 1:last + 1], 0.0).mean())
                th_before = TH[i, 0] if s == 0 else th[i, s - 1]
                rise = th[i, e] - th_before
                if np.isfinite(rise) and rise >= AMP_MIN:
                    last = min(e + K_RATE, stop)
                    rv.append(np.minimum(dfull[i, e + 1:last + 1], 0.0).mean() / rise)
            for k in K_POST:
                if vals[k]:
                    post[k][i] = np.mean(vals[k])
            if rv:
                rate[i] = np.mean(rv)
                n_used_rate[i] = len(rv)
            near = np.zeros(T_LEN, bool)              # 雨の日の後 K_DRY 日以内
            for t in np.flatnonzero(rain[i]):
                near[t + 1:t + K_DRY + 1] = True
            ok = ~rain[i, 1:] & ~near[1:]
            if ok.any():
                dry[i] = np.minimum(d[i, ok], 0.0).mean()
        for k in K_POST:
            cols.append(post[k])
            spec.append({'feature': f'N_post_k{k}', 'family_feat': 'F1 N_post', 'rain': rn, 'k': k})
        cols.append(rate)
        spec.append({'feature': f'rate_k{K_RATE}', 'family_feat': 'F2 減少率', 'rain': rn, 'k': K_RATE})
        cols.append(dry)
        spec.append({'feature': 'N_dry', 'family_feat': 'F3 N_dry', 'rain': rn, 'k': K_DRY})
        cols.append(nev)
        spec.append({'feature': 'n_events', 'family_feat': 'F5 イベント数', 'rain': rn, 'k': np.nan})
        info.append({'rain': rn, 'rain_days_mean': rain.sum(1).mean(), 'events_mean': nev.mean(), 'events_min': nev.min(),
                     'events_max': nev.max(), 'rate_events_used_mean': n_used_rate.mean(),
                     'rate_nan': int(np.isnan(rate).sum()), 'dry_nan': int(np.isnan(dry).sum()),
                     'post_nan': int(np.isnan(post[1]).sum())})
    cols.append(np.minimum(d, 0.0).mean(1))
    spec.append({'feature': 'N', 'family_feat': 'F4 N', 'rain': '—', 'k': np.nan})
    cols.append(np.maximum(d, 0.0).mean(1))
    spec.append({'feature': 'P', 'family_feat': 'F6 P(対照)', 'rain': '—', 'k': np.nan})
    F = np.column_stack(cols)
    spec = pd.DataFrame(spec)
    spec['fname'] = [f'{r.feature}|{r.rain}' if r.rain != '—' else r.feature for r in spec.itertuples()]
    return F, spec, pd.DataFrame(info)


# ---------------------------------------------------------------- 統計量(置換をまとめて計算)
def urank(A):
    """行ごとの正規化した順位 (rank − 0.5)/m(同点は平均順位)。A: (R, m)。"""
    m = A.shape[1]
    return (stats.rankdata(A, axis=1) - 0.5) / m


def pearson_rows(x, Y):
    """固定の x (m,) と各行 Y (R, m) の Pearson 相関。"""
    xc = x - x.mean()
    Yc = Y - Y.mean(1, keepdims=True)
    den = np.sqrt((xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(1))
    return (Yc @ xc) / den


def stat_S1(x, Yrel, strata):
    """S1: 層ごとの Spearman ρ の件数重み付き平均(3件以上・特徴量が全部同点でない層)。Yrel: (R, n)。"""
    num = np.zeros(len(Yrel))
    wsum = 0
    for g in np.unique(strata):
        idx = np.flatnonzero((strata == g) & np.isfinite(x))
        if len(idx) < MIN_S1 or np.ptp(x[idx]) == 0:
            continue
        rx = stats.rankdata(x[idx])
        ry = stats.rankdata(Yrel[:, idx], axis=1)
        num += len(idx) * pearson_rows(rx, ry)
        wsum += len(idx)
    return num / wsum


def stat_S2(x, Yrel, strata):
    """S2: 層ごとの正規化した順位 u を特徴量と y_rel の両方に作り、全件で Pearson 相関。"""
    ok = np.isfinite(x)
    ux_parts, uy_parts = [], []
    for g in np.unique(strata):
        idx = np.flatnonzero((strata == g) & ok)
        if len(idx) == 0:
            continue
        ux_parts.append((stats.rankdata(x[idx]) - 0.5) / len(idx))
        uy_parts.append(urank(Yrel[:, idx]))
    return pearson_rows(np.concatenate(ux_parts), np.concatenate(uy_parts, axis=1))


def stat_S3(x, Lab, strata):
    """S3: 層内 AUC。Lab: (R, n)、1 = 多収、0 = 低収、NaN = 中央の空欄。"""
    num = np.zeros(len(Lab))
    den = np.zeros(len(Lab))
    for g in np.unique(strata):
        idx = np.flatnonzero((strata == g) & np.isfinite(x))
        if len(idx) < 2:
            continue
        xv = x[idx]
        M = (xv[:, None] > xv[None, :]).astype(float) + 0.5 * (xv[:, None] == xv[None, :])
        L = Lab[:, idx]
        H = (L == 1).astype(float)
        Lo = (L == 0).astype(float)
        num += ((H @ M) * Lo).sum(1)
        den += H.sum(1) * Lo.sum(1)
    return num / den


def compute_all(F, spec, Yrel, Lab, strata):
    """全組の統計量。戻り値 combos(表)と T (R, 組数)。"""
    rows, T = [], []
    for j, r in spec.iterrows():
        x = F[:, j]
        ctrl = r.feature == 'P'
        for st, fn, Y in [('S1', stat_S1, Yrel), ('S2', stat_S2, Yrel), ('S3', stat_S3, Lab)]:
            fam = ('対照' if ctrl else ('Y1' if st != 'S3' else 'Y2'))
            rows.append({**r.to_dict(), 'stat': st, 'family': fam, 'n_valid': int(np.isfinite(x).sum())})
            T.append(fn(x, Y, strata))
    return pd.DataFrame(rows), np.column_stack(T)


# ---------------------------------------------------------------- 検定
def test(combos, T):
    """z、補正前の両側 p、Holm、選択込みの p(族ごとの max-T)。T の行0 = 実データ。"""
    obs, null = T[0], T[1:]
    B = len(null)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    Z = (T - mu) / sd
    z_obs, z_null = Z[0], np.abs(Z[1:])
    c = combos.copy()
    c['value'] = obs
    c['null_mean'] = mu
    c['null_sd'] = sd
    c['z'] = z_obs
    c['p_raw'] = ((z_null >= np.abs(z_obs)[None, :] - 1e-12).sum(0) + 1) / (B + 1)
    c['p_holm'] = np.nan
    c['p_sel'] = np.nan
    maxes = {}
    for fam in ['Y1', 'Y2']:
        m = (c.family == fam).to_numpy()
        mx = z_null[:, m].max(1)
        maxes[fam] = mx
        c.loc[m, 'p_sel'] = ((mx[:, None] >= np.abs(z_obs[m])[None, :] - 1e-12).sum(0) + 1) / (B + 1)
        c.loc[m, 'p_holm'] = s32.holm(c.loc[m, 'p_raw'].to_numpy())
    return c, maxes


# ---------------------------------------------------------------- 記述
def stat_by_name(st):
    return {'S1': stat_S1, 'S2': stat_S2, 'S3': stat_S3}[st]


def by_year(F, spec, best, meta, yrel, lab):
    rows = []
    for _, b in best.iterrows():
        x = F[:, spec.index[spec.fname == b.fname][0]]
        for yr in [2016, 2017, 2018]:
            m = (meta.year == yr).to_numpy()
            st = meta.muniyear.to_numpy()[m]
            n_lab = int(np.isfinite(lab[m]).sum())
            r = {'family': b.family, 'combo': f'{b.fname}・{b.stat}', 'year': yr, 'n_cases': int(m.sum()),
                 'n_strata': len(np.unique(st)), 'n_labeled': n_lab,
                 'n_strata_S1': int(sum((st == g).sum() >= MIN_S1 for g in np.unique(st)))}
            r['S1_rho'] = float(stat_S1(x[m], yrel[m][None, :], st)[0]) if r['n_strata_S1'] > 0 else np.nan
            r['S2'] = float(stat_S2(x[m], yrel[m][None, :], st)[0])
            r['S3_auc'] = float(stat_S3(x[m], lab[m][None, :], st)[0]) if n_lab > 0 else np.nan
            r['note'] = '記述のみ(解釈しない)' if yr == 2018 else ''
            rows.append(r)
    return pd.DataFrame(rows)


def bootstrap(F, spec, best, meta, yrel, lab, n_boot, log):
    """市町村単位のブートストラップ(市町村を復元抽出。重複した市町村は別の層として扱う)。"""
    muni = meta.muniCd.to_numpy()
    st = meta.muniyear.to_numpy()
    um = np.unique(muni)
    rng = np.random.default_rng(12345)
    rows = []
    for _, b in best.iterrows():
        x = F[:, spec.index[spec.fname == b.fname][0]]
        out = {'S1': [], b.stat: []} if b.stat != 'S1' else {'S1': []}
        for _ in range(n_boot):
            pick = rng.choice(um, len(um), replace=True)
            idx, s2 = [], []
            for c_, mname in enumerate(pick):
                ii = np.flatnonzero(muni == mname)
                idx.append(ii)
                s2.append(np.array([f'{c_}|{v}' for v in st[ii]]))
            idx = np.concatenate(idx)
            s2 = np.concatenate(s2)
            for key in out:
                Y = lab if key == 'S3' else yrel
                v = stat_by_name(key)(x[idx], Y[idx][None, :], s2)[0]
                out[key].append(v)
        for key, vals in out.items():
            vals = np.asarray(vals, float)
            obs = stat_by_name(key)(x, (lab if key == 'S3' else yrel)[None, :], st)[0]
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            rows.append({'family': b.family, 'combo': f'{b.fname}・{b.stat}', 'statistic': key, 'observed': obs,
                         'ci_low': lo, 'ci_high': hi, 'n_boot': n_boot, 'n_nan': int(np.isnan(vals).sum())})
            log(f'  {b.family} 最良 {b.fname}・{b.stat}: {key} = {obs:.3f}、95%区間 [{lo:.3f}, {hi:.3f}]')
    return pd.DataFrame(rows)


def style(ax):
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)


def plot_spec_curve(res, out_dir):
    """specification curve: 上段に並べ替えた z、下段に組の内容。左 Y1 の族(+ 対照 P の S1・S2)、右 Y2 の族(+ 対照 P の S3)。"""
    panels = [('Y1(相対収量、S1・S2)', res[(res.family == 'Y1') | ((res.family == '対照') & (res.stat != 'S3'))], COL_Y1),
              ('Y2(主ラベル、S3)', res[(res.family == 'Y2') | ((res.family == '対照') & (res.stat == 'S3'))], COL_Y2)]
    feats = ['F1 N_post', 'F2 減少率', 'F3 N_dry', 'F4 N', 'F5 イベント数', 'F6 P(対照)']
    rows_lab = feats + ['k = 1', 'k = 2', 'k = 3', 'k = 5', 'θ0', 'θ2', 'θ5', 'S1', 'S2', 'S3']
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[46, 23], hspace=0.06, wspace=0.12)
    for col, (title, d, color) in enumerate(panels):
        d = d.sort_values('z').reset_index(drop=True)
        ax = fig.add_subplot(gs[0, col])
        x = np.arange(len(d))
        ctrl = (d.family == '対照').to_numpy()
        ax.axhline(0, color=MUTED, lw=0.8)
        for zz in [-1.96, 1.96]:
            ax.axhline(zz, color=MUTED, lw=0.6, ls=':')
        ax.vlines(x, 0, d.z, color=np.where(ctrl, COL_P, color), lw=2)
        ax.scatter(x[~ctrl], d.z[~ctrl], s=22, color=color, zorder=3, label='族の組')
        ax.scatter(x[ctrl], d.z[ctrl], s=30, color=COL_P, marker='D', zorder=3, label='対照 P(族の外)')
        ax.set_xlim(-0.8, len(d) - 0.2)
        ax.set_xticks([])
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        if col == 0:
            ax.set_ylabel('z(置換の帰無分布で標準化)', color=INK)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        style(ax)
        ax2 = fig.add_subplot(gs[1, col], sharex=ax)
        for i, r in d.iterrows():
            on = [r.family_feat, f'k = {int(r.k)}' if (r.family_feat.startswith('F1') and np.isfinite(r.k)) else None,
                  r.rain if r.rain != '—' else None, r.stat]
            for lbl in on:
                if lbl is None:
                    continue
                yy = rows_lab.index(lbl)
                ax2.scatter(i, yy, s=14, color=COL_P if ctrl[i] else INK, marker='s')
        ax2.set_yticks(range(len(rows_lab)))
        ax2.set_yticklabels(rows_lab if col == 0 else [''] * len(rows_lab), fontsize=8, color=INK)
        ax2.set_ylim(len(rows_lab) - 0.5, -0.5)
        for yy in [len(feats) - 0.5, len(feats) + 3.5, len(feats) + 6.5]:
            ax2.axhline(yy, color=GRID, lw=0.8)
        ax2.set_xlabel('組(z の小さい順)', color=INK)
        for sp in ['top', 'right']:
            ax2.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, '点線は |z| = 1.96(補正前の目安)。F2・F3 の k は3で固定。減少は負の値なので、'
             'N_post などで z < 0 は「大きく減る圃場ほど相対収量が高い / 多収」の向き', fontsize=8, color=MUTED)
    fig.savefig(os.path.join(out_dir, 'fig_spec_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_null_max(maxes, res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, fam, color in zip(axes, ['Y1', 'Y2'], [COL_Y1, COL_Y2]):
        mx = maxes[fam]
        ax.hist(mx, bins=60, color=color, alpha=0.55, lw=0)
        q = np.quantile(mx, 0.95)
        zb = res.loc[res.family == fam, 'z'].abs().max()
        ax.axvline(q, color=MUTED, ls='--', lw=1)
        ax.axvline(zb, color=INK, lw=1.6)
        ax.text(q, ax.get_ylim()[1] * 0.92, f' 95%点 {q:.2f}', fontsize=8, color=MUTED)
        ax.text(zb, ax.get_ylim()[1] * 0.80, f' 実データの最大 {zb:.2f}', fontsize=8, color=INK)
        ax.set_title(f'{fam} の族: 置換の各回の最大 |z|', loc='left', fontsize=10, color=INK)
        ax.set_xlabel('最大 |z|', color=INK)
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_null_max.png'), dpi=150)
    plt.close(fig)


def plot_best(F, spec, best_y1, meta, yrel, out_dir):
    x = F[:, spec.index[spec.fname == best_y1.fname][0]]
    st = meta.muniyear.to_numpy()
    ux, uy = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for g in np.unique(st):
        idx = np.flatnonzero((st == g) & np.isfinite(x))
        ux[idx] = (stats.rankdata(x[idx]) - 0.5) / len(idx)
        uy[idx] = (stats.rankdata(yrel[idx]) - 0.5) / len(idx)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cols = {2016: '#2a78d6', 2017: '#eb6834', 2018: '#1f9e74'}
    for yr, c in cols.items():
        m = (meta.year == yr).to_numpy()
        ax.scatter(ux[m], uy[m], s=26, color=c, alpha=0.75, edgecolor='white', lw=0.6, label=f'{yr}({int(m.sum())}件)')
    ax.set_xlabel(f'{best_y1.fname} の層内の正規化した順位', color=INK)
    ax.set_ylabel('y_rel の層内の正規化した順位', color=INK)
    ax.set_title(f'Y1 の最良の組({best_y1.fname}・{best_y1.stat}、z = {best_y1.z:.2f})', loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_best_scatter.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本体
def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['trial', 'main'], required=True)
    a = ap.parse_args()
    B = 100 if a.part == 'trial' else 10000
    out = os.path.join(OUT_MAIN, 'trial') if a.part == 'trial' else OUT_MAIN
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'run_log.txt'), 'w', encoding='utf-8')

    def log(m):
        print(m)
        logf.write(m + '\n')
        logf.flush()

    t0 = time.time()
    log(f'段階36({a.part}、置換 {B} 回)。設定は D-016')
    if a.part == 'trial':
        log('試運転: コードの確認と所要時間の見積もりだけ。p はレポートに載せず、結果を見て組や設定を変えない(D-016)')
    TH, meta = build_inputs(log)
    yrel = meta.y_rel.to_numpy(float)
    lab = meta[LABEL].to_numpy(float)
    strata = meta.muniyear.to_numpy()
    F, spec, info = features(TH)
    log('\n[特徴量の記録]\n' + info.round(2).to_string(index=False))
    log(f'特徴量 {F.shape[1]} 個(対照 P を含む)。NaN の数: ' +
        ', '.join(f'{nm} {int(np.isnan(F[:, j]).sum())}' for j, nm in enumerate(spec.fname) if np.isnan(F[:, j]).any()))
    fdf = pd.concat([meta[['field_id', 'year', 'muniCd', 'muniyear', 'yield', 'y_rel', LABEL]],
                     pd.DataFrame(F, columns=spec.fname)], axis=1)
    fdf.to_csv(os.path.join(out, 'features.csv'), index=False, encoding='utf-8-sig')
    cor = pd.DataFrame(F, columns=spec.fname).corr(method='spearman')
    cor.to_csv(os.path.join(out, 'feature_spearman_corr.csv'), encoding='utf-8-sig')

    # 層内置換(圃場と (y_rel, ラベル) の対応を同時に並べ替える)
    t1 = time.time()
    perm = np.empty((B + 1, len(meta)), dtype=int)
    perm[0] = np.arange(len(meta))
    for i in range(B):
        perm[i + 1] = permute_within(np.arange(len(meta)), strata, np.random.default_rng([0, i]))
    Yrel, Lab = yrel[perm], lab[perm]
    chk = all(np.allclose(np.sort(Yrel[i][strata == g]), np.sort(yrel[strata == g])) for i in [1, B] for g in np.unique(strata))
    log(f'\n置換の確認: 層の中だけで動いている {chk}。各層のラベルの数が保たれる '
        f'{all(np.nansum(Lab[B][strata == g]) == np.nansum(lab[strata == g]) for g in np.unique(strata))}')
    combos, T = compute_all(F, spec, Yrel, Lab, strata)
    t2 = time.time()
    log(f'統計量の計算: {len(combos)} 組 × {B + 1} 回、{t2 - t1:.1f} 秒')
    log(f'組の数: Y1 {int((combos.family == "Y1").sum())}、Y2 {int((combos.family == "Y2").sum())}、'
        f'対照 {int((combos.family == "対照").sum())}、合計 {len(combos)}')

    # 確認: 実データの統計量を、別の書き方(scipy.stats)で計算し直す
    j = spec.index[spec.fname == 'N_post_k3|θ0'][0]
    x = F[:, j]
    rr, ww = [], []
    for g in np.unique(strata):
        idx = np.flatnonzero(strata == g)
        if len(idx) >= MIN_S1:
            rr.append(stats.spearmanr(x[idx], yrel[idx])[0])
            ww.append(len(idx))
    s1_chk = np.average(rr, weights=ww)
    s1_mine = combos.index[(combos.fname == 'N_post_k3|θ0') & (combos.stat == 'S1')][0]
    log(f'確認(N_post_k3|θ0): S1 = {T[0, s1_mine]:.6f}、scipy.spearmanr での計算 {s1_chk:.6f}')
    pairs = [(x[i] > x[k_]) + 0.5 * (x[i] == x[k_]) for g in np.unique(strata) for i in np.flatnonzero((strata == g) & (lab == 1))
             for k_ in np.flatnonzero((strata == g) & (lab == 0))]
    s3_mine = combos.index[(combos.fname == 'N_post_k3|θ0') & (combos.stat == 'S3')][0]
    log(f'確認(N_post_k3|θ0): S3 = {T[0, s3_mine]:.6f}、ペアを1つずつ数えた値 {np.mean(pairs):.6f}(ペア {len(pairs)} 組)')

    res, maxes = test(combos, T)
    res.to_csv(os.path.join(out, 'combos.csv'), index=False, encoding='utf-8-sig')
    np.save(os.path.join(out, 'null_max_Y1.npy'), maxes['Y1'])
    np.save(os.path.join(out, 'null_max_Y2.npy'), maxes['Y2'])
    np.save(os.path.join(out, 'stat_matrix.npy'), T)

    log('\n==== 結果 ====')
    for fam in ['Y1', 'Y2']:
        d = res[res.family == fam].assign(az=lambda x: x.z.abs()).sort_values('az', ascending=False)
        q = np.quantile(maxes[fam], 0.95)
        log(f'\n[{fam} の族、{len(d)} 組] 置換の最大 |z| の95%点 {q:.3f}。選択込みの p の最小 {d.p_sel.min():.4f}、'
            f'補正前 p < 0.05 {int((d.p_raw < 0.05).sum())} 組、Holm 後 p < 0.05 {int((d.p_holm < 0.05).sum())} 組、'
            f'z > 0 の組 {int((d.z > 0).sum())} / {len(d)}')
        log(d.head(10)[['fname', 'stat', 'value', 'z', 'p_raw', 'p_holm', 'p_sel']].round(4).to_string(index=False))
    log('\n[対照 P]\n' + res[res.family == '対照'][['fname', 'stat', 'value', 'z', 'p_raw']].round(4).to_string(index=False))

    best = pd.concat([res[res.family == fam].assign(az=lambda x: x.z.abs()).sort_values('az', ascending=False).head(1)
                      for fam in ['Y1', 'Y2']])
    yr = by_year(F, spec, best, meta, yrel, lab)
    yr.to_csv(os.path.join(out, 'best_by_year.csv'), index=False, encoding='utf-8-sig')
    log('\n[最良の組の年ごとの向き(2018年は記述のみ)]\n' + yr.round(3).to_string(index=False))
    log('\n[最良の組の効果量(市町村単位のブートストラップ)]')
    bs = bootstrap(F, spec, best, meta, yrel, lab, N_BOOT if a.part == 'main' else 100, log)
    bs.to_csv(os.path.join(out, 'best_bootstrap.csv'), index=False, encoding='utf-8-sig')
    sg = res[res.family != '対照'].groupby('family').apply(
        lambda d: pd.Series({'n': len(d), 'z_pos': int((d.z > 0).sum()), 'z_neg': int((d.z < 0).sum())}))
    sg.to_csv(os.path.join(out, 'sign_counts.csv'), encoding='utf-8-sig')
    sf = res[res.family != '対照'].groupby(['family', 'family_feat']).apply(
        lambda d: pd.Series({'n': len(d), 'z_pos': int((d.z > 0).sum()), 'z_mean': d.z.mean()}))
    sf.to_csv(os.path.join(out, 'sign_by_feature.csv'), encoding='utf-8-sig')
    log('\n[向きの数]\n' + sg.to_string() + '\n' + sf.round(2).to_string())

    plot_spec_curve(res, out)
    plot_null_max(maxes, res, out)
    plot_best(F, spec, best.iloc[0], meta, yrel, out)
    log(f'\n所要時間 {time.time() - t0:.1f} 秒(うち統計量 {t2 - t1:.1f} 秒)')
    logf.close()


if __name__ == '__main__':
    main()
