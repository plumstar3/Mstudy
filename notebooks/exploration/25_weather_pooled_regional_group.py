"""
25_weather_pooled_regional_group.py
====================================
「地域内 上位/下位33% を全地域でプール」した多収群 vs 低収群の気象特徴量分析。

[グループ定義]  (08 / 23 と同じ地域内定義)
  市町村 × 年 の収量 上位33% → High / 下位33% → Low（その年6件以上）
  対象市町村: NDVI保有レコード数 >= MIN_RECORDS
  → 各地域・各年の High をすべて集めて 1つの多収群、Low も同様に 1つの低収群とする。

[解析]
  A. 気象時系列（相対生育ステージ 0〜100% に正規化, 年別z-score）
     - High/Low の平均 ± 95%CI、各時刻 MWU + FDR補正 → 有意区間を帯で表示
  B. 期間集約特徴量（early/mid/late 3期間の平均, 期間内の合計(降水・日射・日照)）
     - プール MWU + 効果量(rank-biserial) + FDR
     - 頑健性確認: 地域×年セルごとの (High平均 - Low平均) の符号検定/Wilcoxon（地域構成の影響を除去）

[出力]  outputs/yield_pooled_regional/
    25_weather_ts_pooled.png
    25_weather_period_stats.csv
    25_weather_period_effect.png
    25_group_assignment.csv
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── パス・定数 ────────────────────────────────────────────────
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB    = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB  = os.path.join(_ROOT, 'data', 'processed', 'weather_database_fieldid.db')
NDVI_CSV    = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'yield_pooled_regional')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33
MIN_RECORDS    = 12     # 市町村の最低レコード数（08と同じ）
MIN_PER_YEAR   = 6      # 市町村×年の最低件数（08と同じ）
N_RESAMPLE     = 100
ALPHA          = 0.05

WEATHER_COLS = ['APCPRA', 'TMP_mea', 'WIND', 'GSR', 'RH', 'TMP_max', 'TMP_min', 'SSD', 'SWE']
WEATHER_LABELS = {
    'APCPRA': '降水量 (mm)',   'TMP_mea': '日平均気温 (℃)',  'WIND': '風速 (m/s)',
    'GSR':    '全天日射量',     'RH':      '相対湿度 (%)',     'TMP_max': '日最高気温 (℃)',
    'TMP_min':'日最低気温 (℃)', 'SSD':    '日照時間 (h)',     'SWE':  '積雪水当量',
}
SUM_COLS = ['APCPRA', 'GSR', 'SSD']   # 期間合計も算出する変数
PERIODS  = {'early': (0, 33), 'mid': (33, 67), 'late': (67, 100.0001)}

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'
COL_SIG  = '#AAAAAA'
RELATIVE_PTS = np.linspace(0, 100, N_RESAMPLE)

# ── データ読み込み ───────────────────────────────────────────
print('=' * 60 + '\n  データ読み込み\n' + '=' * 60)
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()
q['field_id']     = q['field_id'].astype(int)
q['year']         = q['year'].astype(int)
q['seed_date']    = pd.to_datetime(q['seed_date'])
q['harvest_date'] = pd.to_datetime(q['harvest_date'])
q['growing_days'] = (q['harvest_date'] - q['seed_date']).dt.days
q = q[q['growing_days'] > 0]

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
q = q.merge(geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id'),
            on='field_id', how='left')

ndvi = pd.read_csv(NDVI_CSV, usecols=['field_id', 'year'])
ndvi_key = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))
q = q[[(f, y) in ndvi_key for f, y in zip(q['field_id'], q['year'])]].reset_index(drop=True)
print(f'  対象レコード（NDVI保有）: {len(q)} 件')

# ── グループ定義（地域内 33% → 全地域プール） ─────────────────
cnt = q.groupby('muniCd').size()
q = q[q['muniCd'].isin(cnt[cnt >= MIN_RECORDS].index)].copy()
q['group'] = 'Mid'
for (muni, yr), g in q.groupby(['muniCd', 'year']):
    if len(g) < MIN_PER_YEAR:
        q.loc[g.index, 'group'] = 'Skip'
        continue
    lo = g['yield'].quantile(GROUP_QUANTILE)
    hi = g['yield'].quantile(1 - GROUP_QUANTILE)
    q.loc[g.index[g['yield'] <= lo], 'group'] = 'Low'
    q.loc[g.index[g['yield'] >= hi], 'group'] = 'High'

# 標準化は Mid/Skip を含む対象市町村の全圃場で行い（07と同じ順序）、検定時に High/Low を抽出する
sel = q.reset_index(drop=True)
sel.to_csv(os.path.join(OUT_DIR, '25_group_assignment.csv'), index=False, encoding='utf-8-sig')
n_h, n_l = (sel['group'] == 'High').sum(), (sel['group'] == 'Low').sum()
print(f'  対象市町村: {sel["muniCd"].nunique()} / プール後 High={n_h} / Low={n_l}')
print(sel.groupby(['year', 'group']).size().unstack().to_string())

# ── 気象読み込み ─────────────────────────────────────────────
print('  気象データ読み込み...')
conn_w = sqlite3.connect(WEATHER_DB)
wraw = pd.read_sql(f"""
    SELECT field_id, date, {', '.join(WEATHER_COLS)} FROM weather_data
    WHERE field_id IN ({','.join(map(str, sel['field_id'].unique()))})
""", conn_w)
conn_w.close()
wraw['field_id'] = wraw['field_id'].astype(int)
wraw['date'] = pd.to_datetime(wraw['date'])
for c in WEATHER_COLS:
    wraw[c] = pd.to_numeric(wraw[c], errors='coerce')
wgrp = {fid: g for fid, g in wraw.groupby('field_id')}

# ── 相対生育ステージへ正規化 + 期間集約特徴量 ─────────────────
ts = {c: np.full((len(sel), N_RESAMPLE), np.nan) for c in WEATHER_COLS}
feat_rows = []
for i, r in tqdm(sel.iterrows(), total=len(sel), desc='気象正規化'):
    wf = wgrp.get(int(r['field_id']))
    row = {}
    if wf is not None:
        wf = wf.copy()
        wf['DAP'] = (wf['date'] - r['seed_date']).dt.days
        wf = wf[(wf['DAP'] >= 0) & (wf['DAP'] <= r['growing_days'])]
        if len(wf) >= 2:
            rel = wf['DAP'].values / r['growing_days'] * 100
            for c in WEATHER_COLS:
                v = wf[c].values.astype(float)
                m = ~np.isnan(v)
                if m.sum() >= 2:
                    ts[c][i] = np.interp(RELATIVE_PTS, rel[m], v[m])
                for pn, (lo, hi) in PERIODS.items():
                    pm = m & (rel >= lo) & (rel < hi)
                    row[f'{c}_mean_{pn}'] = float(v[pm].mean()) if pm.sum() else np.nan
                    if c in SUM_COLS:
                        row[f'{c}_sum_{pn}'] = float(v[pm].sum()) if pm.sum() else np.nan
                row[f'{c}_mean_all'] = float(v[m].mean()) if m.sum() else np.nan
                if c in SUM_COLS:
                    row[f'{c}_sum_all'] = float(v[m].sum()) if m.sum() else np.nan
    feat_rows.append(row)
feat = pd.DataFrame(feat_rows)
sel = pd.concat([sel, feat], axis=1)

# 年別 z-score（時系列は時刻点ごと、集約特徴量は年内で標準化）
yr_arr = sel['year'].values
for c in WEATHER_COLS:
    for yr in TARGET_YEARS:
        m = yr_arr == yr
        if m.sum() == 0:
            continue
        mu = np.nanmean(ts[c][m], axis=0)
        sd = np.nanstd(ts[c][m], axis=0)
        sd[sd == 0] = 1.0
        ts[c][m] = (ts[c][m] - mu) / sd
feat_cols = list(feat.columns)
for fc in feat_cols:
    sel[fc + '_z'] = sel.groupby('year')[fc].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))

mh = (sel['group'] == 'High').values
ml = (sel['group'] == 'Low').values

# ── A. 時系列プロット ────────────────────────────────────────
def ts_stats(mat):
    mu_h, mu_l = np.nanmean(mat[mh], 0), np.nanmean(mat[ml], 0)
    ci_h = 1.96 * np.nanstd(mat[mh], 0) / np.sqrt(mh.sum())
    ci_l = 1.96 * np.nanstd(mat[ml], 0) / np.sqrt(ml.sum())
    p = np.ones(N_RESAMPLE)
    for t in range(N_RESAMPLE):
        h = mat[mh, t]; l = mat[ml, t]
        h = h[~np.isnan(h)]; l = l[~np.isnan(l)]
        if len(h) >= 3 and len(l) >= 3:
            p[t] = mannwhitneyu(h, l, alternative='two-sided')[1]
    _, pc, _, _ = multipletests(p, method='fdr_bh')
    return mu_h, mu_l, ci_h, ci_l, pc < ALPHA

fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
for ax, c in zip(axes.flatten(), WEATHER_COLS):
    mu_h, mu_l, ci_h, ci_l, sig = ts_stats(ts[c])
    print(f'  [有意時刻%] {c}: {sig.mean() * 100:.0f}%')
    start = None
    for t in range(N_RESAMPLE + 1):
        on = t < N_RESAMPLE and sig[t]
        if on and start is None:
            start = t
        elif not on and start is not None:
            ax.axvspan(RELATIVE_PTS[start], RELATIVE_PTS[t - 1], color=COL_SIG, alpha=0.3, zorder=0)
            start = None
    ax.plot(RELATIVE_PTS, mu_h, color=COL_HIGH, lw=1.8, label=f'High (n={mh.sum()})')
    ax.fill_between(RELATIVE_PTS, mu_h - ci_h, mu_h + ci_h, color=COL_HIGH, alpha=0.2)
    ax.plot(RELATIVE_PTS, mu_l, color=COL_LOW, lw=1.8, label=f'Low (n={ml.sum()})')
    ax.fill_between(RELATIVE_PTS, mu_l - ci_l, mu_l + ci_l, color=COL_LOW, alpha=0.2)
    ax.set_title(f'{WEATHER_LABELS[c]}  [有意時刻 {sig.mean() * 100:.0f}%]', fontsize=10)
    ax.set_ylabel('標準化値(年別)', fontsize=8)
    ax.grid(True, ls='--', alpha=0.5)
axes[0, 0].legend(fontsize=8)
for ax in axes[2]:
    ax.set_xlabel('相対生育ステージ (%)', fontsize=8)
fig.suptitle('気象時系列: 地域内上位/下位33%をプールした多収群 vs 低収群（灰帯=FDR<0.05）',
             fontsize=13, y=1.01)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, '25_weather_ts_pooled.png')
plt.savefig(out1, dpi=200, bbox_inches='tight'); plt.close()
print(f'  保存: {out1}')

# ── B. 期間集約特徴量の検定 ──────────────────────────────────
rows = []
for fc in feat_cols:
    zc = fc + '_z'
    h = sel.loc[mh, zc].dropna(); l = sel.loc[ml, zc].dropna()
    if len(h) < 5 or len(l) < 5:
        continue
    u, p = mannwhitneyu(h, l, alternative='two-sided')
    rbc = 2 * u / (len(h) * len(l)) - 1     # rank-biserial (>0: Highが大きい)

    # 地域×年セル内の (High平均 - Low平均) → Wilcoxon（地域構成の影響を除去）
    diffs = []
    for _, g in sel.groupby(['muniCd', 'year']):
        gh = g.loc[g['group'] == 'High', zc].dropna(); gl = g.loc[g['group'] == 'Low', zc].dropna()
        if len(gh) and len(gl):
            diffs.append(gh.mean() - gl.mean())
    diffs = np.array(diffs)
    try:
        p_cell = wilcoxon(diffs)[1] if len(diffs) >= 6 and np.any(diffs != 0) else np.nan
    except Exception:
        p_cell = np.nan
    rows.append({'feature': fc, 'n_high': len(h), 'n_low': len(l),
                 'high_mean_z': h.mean(), 'low_mean_z': l.mean(),
                 'mwu_p': p, 'rank_biserial': rbc,
                 'n_cells': len(diffs), 'cell_mean_diff': diffs.mean() if len(diffs) else np.nan,
                 'cell_pos_ratio': (diffs > 0).mean() if len(diffs) else np.nan,
                 'cell_wilcoxon_p': p_cell})
st = pd.DataFrame(rows)
st['mwu_q'] = multipletests(st['mwu_p'], method='fdr_bh')[1]
valid = st['cell_wilcoxon_p'].notna()
st.loc[valid, 'cell_q'] = multipletests(st.loc[valid, 'cell_wilcoxon_p'], method='fdr_bh')[1]
st['sig'] = np.where(st['mwu_q'] < 0.01, '**', np.where(st['mwu_q'] < ALPHA, '*', 'ns'))
st = st.sort_values('mwu_p').reset_index(drop=True)
out_csv = os.path.join(OUT_DIR, '25_weather_period_stats.csv')
st.to_csv(out_csv, index=False, encoding='utf-8-sig')
print('\n=== 期間集約特徴量 (MWU, FDR) 上位 ===')
print(st.head(20)[['feature', 'high_mean_z', 'low_mean_z', 'rank_biserial', 'mwu_p', 'mwu_q',
                   'cell_pos_ratio', 'cell_wilcoxon_p', 'sig']].round(4).to_string(index=False))
print(f'\n有意 (FDR<{ALPHA}): {(st["mwu_q"] < ALPHA).sum()} / {len(st)}')

# 効果量バー（上位25）
top = st.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 9))
colors = [COL_HIGH if v > 0 else COL_LOW for v in top['rank_biserial']]
ax.barh(top['feature'], top['rank_biserial'], color=colors, alpha=0.85)
for y, (rb, s) in enumerate(zip(top['rank_biserial'], top['sig'])):
    ax.text(rb, y, f' {s}', va='center', ha='left' if rb >= 0 else 'right', fontsize=8)
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('rank-biserial 相関（+: Highが大きい / −: Lowが大きい）')
ax.set_title('期間集約気象特徴量の効果量（上位25, 年別z-score, **/*=FDR）')
ax.grid(axis='x', ls='--', alpha=0.5)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, '25_weather_period_effect.png')
plt.savefig(out2, dpi=200, bbox_inches='tight'); plt.close()
print(f'  保存: {out2}\n完了')
