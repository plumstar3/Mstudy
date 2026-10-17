"""
15_feature_local_effect.py
==========================
全特徴量を「同じ市町村×同じ年の中での偏差」で横並び評価する。

[手順]
  1. 母集団: VWC記録のある圃場×年 (2016-2018)。全特徴量を同一サンプルで比較する。
  2. 各特徴量・収量から 市町村×年 の平均を引く（圃場数 >= MIN_CELL のセルのみ）。
  3. Spearman ρ(収量偏差, 特徴量偏差) + ブートストラップ95%CI + BH法q値。
     参考として補正なしのρ（生の収量 vs 生の特徴量）も並べる。

[特徴量]
  気象 9変数×GDD3期間(mean) / 病害 5 / 栽培 2 / VWC要約 7 / NDVI(CSVがあれば)
  ※ breed(カテゴリ)は対象外

[出力] outputs/soil_analysis/15_feature_local_effect.csv / .png
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')
for _fn in ['IPAexGothic', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)
FIELD_DB, WEATHER_DB = P('data', 'processed', 'FieldData_fieldid.db'), P('data', 'processed', 'weather_database_fieldid.db')
GDD_CSV, GEO_CSV = P('outputs', 'gdd', 'gdd_daily.csv'), P('outputs', 'reverse_geocode', 'field_Addresses.csv')
NDVI_CSV = P('data', 'processed', 'ndvi_features_v2.csv')
OUT = P('outputs', 'soil_analysis'); os.makedirs(OUT, exist_ok=True)

MIN_CELL, N_BOOT, SEED = 3, 2000, 42
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
HARM_SCORE = ['sick', 'wet', 'unripen', 'weed']; HARM_FLAG = ['typhoon']
NDVI_COLS = ['ndvi_dap030', 'ndvi_dap045', 'ndvi_dap060', 'ndvi_dap075', 'ndvi_dap090',
             'ndvi_dap105', 'ndvi_dap120', 'ndvi_r5r6_mean']

# ── 1. 母集団 ────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql('''SELECT field_id, year, yield, between_lines, between_stocks, seed_date, harvest_date
    FROM Questionaire WHERE field_id IS NOT NULL AND yield IS NOT NULL AND seed_date IS NOT NULL
    AND harvest_date IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
harm = pd.read_sql(f"SELECT field_id, year, {', '.join(HARM_SCORE + HARM_FLAG)} FROM harm "
                   "WHERE field_id IS NOT NULL AND year BETWEEN 2016 AND 2018", conn)
conn.close()

q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date']); q['harvest_date'] = pd.to_datetime(q['harvest_date'])
q['gd'] = (q['harvest_date'] - q['seed_date']).dt.days; q = q[q['gd'] > 0]
for c in ['between_lines', 'between_stocks']:
    q[c] = pd.to_numeric(q[c], errors='coerce')

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce'); sm = sm.dropna(subset=['field_id'])
sm['field_id'] = sm['field_id'].astype(int); sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce'); sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])
key = set(zip(sm['field_id'], sm['year']))
q = q[[(a, b) in key for a, b in zip(q['field_id'], q['year'])]].drop_duplicates(['field_id', 'year']).reset_index(drop=True)

geo = pd.read_csv(GEO_CSV, encoding='utf-8-sig'); geo['field_id'] = geo['field_id'].astype(int)
q = q.merge(geo[['field_id', 'muniCd']].drop_duplicates('field_id'), on='field_id', how='left')
print(f'母集団: {len(q)} 圃場×年 / 年別 {q.groupby("year").size().to_dict()}')

# ── 2. 特徴量 ───────────────────────────────────────────────
feat_group = {}

# 気象（GDD期間別 mean）
fids = sorted(q['field_id'].unique().tolist())
gdd = pd.read_csv(GDD_CSV, encoding='utf-8-sig'); gdd['date'] = pd.to_datetime(gdd['date'])
cum = [c for c in gdd.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
gdd['period'] = 1; gdd.loc[gdd[cum] > 600, 'period'] = 2; gdd.loc[gdd[cum] > 1000, 'period'] = 3
gdd = gdd[gdd['field_id'].isin(fids)][['field_id', 'year', 'date', 'period']]
cw = sqlite3.connect(WEATHER_DB)
w = pd.read_sql(f"SELECT field_id, date, {', '.join(WEATHER_COLS)} FROM weather_data WHERE field_id IN "
                f"({','.join('?' * len(fids))}) AND CAST(SUBSTR(date,1,4) AS INTEGER) BETWEEN 2016 AND 2018",
                cw, params=fids)
cw.close()
w['field_id'] = w['field_id'].astype(int); w['date'] = pd.to_datetime(w['date'])
mg = gdd.merge(w, on=['field_id', 'date'], how='left')
wf = mg.groupby(['field_id', 'year', 'period'])[WEATHER_COLS].mean().unstack('period')
wf.columns = [f'{v}_p{int(p)}' for v, p in wf.columns]
wf = wf.reset_index()
for c in wf.columns[2:]:
    feat_group[c] = '気象'
D = q.merge(wf, on=['field_id', 'year'], how='left')

# 病害
for c in HARM_SCORE:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
harm['sick'] = harm['sick'].where(harm['sick'] != 3)
for c in HARM_FLAG:
    harm[c] = pd.to_numeric(harm[c].replace({'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0, True: 1, False: 0}),
                            errors='coerce').fillna(0)
harm['field_id'] = harm['field_id'].astype(int); harm['year'] = harm['year'].astype(int)
agg = {c: 'mean' for c in HARM_SCORE}; agg.update({c: 'max' for c in HARM_FLAG})
hg = harm.groupby(['field_id', 'year']).agg(agg).reset_index()
D = D.merge(hg, on=['field_id', 'year'], how='left')
for c in HARM_SCORE + HARM_FLAG:
    feat_group[c] = '病害'
D['typhoon'] = D['typhoon'].fillna(0)
for c in ['between_lines', 'between_stocks']:
    feat_group[c] = '栽培'

# VWC 要約
s2 = sm.merge(q[['field_id', 'year', 'seed_date', 'gd']], on=['field_id', 'year'])
s2['dap'] = (s2['date'] - s2['seed_date']).dt.days / s2['gd'] * 100
rows = []
for (f, y), g in s2.groupby(['field_id', 'year']):
    v, d_ = g['VWC'].values, g['dap'].values
    r = {'field_id': f, 'year': y}
    for nm, lo, hi in [('vwc_early', 20, 50), ('vwc_mid', 50, 75), ('vwc_late', 75, 100)]:
        m = (d_ >= lo) & (d_ < hi)
        r[nm] = v[m].mean() if m.sum() else np.nan
    r['vwc_mean'], r['vwc_std'] = v.mean(), v.std()
    r['vwc_cv'] = v.std() / v.mean() if v.mean() else np.nan
    r['vwc_slope'] = linregress(d_, v)[0] if len(v) >= 3 and np.ptp(d_) > 0 else np.nan
    rows.append(r)
vf = pd.DataFrame(rows)
for c in vf.columns[2:]:
    feat_group[c] = 'VWC'
D = D.merge(vf, on=['field_id', 'year'], how='left')

# NDVI
if os.path.exists(NDVI_CSV):
    nd = pd.read_csv(NDVI_CSV); nd['field_id'] = nd['field_id'].astype(int); nd['year'] = nd['year'].astype(int)
    if 'auc' in nd.columns:
        nd = nd[nd['auc'] > 0]
    cols = [c for c in NDVI_COLS if c in nd.columns]
    D = D.merge(nd[['field_id', 'year'] + cols].drop_duplicates(['field_id', 'year']),
                on=['field_id', 'year'], how='left')
    for c in cols:
        feat_group[c] = 'NDVI'

feats = list(feat_group)

# ── 3. 市町村×年の偏差 ───────────────────────────────────────
cell = D.groupby(['muniCd', 'year'])['field_id'].transform('nunique')
L = D[cell >= MIN_CELL].copy()
print(f'補正対象 (市町村×年 圃場数>={MIN_CELL}): {len(L)} 件 / セル {L[["muniCd","year"]].drop_duplicates().shape[0]}')
grp = L.groupby(['muniCd', 'year'])
L['y_loc'] = L['yield'] - grp['yield'].transform('mean')

rng = np.random.default_rng(SEED)
out = []
for c in feats:
    x_raw = D[c]
    ok_raw = x_raw.notna()
    rho_raw = spearmanr(D.loc[ok_raw, 'yield'], x_raw[ok_raw])[0] if ok_raw.sum() > 10 and x_raw[ok_raw].nunique() > 1 else np.nan
    xd = L[c] - grp[c].transform('mean')
    ok = xd.notna() & (xd.abs() > 0).any()
    xs, ys = xd[xd.notna()].values, L.loc[xd.notna(), 'y_loc'].values
    if len(xs) < 20 or np.std(xs) == 0:
        continue
    rho, p = spearmanr(ys, xs)
    idx = np.arange(len(xs)); bs = []
    for _ in range(N_BOOT):
        b = rng.choice(idx, len(idx))
        if np.std(xs[b]) > 0:
            bs.append(spearmanr(ys[b], xs[b])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append(dict(feature=c, group=feat_group[c], n=len(xs), rho_raw=rho_raw,
                    rho_local=rho, ci_lo=lo, ci_hi=hi, p=p))
R = pd.DataFrame(out)
# Benjamini-Hochberg
o = R['p'].argsort().values; m = len(R)
q_ = np.empty(m); prev = 1.0
for rank, i in list(enumerate(o, 1))[::-1]:
    prev = min(prev, R['p'].iloc[i] * m / rank); q_[i] = prev
R['q_bh'] = q_
R = R.reindex(R['rho_local'].abs().sort_values(ascending=False).index).reset_index(drop=True)
R.to_csv(os.path.join(OUT, '15_feature_local_effect.csv'), index=False, encoding='utf-8-sig')

pd.set_option('display.width', 200)
print('\n=== 同一市町村×年内の Spearman ρ（|ρ|降順） ===')
print(R.round(3).to_string(index=False))
print(f"\nq<0.05: {(R['q_bh'] < 0.05).sum()} / {len(R)}  |  p<0.05(未補正): {(R['p'] < 0.05).sum()}")
print('\nグループ別 |ρ_local| 中央値 / |ρ_raw| 中央値')
print(R.groupby('group').agg(n_feat=('feature', 'size'), abs_local=('rho_local', lambda s: s.abs().median()),
                             abs_raw=('rho_raw', lambda s: s.abs().median())).round(3).to_string())

# ── 図 ──────────────────────────────────────────────────────
colors = {'気象': '#3498DB', '病害': '#E67E22', '栽培': '#7F8C8D', 'VWC': '#27AE60', 'NDVI': '#8E44AD'}
Rp = R.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(Rp))))
ax.barh(Rp.index, Rp['rho_local'], color=[colors[g] for g in Rp['group']], alpha=.85)
ax.errorbar(Rp['rho_local'], Rp.index, xerr=[Rp['rho_local'] - Rp['ci_lo'], Rp['ci_hi'] - Rp['rho_local']],
            fmt='none', ecolor='k', lw=.8, capsize=2)
ax.scatter(Rp['rho_raw'], Rp.index, marker='D', s=22, facecolors='none', edgecolors='r', zorder=5,
           label='補正なし ρ（参考）')
ax.axvline(0, color='gray'); ax.set_yticks(Rp.index); ax.set_yticklabels(Rp['feature'], fontsize=8)
for g, c in colors.items():
    ax.barh([], [], color=c, label=g)
ax.set_xlabel('Spearman ρ（収量偏差 vs 特徴量偏差, 市町村×年内）  棒=補正後±95%CI')
ax.set_title(f'同一市町村×年内での特徴量と収量の関係 (n≈{int(R["n"].median())})')
ax.legend(loc='lower right', fontsize=8); ax.grid(axis='x', ls='--', alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '15_feature_local_effect.png'), dpi=170); plt.close()
print('\n保存:', OUT)
