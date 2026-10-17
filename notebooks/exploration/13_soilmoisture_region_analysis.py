"""
13_soilmoisture_region_analysis.py
==================================
同一市町村内での多収圃場 vs 低収圃場の VWC 時系列比較。

[地域選定]
  SolidMoisture と収量が揃う圃場×年が最も多い市町村（muniCd単位）を自動選択
  （TARGET_MUNI_CD を指定すれば上書き可）。

[グループ定義]
  年次効果を除くため、市町村内の「年平均との差（収量偏差）」を計算し、
  全年プールした偏差の上位33% → High / 下位33% → Low、中間は除外。

[分析]
  DAP%（播種0〜収穫100）を 5% ビンに分割し、High/Low の平均VWC±SE と
  Mann-Whitney U 検定（各群 n>=MIN_PER_GROUP のビンのみ）。
  n が小さいため探索的分析（多重比較補正なし）。

[出力] outputs/soil_analysis/13_region_*.png / .csv
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
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

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEO_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI_CD = None      # None → 圃場×年数が最多の市町村を自動選択
GROUP_QUANTILE = 0.33
BIN_W = 5
MIN_PER_GROUP = 4
COL_HIGH, COL_LOW = '#E74C3C', '#3498DB'

# ── データ読み込み ───────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
quest = pd.read_sql('''SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
sm = sm.dropna(subset=['field_id']); sm['field_id'] = sm['field_id'].astype(int)
sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])

quest['field_id'] = quest['field_id'].astype(int); quest['year'] = quest['year'].astype(int)
quest['seed_date'] = pd.to_datetime(quest['seed_date'], errors='coerce')
quest['harvest_date'] = pd.to_datetime(quest['harvest_date'], errors='coerce')
quest['growing_days'] = (quest['harvest_date'] - quest['seed_date']).dt.days
quest = quest[quest['growing_days'] > 0]

geo = pd.read_csv(GEO_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo = geo[['field_id', 'muniCd', 'prefecture', 'city']].drop_duplicates('field_id')

fy = sm[['field_id', 'year']].drop_duplicates().merge(quest, on=['field_id', 'year']).merge(geo, on='field_id')

# ── 地域選定 ─────────────────────────────────────────────────
cnt = fy.groupby(['muniCd', 'prefecture', 'city']).size().sort_values(ascending=False)
print('市町村別 圃場×年数（上位5）:'); print(cnt.head(5).to_string())
muni = TARGET_MUNI_CD if TARGET_MUNI_CD else cnt.index[0][0]
reg = fy[fy['muniCd'] == muni].copy()
name = f"{reg['prefecture'].iloc[0]}{reg['city'].iloc[0]}"
print(f'\n選択地域: {name} (muniCd={muni}) {len(reg)}件 / 年別 {reg.groupby("year").size().to_dict()}')

# ── グループ定義（市町村×年平均との偏差） ────────────────────
reg['y_dev'] = reg['yield'] - reg.groupby('year')['yield'].transform('mean')
lo, hi = reg['y_dev'].quantile(GROUP_QUANTILE), reg['y_dev'].quantile(1 - GROUP_QUANTILE)
reg['group'] = np.where(reg['y_dev'] >= hi, 'High', np.where(reg['y_dev'] <= lo, 'Low', 'Mid'))
print('\nグループ数:', reg['group'].value_counts().to_dict())
print(reg.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())
print(reg.groupby('group')[['yield', 'y_dev']].mean().round(1).to_string())

# ── DAP% ────────────────────────────────────────────────────
d = sm.merge(reg[['field_id', 'year', 'seed_date', 'growing_days', 'group']], on=['field_id', 'year'])
d['dap'] = (d['date'] - d['seed_date']).dt.days / d['growing_days'] * 100
d['bin'] = (np.floor(d['dap'] / BIN_W) * BIN_W).astype(int)
d = d[(d['dap'] >= 0) & (d['dap'] <= 100)]

# 圃場×年×ビンで平均（観測点でなく圃場を単位に検定）
fb = d.groupby(['field_id', 'year', 'group', 'bin'])['VWC'].mean().reset_index()

rows = []
for b in sorted(fb['bin'].unique()):
    s = fb[fb['bin'] == b]
    h, l = s[s['group'] == 'High']['VWC'], s[s['group'] == 'Low']['VWC']
    if len(h) >= MIN_PER_GROUP and len(l) >= MIN_PER_GROUP:
        p = mannwhitneyu(h, l, alternative='two-sided')[1]
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
    else:
        p, sig = np.nan, 'n.a.'
    rows.append(dict(bin=b + BIN_W / 2, n_high=len(h), n_low=len(l),
                     high_mean=h.mean(), high_se=h.sem(), low_mean=l.mean(), low_se=l.sem(),
                     diff=h.mean() - l.mean(), p=p, sig=sig))
st = pd.DataFrame(rows)
st.to_csv(os.path.join(OUT_DIR, '13_region_stats.csv'), index=False, encoding='utf-8-sig')
print('\n=== DAP%ビン別 High vs Low ===')
print(st.round(4).to_string(index=False))
sg = st[st['sig'].isin(['*', '**'])]
print(f'\n有意(p<0.05)ビン: {len(sg)} / 検定可能 {int((st["sig"] != "n.a.").sum())}')

# ── 可視化 ──────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
v = st[st['sig'] != 'n.a.']
for key, col, lab in [('high', COL_HIGH, 'High'), ('low', COL_LOW, 'Low')]:
    n = int((reg['group'] == lab).sum())
    ax[0].plot(v['bin'], v[f'{key}_mean'], color=col, lw=2.5, label=f'{lab}群 (n={n})')
    ax[0].fill_between(v['bin'], v[f'{key}_mean'] - v[f'{key}_se'],
                       v[f'{key}_mean'] + v[f'{key}_se'], color=col, alpha=0.2)
for _, r in v.iterrows():
    if r['sig'] in ('*', '**'):
        ax[0].text(r['bin'], ax[0].get_ylim()[1] if False else max(r['high_mean'], r['low_mean']) + 0.03,
                   r['sig'], ha='center', color='k', fontsize=11)
ax[0].set_title(f'{name}: 群平均 VWC ± SE（探索的, n小）')
ax[0].set_xlabel('DAP% (播種0 → 収穫100)'); ax[0].set_ylabel('VWC'); ax[0].legend(); ax[0].grid(ls='--', alpha=.35)

for (f, y, g), s in fb.groupby(['field_id', 'year', 'group']):
    if g == 'Mid':
        continue
    s = s.sort_values('bin')
    ax[1].plot(s['bin'] + BIN_W / 2, s['VWC'], color=COL_HIGH if g == 'High' else COL_LOW, lw=1, alpha=.6)
ax[1].set_title('圃場別の VWC 推移（赤=High / 青=Low）')
ax[1].set_xlabel('DAP%'); ax[1].grid(ls='--', alpha=.35)
plt.tight_layout()
out = os.path.join(OUT_DIR, '13_region_vwc.png')
plt.savefig(out, dpi=170, bbox_inches='tight'); plt.close()
print('図保存:', out)
