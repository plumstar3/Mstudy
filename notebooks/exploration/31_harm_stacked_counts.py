"""
31_harm_stacked_counts.py
==========================
病害特徴量の各値ごとに、Low / Mid(該当しない) / High 圃場の件数を積み上げ棒で示す。
対象圃場 = 25〜29 と同じ「対象市町村の圃場×年」（地域内上下33%の High/Low + それ以外=Mid）。
  ・Mid には、件数が少なく分類できなかった市町村×年（Skip）も含める。
  ・積み上げ順: 下から Low(青) → Mid(灰) → High(赤)

[出力] outputs/yield_pooled_regional/
    31_harm_stacked_binary_flag.png   (sick/weed/wet/bug + 気象フラグ6種, 2×5)
    31_harm_stacked_ordinal.png       (unripen/lay/loss, 値1〜4)
    31_harm_stacked_counts.csv
"""
import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB    = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
NDVI_CSV    = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'yield_pooled_regional')
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_QUANTILE, MIN_RECORDS, MIN_PER_YEAR = 0.33, 12, 6
HARM_BINARY_COLS    = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN = {'sick': [3], 'bug': [4]}
HARM_ORDINAL_COLS   = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS      = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_ALL            = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS
ORDER  = ['Low', 'Mid', 'High']
COLORS = {'Low': '#6BB3E0', 'Mid': '#DCDCDC', 'High': '#F07B72'}

# ── データ ───────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
q = pd.read_sql("""SELECT field_id, year, yield FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2015 AND 2018""", conn)
harm = pd.read_sql(f"SELECT field_id, year, {', '.join(HARM_ALL)} FROM harm WHERE field_id IS NOT NULL", conn)
conn.close()
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
harm['field_id'] = harm['field_id'].astype(int); harm['year'] = harm['year'].astype(int)
for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
for c in HARM_FLAG_COLS:
    harm[c] = harm[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm[c] = pd.to_numeric(harm[c], errors='coerce').fillna(0).astype(int)
agg = {c: 'mean' for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS}
agg.update({c: 'max' for c in HARM_FLAG_COLS})
harm = harm.groupby(['field_id', 'year'])[HARM_ALL].agg(agg).reset_index()
for c, bad in HARM_BINARY_UNKNOWN.items():
    harm.loc[harm[c].isin(bad), c] = np.nan

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
q = q.merge(geo[['field_id', 'muniCd']].drop_duplicates('field_id'), on='field_id', how='left')
ndvi = pd.read_csv(NDVI_CSV, usecols=['field_id', 'year'])
key = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))
q = q[[(f, y) in key for f, y in zip(q['field_id'], q['year'])]].reset_index(drop=True)

cnt = q.groupby('muniCd').size()
q = q[q['muniCd'].isin(cnt[cnt >= MIN_RECORDS].index)].copy()
q['group'] = 'Mid'
for (muni, yr), g in q.groupby(['muniCd', 'year']):
    if len(g) < MIN_PER_YEAR:
        continue                       # Skip → Mid(該当しない) のまま
    lo, hi = g['yield'].quantile(GROUP_QUANTILE), g['yield'].quantile(1 - GROUP_QUANTILE)
    q.loc[g.index[g['yield'] <= lo], 'group'] = 'Low'
    q.loc[g.index[g['yield'] >= hi], 'group'] = 'High'
d = q.merge(harm, on=['field_id', 'year'], how='left')
print(f'対象: {len(d)}件  ', d['group'].value_counts().to_dict())

# 二値化: 病害は 1=なし / >=2=あり, フラグは 0/1
for c in HARM_BINARY_COLS:
    d[c + '_v'] = (d[c] >= 2).astype(float).where(d[c].notna())
for c in HARM_FLAG_COLS:
    d[c + '_v'] = d[c].astype(float)
for c in HARM_ORDINAL_COLS:
    d[c + '_v'] = d[c].round().where(d[c].notna())

rows = []
def stacked(ax, col, values, labels):
    sub = d[d[col + '_v'].notna()]
    bottoms = np.zeros(len(values))
    ct = {g: np.array([((sub[col + '_v'] == v) & (sub['group'] == g)).sum() for v in values]) for g in ORDER}
    for g in ORDER:
        ax.bar(range(len(values)), ct[g], bottom=bottoms, color=COLORS[g], edgecolor='black', lw=0.8,
               label=g, width=0.8)
        for i, (b, n) in enumerate(zip(bottoms, ct[g])):
            if n >= 3:
                ax.text(i, b + n / 2, str(n), ha='center', va='center', fontsize=7)
        bottoms += ct[g]
    for i, v in enumerate(values):
        rows.append({'variable': col, 'value': labels[i], 'Low': int(ct['Low'][i]),
                     'Mid': int(ct['Mid'][i]), 'High': int(ct['High'][i]), 'total': int(bottoms[i])})
    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels)
    ax.set_title(col, fontsize=10); ax.set_ylabel('Count')
    for i, t in enumerate(bottoms):
        ax.text(i, t, f'{int(t)}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 図1: 二値 + フラグ（2×5）
cols = HARM_BINARY_COLS + HARM_FLAG_COLS
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, c in zip(axes.flatten(), cols):
    stacked(ax, c, [0, 1], ['No', 'Yes'])
    ax.set_xlabel('Presence (0=No, 1=Yes)')
axes[0, 0].legend(title='群', fontsize=8)
fig.suptitle('病害・気象フラグ: 値ごとの圃場数（Low=青 / Mid=灰 / High=赤、Midは分類対象外含む）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '31_harm_stacked_binary_flag.png'), dpi=160); plt.close()

# 図2: 順序（1〜4）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, c in zip(axes, HARM_ORDINAL_COLS):
    stacked(ax, c, [1, 2, 3, 4], ['1(無)', '2(少)', '3(中)', '4(多)'])
    ax.set_xlabel('被害程度')
axes[0].legend(title='群', fontsize=8)
fig.suptitle('病害順序変数: 値ごとの圃場数（Low=青 / Mid=灰 / High=赤）', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '31_harm_stacked_ordinal.png'), dpi=160); plt.close()

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, '31_harm_stacked_counts.csv'), index=False, encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))
print('完了')
