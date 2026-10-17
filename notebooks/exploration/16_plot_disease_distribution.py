import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# フォント設定
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# パス
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB    = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'disease_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI = 20202 # 松本市

# 病害カラム
HARM_BINARY_COLS    = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN = {'sick': [3], 'bug': [4]}
HARM_ORDINAL_COLS   = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS      = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_ALL_DB_COLS    = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS

# データベース読み込み
conn = sqlite3.connect(FIELD_DB)
harm = pd.read_sql(f"SELECT field_id, year, {', '.join(HARM_ALL_DB_COLS)} FROM harm WHERE field_id IS NOT NULL", conn)

# 型変換・前処理
harm['field_id'] = harm['field_id'].astype(int)
harm['year']     = harm['year'].astype(int)

for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
for c, unk in HARM_BINARY_UNKNOWN.items():
    harm[c] = harm[c].where(~harm[c].isin(unk), other=pd.NA)
for c in HARM_FLAG_COLS:
    harm[c] = harm[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm[c] = pd.to_numeric(harm[c], errors='coerce')

# 松本市に絞り込み、収量データを結合してグループ分け
quest = pd.read_sql("""
    SELECT field_id, year, yield FROM Questionaire 
    WHERE field_id IS NOT NULL AND year BETWEEN 2015 AND 2018 AND yield IS NOT NULL
""", conn)
conn.close()
geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce')
mat_ids = geo[geo['muniCd'] == TARGET_MUNI]['field_id'].unique()

df = harm[harm['field_id'].isin(mat_ids)].copy()
quest = quest[quest['field_id'].isin(mat_ids)].copy()

# 年ごとのグループ分け
quest['group'] = 'Mid'
for yr in [2015, 2016, 2017, 2018]:
    m = quest['year'] == yr
    sub = quest[m]
    if len(sub) >= 6:
        lo = sub['yield'].quantile(0.33)
        hi = sub['yield'].quantile(0.67)
        quest.loc[m & (quest['yield'] <= lo), 'group'] = 'Low'
        quest.loc[m & (quest['yield'] >= hi), 'group'] = 'High'

df = df.merge(quest[['field_id', 'year', 'group']], on=['field_id', 'year'], how='left')
df['group'] = df['group'].fillna('Mid')

# ==========================================
# プロットの作成
# ==========================================
# 1. 順序変数のヒストグラム (High/Low/Mid色分け)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
group_order = ['High', 'Mid', 'Low']
palette = {'High': '#E74C3C', 'Mid': 'lightgray', 'Low': '#3498DB'}

for i, col in enumerate(HARM_ORDINAL_COLS):
    data = df[[col, 'group']].dropna()
    sns.histplot(data=data, x=col, hue='group', hue_order=group_order, multiple='stack', 
                 palette=palette, ax=axes[i], discrete=True, shrink=0.8, legend=True)
    axes[i].set_title(col)
    axes[i].set_xlabel('Score')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks([1, 2, 3, 4])
    axes[i].set_xticklabels(['1', '2', '3', '4'])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'matsumoto_ordinal_distribution.png'))
plt.close()

# 2. 2値変数/フラグ変数の出現数 (High/Low/Mid色分け)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(HARM_BINARY_COLS + HARM_FLAG_COLS):
    data = df[[col, 'group']].dropna()
    if col in HARM_BINARY_COLS:
        # 1:なし, 2:あり (2以上をありとする)
        data[col] = (data[col] >= 2).astype(int)
    else:
        data[col] = (data[col] == 1).astype(int)
    
    sns.histplot(data=data, x=col, hue='group', hue_order=group_order, multiple='stack',
                 palette=palette, ax=axes[i], discrete=True, shrink=0.8, legend=False)
    axes[i].set_title(col)
    axes[i].set_xlabel('Presence (0=No, 1=Yes)')
    axes[i].set_ylabel('Count')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No', 'Yes'])

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'matsumoto_binary_distribution.png'))
plt.close()

print("分布プロットを保存しました: matsumoto_ordinal_distribution.png, matsumoto_binary_distribution.png")
