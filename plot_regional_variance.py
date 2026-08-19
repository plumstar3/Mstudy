import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

# 日本語フォント設定
matplotlib.rcParams['font.family'] = 'MS Gothic'

FIELD_DB = os.path.join('data', 'processed', 'FieldData_fieldid.db')
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')

# 1. 収量データの読み込み (特定の年のみ抽出)
TARGET_YEAR = 2016
conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql(f'''
    SELECT field_id, year, yield
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year = {TARGET_YEAR}
''', conn)
conn.close()

quest_df['yield'] = pd.to_numeric(quest_df['yield'], errors='coerce')
quest_df = quest_df.dropna(subset=['yield'])
quest_df['field_id'] = quest_df['field_id'].astype(int)

# 2. 市町村データの読み込み
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city']].drop_duplicates('field_id')

# 3. 結合
df = quest_df.merge(geo_df, on='field_id', how='inner')
df = df.dropna(subset=['city'])

# サンプル数が多い市町村トップ12を抽出
top_cities = list(df['city'].value_counts().head(12).index)
# 岩見沢市を必ず含める処理
if '岩見沢市' not in top_cities and '岩見沢市' in df['city'].values:
    top_cities[-1] = '岩見沢市'  # 12番目と入れ替え

plot_df = df[df['city'].isin(top_cities)]

# 4. プロット作成
plt.figure(figsize=(12, 6), facecolor='white')

# 箱ひげ図とストリッププロット
sns.boxplot(data=plot_df, x='city', y='yield', color='#a8d5e2', showfliers=False, width=0.6)
sns.stripplot(data=plot_df, x='city', y='yield', color='#1d3557', alpha=0.7, jitter=0.2, size=6)

plt.title('2016年の同市町村における大豆収量', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('収量 (kg/10a)', fontsize=12)
plt.xlabel('市町村', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', alpha=0.4)

out_dir = os.path.join('outputs', 'analysis')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'regional_yield_variance_{TARGET_YEAR}.png')
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"プロットを保存しました: {out_path}")
