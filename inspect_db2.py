import sqlite3
import pandas as pd

FIELD_DB = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\FieldData_fieldid.db'
WEATHER_DB = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\weather_database_fieldid.db'

conn_f = sqlite3.connect(FIELD_DB)
conn_w = sqlite3.connect(WEATHER_DB)

# 1. Questionaire: field_id がある件数とない件数
print("=== Questionaire: field_id の有無 ===")
df_q = pd.read_sql("SELECT field_id, year, yield FROM Questionaire", conn_f)
print(f"全件数        : {len(df_q)}")
print(f"field_id あり : {df_q['field_id'].notna().sum()}")
print(f"field_id なし : {df_q['field_id'].isna().sum()}")

df_q_valid = df_q[df_q['field_id'].notna()].copy()
df_q_valid['field_id'] = df_q_valid['field_id'].astype(int)
print(f"\nfield_id ありの year 範囲: {df_q_valid['year'].min()} 〜 {df_q_valid['year'].max()}")
print(f"ユニーク field_id 数: {df_q_valid['field_id'].nunique()}")
print(f"yield がNULLでない件数: {df_q_valid['yield'].notna().sum()}")

# 2. weather_data: 日付範囲と対象 field_id
print("\n=== weather_data: 日付・field_id 範囲 ===")
row = conn_w.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT field_id) FROM weather_data").fetchone()
print(f"date 範囲: {row[0]} 〜 {row[1]}")
print(f"ユニーク field_id 数: {row[2]}")

# 3. Questionaire に出現する field_id が weather_data に存在するか確認
fids = df_q_valid['field_id'].unique().tolist()
placeholders = ','.join(['?' for _ in fids])
result = conn_w.execute(
    f"SELECT COUNT(DISTINCT field_id) FROM weather_data WHERE field_id IN ({placeholders})", fids
).fetchone()
print(f"\nQuestionaire の field_id ({len(fids)}個) のうち weather_data に存在するもの: {result[0]}個")

# 4. サンプル: 特定の field_id × year の気象データ件数
sample_fid = int(df_q_valid['field_id'].iloc[0])
sample_year = int(df_q_valid['year'].iloc[0])
cnt = conn_w.execute(
    "SELECT COUNT(*) FROM weather_data WHERE field_id=? AND date LIKE ?",
    (sample_fid, f"{sample_year}%")
).fetchone()[0]
print(f"\nサンプル: field_id={sample_fid}, year={sample_year} の気象行数: {cnt}")

# 5. 全 field_id × year の気象データ件数分布
print("\n=== 各 (field_id, year) ペアの気象行数 分布（上位20件）===")
df_sample = pd.read_sql(
    f"SELECT field_id, SUBSTR(date,1,4) as yr, COUNT(*) as cnt FROM weather_data "
    f"WHERE field_id IN ({placeholders}) GROUP BY field_id, yr",
    conn_w, params=fids
)
print(df_sample['cnt'].describe())
print(f"\n(field_id, year) ペア数: {len(df_sample)}")

conn_f.close()
conn_w.close()
