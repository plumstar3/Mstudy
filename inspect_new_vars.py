import sqlite3, pandas as pd, numpy as np

db_weather = 'data/processed/weather_database_fieldid.db'
db_field   = 'data/processed/FieldData_fieldid.db'

# 気象DB: APCPRA と WIND の有効データ期間確認
conn = sqlite3.connect(db_weather)
df = pd.read_sql('''
    SELECT date, field_id, APCPRA, WIND
    FROM weather_data
    WHERE field_id IS NOT NULL
    ORDER BY date
''', conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])
df_valid_apcp = df[df['APCPRA'].notna()]
df_valid_wind = df[df['WIND'].notna()]

print('=== APCPRA 有効データ ===')
print(f'  有効件数: {len(df_valid_apcp)} / {len(df)} ({len(df_valid_apcp)/len(df)*100:.1f}%)')
if len(df_valid_apcp) > 0:
    print(f'  期間: {df_valid_apcp["date"].min().date()} ~ {df_valid_apcp["date"].max().date()}')

print()
print('=== WIND 有効データ ===')
print(f'  有効件数: {len(df_valid_wind)} / {len(df)} ({len(df_valid_wind)/len(df)*100:.1f}%)')
if len(df_valid_wind) > 0:
    print(f'  期間: {df_valid_wind["date"].min().date()} ~ {df_valid_wind["date"].max().date()}')

# 対象年度（2015-2018）でフィルタ
df_target = df[df['date'].dt.year.between(2015, 2018)]
print()
print('=== 2015-2018 年度での有効率 ===')
for col in ['APCPRA', 'WIND']:
    valid_rate = df_target[col].notna().mean() * 100
    print(f'  {col}: {valid_rate:.1f}% 有効')

# Questionnaire: seed_date / harvest_date
conn2 = sqlite3.connect(db_field)
qdf = pd.read_sql('''
    SELECT field_id, year, seed_date, harvest_date, yield
    FROM Questionaire
    WHERE year BETWEEN 2015 AND 2018
      AND field_id IS NOT NULL AND yield IS NOT NULL
''', conn2)
conn2.close()

qdf['seed_date']    = pd.to_datetime(qdf['seed_date'],    errors='coerce')
qdf['harvest_date'] = pd.to_datetime(qdf['harvest_date'], errors='coerce')
both_valid = qdf['seed_date'].notna() & qdf['harvest_date'].notna()

print()
print('=== 播種日・収穫日の有無（2015-2018） ===')
print(f'  総サンプル数: {len(qdf)}')
print(f'  両方あり: {both_valid.sum()} ({both_valid.mean()*100:.1f}%)')
print(f'  seed のみ: {(qdf["seed_date"].notna() & qdf["harvest_date"].isna()).sum()}')
print(f'  どちらもなし: {(qdf["seed_date"].isna() & qdf["harvest_date"].isna()).sum()}')
print()
print('=== 年度別 有効サンプル数 ===')
for yr in [2015,2016,2017,2018]:
    sub = qdf[qdf['year']==yr]
    bv  = sub['seed_date'].notna() & sub['harvest_date'].notna()
    print(f'  {yr}: 総計={len(sub)}  両日付あり={bv.sum()}  ({bv.mean()*100:.0f}%)')
