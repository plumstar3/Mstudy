import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')

# SolidMoistureのfield_id一覧
sm = pd.read_sql('SELECT DISTINCT field_id, year FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce').dropna().astype(int)
sm['year'] = sm['year'].astype(int)
print(f'SolidMoisture: {len(sm)} 件の field_id × year')
print(f'  field_id の数: {sm["field_id"].nunique()} 圃場')
print(f'  年: {sorted(sm["year"].unique())}')

# Questionnaireと結合してseed_dateが取れるか
quest = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2015 AND 2018
""", conn)
conn.close()

quest = quest.dropna(subset=['field_id'])
quest['field_id'] = quest['field_id'].astype(int)
quest['year'] = quest['year'].astype(int)
quest['seed_date']    = pd.to_datetime(quest['seed_date'], errors='coerce')
quest['harvest_date'] = pd.to_datetime(quest['harvest_date'], errors='coerce')
quest['growing_days'] = (quest['harvest_date'] - quest['seed_date']).dt.days

# マッチ確認
merged = sm.merge(quest[['field_id','year','yield','seed_date','harvest_date','growing_days']],
                  on=['field_id','year'], how='inner')

print(f'\nSolidMoisture × 収量・播種日マッチ数: {len(merged)} 件')
print(f'  マッチしたfield_id数: {merged["field_id"].nunique()} 圃場')
print(f'  マッチしなかった: {len(sm) - len(merged)} 件')
print(f'\n生育日数の分布:')
print(merged['growing_days'].describe().round(1))

# 収量の分布確認
print(f'\n収量の分布:')
print(merged['yield'].describe().round(1))

# 年別マッチ数
print('\n年別:')
print(merged.groupby('year').agg(
    n_fields=('field_id','nunique'),
    yield_mean=('yield','mean'),
    yield_std=('yield','std')
).round(1))
