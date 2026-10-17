import sqlite3, pandas as pd, numpy as np

ndvi = pd.read_csv('data/processed/ndvi_daily_smooth.csv')
ndvi_fields = set(zip(ndvi['field_id'].astype(int), ndvi['year'].astype(int)))

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
q = pd.read_sql('''
    SELECT field_id, year, yield FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL
      AND harvest_date IS NOT NULL AND year BETWEEN 2015 AND 2018
''', conn)
conn.close()

q['field_id'] = q['field_id'].astype(int)
q['year']     = q['year'].astype(int)
q = q[q.apply(lambda r: (r['field_id'], r['year']) in ndvi_fields, axis=1)].copy()

geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
# city列が市町村名
q = q.merge(geo[['field_id','muniCd','city']].drop_duplicates('field_id'), on='field_id', how='left')

# 市町村ごとの集計
by_muni = q.groupby(['muniCd','city']).agg(
    n_records   =('field_id','count'),
    n_fields    =('field_id','nunique'),
    n_years     =('year','nunique'),
    yield_median=('yield','median'),
).reset_index().sort_values('n_records', ascending=False)

print('圃場×年レコード数が多い市町村（上位25）:')
print(by_muni.head(25).to_string(index=False))
print()

# 年別内訳（n_records >= 8の市町村のみ）
big_munis = by_muni[by_muni['n_records'] >= 8]['muniCd'].tolist()
by_year = q[q['muniCd'].isin(big_munis)].groupby(['muniCd','city','year']).size().reset_index(name='n')
pivot = by_year.pivot_table(index=['muniCd','city'], columns='year', values='n', fill_value=0)
pivot['合計'] = pivot.sum(axis=1)
print('市町村×年 別件数（合計8件以上）:')
print(pivot.sort_values('合計', ascending=False).to_string())
