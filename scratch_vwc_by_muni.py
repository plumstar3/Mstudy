import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
sm = pd.read_sql('SELECT DISTINCT field_id, year FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
q = pd.read_sql('''SELECT field_id, year, yield FROM Questionaire WHERE yield IS NOT NULL
 AND seed_date IS NOT NULL AND harvest_date IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
conn.close()
sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce'); sm = sm.dropna(); sm['field_id'] = sm['field_id'].astype(int)
q = q.dropna(subset=['field_id']); q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
m = sm.merge(q, on=['field_id', 'year'])
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
m = m.merge(geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id'), on='field_id', how='left')
print(len(m), m['muniCd'].isna().sum())
t = m.groupby(['muniCd', 'city']).agg(n=('yield', 'size'), n_field=('field_id', 'nunique'),
    y2016=('year', lambda s: (s == 2016).sum()), y2017=('year', lambda s: (s == 2017).sum()),
    y2018=('year', lambda s: (s == 2018).sum()), ymean=('yield', 'mean'), ystd=('yield', 'std')).reset_index()
print(t.sort_values('n', ascending=False).head(12).to_string(index=False))
