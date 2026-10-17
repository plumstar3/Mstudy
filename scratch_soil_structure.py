import sqlite3, pandas as pd

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
solid = pd.read_sql('SELECT field_id, year, place, date FROM Solid WHERE field_id IS NOT NULL AND date IS NOT NULL', conn)
solid['field_id'] = solid['field_id'].astype(int)

# field_id×年 あたりの土壌レコード数
n_per = solid.groupby(['field_id','year']).size()
print('=== field_id×年 あたりの土壌レコード数の分布 ===')
print(n_per.value_counts().sort_index())
print(f'最大: {n_per.max()} 件 / 中央値: {n_per.median()} 件')
print()

# 複数レコードがある例を表示
multi_idx = n_per[n_per > 1]
print(f'複数レコードあり: {len(multi_idx)} 件の field_id×年')
if len(multi_idx) > 0:
    ex_fid, ex_yr = multi_idx.index[0]
    ex = solid[(solid['field_id']==ex_fid) & (solid['year']==ex_yr)]
    print(f'例: field_id={ex_fid}, year={ex_yr}')
    print(ex[['field_id','year','place','date']].to_string(index=False))

# 採取タイミング別の地域分布を確認
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
solid = solid.merge(geo[['field_id','city']].drop_duplicates('field_id'), on='field_id', how='left')
solid['date'] = pd.to_datetime(solid['date'])
solid['month'] = solid['date'].dt.month
solid['timing'] = solid['month'].apply(
    lambda m: 'pre_sow' if m in [5,6,7] else ('post_harv' if m in [10,11,12] else 'exclude')
)

print()
print('=== 採取タイミング別 地域分布（上位10市）===')
for timing in ['pre_sow', 'post_harv']:
    t = solid[solid['timing']==timing]
    top_city = t['city'].value_counts().head(10)
    print(f'\n[{timing}]')
    print(top_city.to_string())

conn.close()
