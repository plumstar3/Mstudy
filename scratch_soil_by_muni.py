import sqlite3, pandas as pd

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
solid = pd.read_sql('SELECT field_id, year, date FROM Solid WHERE field_id IS NOT NULL AND date IS NOT NULL', conn)
solid['field_id'] = pd.to_numeric(solid['field_id'], errors='coerce').dropna().astype(int)
solid['year'] = solid['year'].astype(int)

quest = pd.read_sql('SELECT field_id, year, yield FROM Questionaire WHERE yield IS NOT NULL AND year BETWEEN 2015 AND 2018', conn)
quest = quest.dropna(subset=['field_id'])
quest['field_id'] = quest['field_id'].astype(int)
quest['year'] = quest['year'].astype(int)
conn.close()

# 土壌×収量マッチ
matched = solid.merge(quest[['field_id','year']], on=['field_id','year'], how='inner')

# 地域付与
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce')
matched = matched.merge(geo[['field_id','muniCd','city']].drop_duplicates('field_id'), on='field_id', how='left')

# タイミング分類
matched['date'] = pd.to_datetime(matched['date'])
matched['month'] = matched['date'].dt.month
matched['timing'] = matched['month'].apply(
    lambda m: 'pre_sow' if m in [5,6,7] else ('post_harv' if m in [10,11,12] else 'exclude'))

# 除外なしの件数集計
valid = matched[matched['timing'] != 'exclude'].copy()
by_muni = valid.groupby(['muniCd','city','timing']).size().unstack(fill_value=0).reset_index()
if 'pre_sow' not in by_muni.columns:
    by_muni['pre_sow'] = 0
if 'post_harv' not in by_muni.columns:
    by_muni['post_harv'] = 0
by_muni['total'] = by_muni['pre_sow'] + by_muni['post_harv']
by_muni = by_muni.sort_values('total', ascending=False).reset_index(drop=True)

print('=== 市町村別 土壌×収量マッチ件数 (上位20) ===')
print(by_muni.head(20).to_string(index=False))
print()
print('=== 播種前後データがある市町村 ===')
pre = by_muni[by_muni['pre_sow'] > 0].sort_values('pre_sow', ascending=False)
print(pre.to_string(index=False))
