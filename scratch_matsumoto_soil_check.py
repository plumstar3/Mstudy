import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd'] = pd.to_numeric(geo['muniCd'], errors='coerce')
mat_ids = set(geo[geo['muniCd']==20202]['field_id'].unique())

solid = pd.read_sql(
    'SELECT field_id, year, place, date FROM Solid WHERE field_id IS NOT NULL AND date IS NOT NULL',
    conn)
solid['field_id'] = pd.to_numeric(solid['field_id'], errors='coerce').dropna().astype(int)
solid['year'] = solid['year'].astype(int)
solid = solid[solid['field_id'].isin(mat_ids)].copy()
solid['date'] = pd.to_datetime(solid['date'])
solid['month'] = solid['date'].dt.month
solid['timing'] = solid['month'].apply(
    lambda m: 'pre_sow' if m in [5,6,7] else ('post_harv' if m in [10,11,12] else 'exclude'))

quest = pd.read_sql(
    'SELECT field_id, year, yield FROM Questionaire WHERE yield IS NOT NULL AND year BETWEEN 2015 AND 2018',
    conn)
conn.close()

quest = quest.dropna(subset=['field_id'])
quest['field_id'] = quest['field_id'].astype(int)
quest['year'] = quest['year'].astype(int)
quest = quest[quest['field_id'].isin(mat_ids)].copy()

# 08スタイル: 年ごとH/L定義 (上下33%)
quest['group'] = 'Mid'
for yr in [2015,2016,2017,2018]:
    m = quest['year']==yr
    sub = quest[m]
    if len(sub) < 6: continue
    lo = sub['yield'].quantile(0.33)
    hi = sub['yield'].quantile(0.67)
    quest.loc[m & (quest['yield'] <= lo), 'group'] = 'Low'
    quest.loc[m & (quest['yield'] >= hi), 'group'] = 'High'

print('=== 松本市 収量グループ ===')
print(quest.groupby(['year','group']).size().unstack(fill_value=0))

merged = solid.merge(quest[['field_id','year','yield','group']], on=['field_id','year'], how='inner')
merged_valid = merged[merged['timing'] != 'exclude']

print()
print('=== 土壌×収量マッチ件数 (タイミング × グループ) ===')
print(merged_valid.groupby(['timing','group']).size().unstack(fill_value=0))
print()
print('=== 年×タイミング×グループ ===')
print(merged_valid.groupby(['year','timing','group']).size().unstack(fill_value=0))
print()
print(f'松本市 field_id 一覧: {sorted(mat_ids)}')
print(f'土壌データあり field_id: {sorted(solid["field_id"].unique())}')
