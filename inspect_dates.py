import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
df = pd.read_sql('''
    SELECT field_id, year, seed_date, harvest_date, yield
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
''', conn)
conn.close()

df['seed_date']    = pd.to_datetime(df['seed_date'],    errors='coerce')
df['harvest_date'] = pd.to_datetime(df['harvest_date'], errors='coerce')

total = len(df)
has_seed    = df['seed_date'].notna().sum()
has_harvest = df['harvest_date'].notna().sum()
has_both    = (df['seed_date'].notna() & df['harvest_date'].notna()).sum()

print(f'Total samples   : {total}')
print(f'seed_date valid : {has_seed} ({has_seed/total*100:.1f}%)')
print(f'harvest_date valid: {has_harvest} ({has_harvest/total*100:.1f}%)')
print(f'Both valid      : {has_both} ({has_both/total*100:.1f}%)')

d = df.dropna(subset=['seed_date','harvest_date']).copy()
d['seed_doy']     = d['seed_date'].dt.dayofyear
d['harvest_doy']  = d['harvest_date'].dt.dayofyear
d['growing_days'] = (d['harvest_date'] - d['seed_date']).dt.days

print()
print('=== seed_date (DOY) ===')
print(d['seed_doy'].describe().round(1))

print()
print('=== harvest_date (DOY) ===')
print(d['harvest_doy'].describe().round(1))

print()
print('=== growing days ===')
print(d['growing_days'].describe().round(1))

print()
print('May 1  = DOY 121 (leap) / 120 (non-leap) <- window start')
print('Dec 27 = DOY 361 (leap) / 360 (non-leap) <- window end')
print(f'Actual min seed DOY  : {d["seed_doy"].min()}')
print(f'Actual max seed DOY  : {d["seed_doy"].max()}')
print(f'Actual min harvest DOY: {d["harvest_doy"].min()}')
print(f'Actual max harvest DOY: {d["harvest_doy"].max()}')

print()
print('=== Per-year stats ===')
for yr in [2015, 2016, 2017, 2018]:
    sub = d[d['year'] == yr]
    if len(sub) == 0:
        continue
    print(f'  {yr}: n={len(sub):3d}  '
          f'seed DOY=[{sub["seed_doy"].min()}-{sub["seed_doy"].max()}]  '
          f'harvest DOY=[{sub["harvest_doy"].min()}-{sub["harvest_doy"].max()}]  '
          f'grow_days=[{sub["growing_days"].min()}-{sub["growing_days"].max()}]')

# yield vs growing_days の相関
print()
corr = d[['yield','seed_doy','harvest_doy','growing_days']].corr()
print('=== yield との相関 ===')
print(corr['yield'].round(3))

# harvest_dateがない(None)サンプルの収量分布
no_harvest = df[df['harvest_date'].isna() & df['seed_date'].notna()]
has_harvest_df = d.copy()
print()
print(f'harvest_date なしサンプル: {len(no_harvest)}件')
if len(no_harvest) > 0:
    print(f'  yield mean={no_harvest["yield"].mean():.1f}  std={no_harvest["yield"].std():.1f}')
print(f'harvest_date ありサンプル: {len(has_harvest_df)}件')
print(f'  yield mean={has_harvest_df["yield"].mean():.1f}  std={has_harvest_df["yield"].std():.1f}')
