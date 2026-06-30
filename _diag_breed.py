import sqlite3, pandas as pd
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
df = pd.read_sql('''
    SELECT field_id, year, breed FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
''', conn)
conn.close()
df['breed'] = df['breed'].astype(str).str.strip()

print(f'総行数: {len(df)}  欠損(NULL/空): {df["breed"].isin(["None","","nan"]).sum()}')
print(f'\n=== 品種の出現頻度 ===')
vc = df['breed'].value_counts()
print(vc.to_string())

print(f'\nユニーク品種数: {df["breed"].nunique()}')
print(f'\n=== 年度別 品種分布 ===')
print(pd.crosstab(df['year'], df['breed']).to_string())
