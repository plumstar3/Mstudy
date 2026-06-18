import sqlite3, pandas as pd, os

# 気象DBの確認
db_path = os.path.join('data', 'processed', 'weather_database_fieldid.db')
conn = sqlite3.connect(db_path)

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print('Tables:', tables['name'].tolist())

for t in tables['name'].tolist():
    cols = pd.read_sql(f'PRAGMA table_info({t})', conn)
    print(f'\n=== {t} columns ===')
    print(cols[['name','type']].to_string(index=False))
    df = pd.read_sql(f'SELECT * FROM "{t}" LIMIT 5', conn)
    print(df.to_string(index=False))
conn.close()

# 既存の parquet も確認
print('\n=== soybean_dataset.parquet columns ===')
df2 = pd.read_parquet(os.path.join('data','processed','soybean_ts2vec','soybean_dataset.parquet'))
print(list(df2.columns))
