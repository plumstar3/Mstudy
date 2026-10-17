import sqlite3, pandas as pd
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print('テーブル一覧:', tables['name'].tolist())

# 病害データをどのテーブルから読んでいるかを eval_yield_deviation.py で確認
# コードでは "Harm" という名前で読んでいる可能性
for tname in tables['name'].tolist():
    if 'harm' in tname.lower() or 'disease' in tname.lower() or 'sick' in tname.lower():
        print(f'\n=== {tname} スキーマ ===')
        schema = pd.read_sql(f"PRAGMA table_info('{tname}')", conn)
        print(schema[['name','type']].to_string(index=False))

        print(f'\n=== {tname} サンプル ===')
        df = pd.read_sql(f"SELECT * FROM '{tname}' LIMIT 20", conn)
        print(df.to_string(index=False))

        print(f'\n=== field_id重複確認 ===')
        dup = pd.read_sql(f"""
            SELECT field_id, year, COUNT(*) as cnt
            FROM '{tname}'
            GROUP BY field_id, year
            HAVING cnt > 1
            LIMIT 10
        """, conn)
        print(f'重複あり行数: {len(dup)}')
        print(dup.to_string(index=False))

conn.close()
