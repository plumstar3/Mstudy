import sqlite3, pandas as pd

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('=== FieldData_fieldid.db テーブル一覧 ===')
for (t,) in tables:
    cols = [c[1] for c in conn.execute(f'PRAGMA table_info({t})').fetchall()]
    n    = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'\n  [{t}]  ({n} 行)')
    print(f'    カラム: {cols}')
conn.close()

print()
conn2 = sqlite3.connect('data/processed/weather_database_fieldid.db')
tables2 = conn2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('=== weather_database_fieldid.db テーブル一覧 ===')
for (t,) in tables2:
    cols = [c[1] for c in conn2.execute(f'PRAGMA table_info({t})').fetchall()]
    n    = conn2.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'\n  [{t}]  ({n} 行)')
    print(f'    カラム: {cols}')
conn2.close()
