import sqlite3, glob

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('=== FieldData_fieldid.db のテーブル一覧 ===')
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f'  {t:<30} {n:>8,} 件')
conn.close()

print()
print('=== プロジェクト内の .db ファイル ===')
found = glob.glob('**/*.db', recursive=True)
for p in found:
    print(f'  {p}')

# 各DBのテーブル確認
for p in found:
    if 'FieldData_fieldid' in p:
        continue
    try:
        c = sqlite3.connect(p)
        ts = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        soil_related = [t for t in ts if 'soil' in t.lower() or 'moisture' in t.lower() or 'vwc' in t.lower()]
        if soil_related:
            print(f'\n  {p} に土壌水分関連テーブルあり: {soil_related}')
        c.close()
    except Exception as e:
        print(f'  {p}: エラー {e}')
