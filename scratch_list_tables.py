import sqlite3

for dbpath in ['data/processed/FieldData_fieldid.db', 'data/processed/weather_database_fieldid.db']:
    print(f'\n=== {dbpath} ===')
    try:
        conn = sqlite3.connect(dbpath)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        for (t,) in tables:
            try:
                n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                # SoilMoisture関連かチェック
                flag = ' ← 土壌水分関連?' if any(k in t.lower() for k in ['soil','moisture','vwc','water']) else ''
                print(f'  {t:<35} {n:>10,} 件{flag}')
            except Exception as e:
                print(f'  {t:<35} エラー: {e}')
        conn.close()
    except Exception as e:
        print(f'  接続エラー: {e}')
