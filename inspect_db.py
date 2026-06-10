import sqlite3

# FieldData_fieldid.db の調査
db_path = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\FieldData_fieldid.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('=== Tables in FieldData_fieldid.db ===')
for t in tables:
    tname = t[0]
    print(f'\nTable: {tname}')
    cursor.execute(f'PRAGMA table_info({tname})')
    cols = cursor.fetchall()
    for c in cols:
        print(f'  {c}')
    cursor.execute(f'SELECT COUNT(*) FROM {tname}')
    print(f'  Row count: {cursor.fetchone()[0]}')

conn.close()

print('\n\n')

# weather_database_fieldid.db の調査
db_path2 = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\weather_database_fieldid.db'
conn2 = sqlite3.connect(db_path2)
cursor2 = conn2.cursor()

cursor2.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables2 = cursor2.fetchall()
print('=== Tables in weather_database_fieldid.db ===')
for t in tables2:
    tname = t[0]
    print(f'\nTable: {tname}')
    cursor2.execute(f'PRAGMA table_info({tname})')
    cols = cursor2.fetchall()
    for c in cols:
        print(f'  {c}')
    cursor2.execute(f'SELECT COUNT(*) FROM {tname}')
    print(f'  Row count: {cursor2.fetchone()[0]}')

conn2.close()
