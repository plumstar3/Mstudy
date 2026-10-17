import sqlite3, pandas as pd

conn = sqlite3.connect('data/processed/weather_database_fieldid.db')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables:", tables['name'].tolist())
df = pd.read_sql("SELECT * FROM weather_data LIMIT 3", conn)
print(df.dtypes)
print(df.head(3))

# 圃場データのスキーマも確認
conn2 = sqlite3.connect('data/processed/FieldData_fieldid.db')
tables2 = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn2)
print("\nFieldData tables:", tables2['name'].tolist())
q = pd.read_sql("SELECT * FROM Questionaire LIMIT 3", conn2)
print(q.columns.tolist())
conn2.close()
conn.close()
