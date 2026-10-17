import sqlite3, pandas as pd

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')

# E1〜E10（2015年）を 2015-10-01 に更新
places = [f'E{i}' for i in range(1, 11)]
for p in places:
    cur = conn.execute('UPDATE Solid SET date = ? WHERE place = ? AND year = ?',
                       ('2015-10-01', p, 2015))
    status = 'OK' if cur.rowcount > 0 else '警告: マッチなし'
    print(f'  {p} 2015年 → {status}')

conn.commit()

# 確認
result = pd.read_sql(
    "SELECT place, year, date FROM Solid WHERE year=2015 AND place LIKE 'E%' ORDER BY place",
    conn
)
print('\n=== 更新後 Solid 2015年 E系 ===')
print(result.to_string(index=False))

# 年別集計
print()
for yr in [2015, 2016, 2017, 2018]:
    n     = conn.execute(f'SELECT COUNT(*) FROM Solid WHERE year={yr} AND date IS NOT NULL').fetchone()[0]
    total = conn.execute(f'SELECT COUNT(*) FROM Solid WHERE year={yr}').fetchone()[0]
    print(f'  {yr}年: {n}/{total} 件に日付あり')

conn.close()
print('\n完了。')
