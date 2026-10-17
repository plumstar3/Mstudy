import pandas as pd, sqlite3

raw = pd.read_csv('Data/raw/solid_date.csv', header=None, encoding='cp932', dtype=str)
data = raw.iloc[1:].reset_index(drop=True)
print('=== CSV 2015年の内容（col9=place, col10=date）===')
col9 = data.iloc[:, 9].fillna('')
col10 = data.iloc[:, 10].fillna('')
e_mask = col9.str.startswith('E', na=False)
d_mask = col9.str.startswith('D', na=False)
print(data[e_mask | d_mask][[9,10]].to_string())

print()
print('=== CSV 2015年 全place値 ===')
all_places_2015 = col9[col9 != ''].tolist()
print(sorted(set(all_places_2015)))

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
e_solid = pd.read_sql("SELECT place, year, date FROM Solid WHERE year=2015 AND place LIKE 'E%' ORDER BY place", conn)
print()
print('=== Solid 2015年 E系 ===')
if len(e_solid) > 0:
    print(e_solid.to_string(index=False))
else:
    print('（E系レコードなし — Solidテーブルに2015年E系の圃場が存在しない）')

all_solid_2015 = pd.read_sql("SELECT DISTINCT place FROM Solid WHERE year=2015 ORDER BY place", conn)
print()
print('=== Solid 2015年 全place ===')
print(all_solid_2015['place'].tolist())
conn.close()
