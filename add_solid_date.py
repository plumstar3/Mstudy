"""
solid_dateテーブルへのdate列追加スクリプト。
Data/raw/solid_date.csv から土壌採取日を読み込み、
FieldData_fieldid.db の Solid テーブルに date 列を追加して更新する。

[CSV構造]
  1行目: 年ヘッダ (2018, 2018, NaN, 2017, 2017, NaN, 2016, 2016, NaN, 2015, 2015)
  2行目以降: (place, date, separator) × 4年分 が横並び

[日付形式]
  - yyyy/mm/dd 形式: そのまま使用
  - mm月dd日 形式: ヘッダの年と組み合わせて yyyy/mm/dd に変換

[キー]
  Solid.place + Solid.year で突合
  圃場識別子は A01 → A1 のようにゼロ埋めを除去して正規化
"""

import sqlite3
import re
import pandas as pd
import numpy as np

CSV_PATH = 'Data/raw/solid_date.csv'
DB_PATH  = 'data/processed/FieldData_fieldid.db'

# =============================================================================
# 1. CSVの読み込みと解析
# =============================================================================
raw = pd.read_csv(CSV_PATH, header=None, encoding='cp932', dtype=str)
print(f'CSV shape: {raw.shape}')

def normalize_place(s):
    """A01 → A1 のようにゼロ埋めを除去する。"""
    if pd.isna(s) or str(s).strip() == '':
        return None
    s = str(s).strip()
    # 例: A01 → A1, B003 → B3 (先頭英字 + 数字のゼロパディングを除去)
    m = re.match(r'^([A-Za-z]+)0*(\d+)$', s)
    if m:
        return m.group(1) + m.group(2)
    return s


def parse_date(date_str, year):
    """
    日付文字列を yyyy-mm-dd 形式に変換する。
    対応形式:
      - 2018/10/5  → 2018-10-05
      - 5月16日   → {year}-05-16
      - H29.10.19 → 2017-10-19 （平成: H元年=1989, H29=2017）
      - 2015年10月 → NULL（日が不明なため）
      - NaN/空    → None
    """
    if pd.isna(date_str) or str(date_str).strip() == '' or str(date_str).strip().lower() == 'nan':
        return None
    s = str(date_str).strip()

    # yyyy/mm/dd 形式
    m = re.match(r'^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$', s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f'{y:04d}-{mo:02d}-{d:02d}'

    # 和暦（平成）: H28.10.20 / H29.10.19 など
    m = re.match(r'^[Hh](\d{1,2})[.\-/](\d{1,2})[.\-/](\d{1,2})$', s)
    if m:
        heisei_yr = int(m.group(1))
        seireki   = 1988 + heisei_yr  # H1=1989, H28=2016, H29=2017 etc
        mo, d = int(m.group(2)), int(m.group(3))
        return f'{seireki:04d}-{mo:02d}-{d:02d}'

    # mm月dd日 形式（年はheaderから）
    m = re.match(r'^(\d{1,2})月(\d{1,2})日?$', s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return f'{int(year):04d}-{mo:02d}-{d:02d}'

    # m/d 形式（年はheaderから）
    m = re.match(r'^(\d{1,2})/(\d{1,2})$', s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        return f'{int(year):04d}-{mo:02d}-{d:02d}'

    # yyyy年mm月 形式（日が不明→NULL）
    m = re.match(r'^(\d{4})年(\d{1,2})月$', s)
    if m:
        print(f'  [警告] 日付不明（年月のみ）: "{s}" → NULL')
        return None

    print(f'  [警告] 解析できない日付: "{s}" (年={year}) → NULL')
    return None


# ヘッダ行から年と列インデックスのマッピングを取得
# row 0: 2018, 2018, NaN, 2017, 2017, NaN, 2016, 2016, NaN, 2015, 2015
header = raw.iloc[0].tolist()
data   = raw.iloc[1:].reset_index(drop=True)

# 年が記載されている列インデックスを特定（値が 4桁数字）
year_col_indices = []
for i, val in enumerate(header):
    if pd.notna(val) and str(val).strip().isdigit() and len(str(val).strip()) == 4:
        year_col_indices.append((i, int(str(val).strip())))

# 連続した同じ年ペアは (place列, date列) のペアとして解釈
# 例: [0=2018, 1=2018, 3=2017, 4=2017, 6=2016, 7=2016, 9=2015, 10=2015]
# → (0,1,2018), (3,4,2017), (6,7,2016), (9,10,2015)
parsed_groups = []
i = 0
while i < len(year_col_indices):
    col_i, yr_i = year_col_indices[i]
    if i + 1 < len(year_col_indices):
        col_j, yr_j = year_col_indices[i + 1]
        if yr_i == yr_j and col_j == col_i + 1:
            # ペア確認
            parsed_groups.append((col_i, col_j, yr_i))
            i += 2
            continue
    # 単独列（念のため対処）
    parsed_groups.append((col_i, None, yr_i))
    i += 1

print(f'検出されたグループ（place列, date列, 年）: {parsed_groups}')

# 全レコードを (place_orig, year, date_str) に変換
# ※ place はCSVの元の値をそのまま保存（ゼロ除去はDB照合フェーズで行う）
records = []
for place_col, date_col, year in parsed_groups:
    for idx in range(len(data)):
        place_raw = data.iloc[idx, place_col]
        if pd.isna(place_raw) or str(place_raw).strip() == '':
            continue
        place_orig = str(place_raw).strip()   # 元の値（例: A01, A1）
        if date_col is not None:
            date_raw = data.iloc[idx, date_col]
        else:
            date_raw = None
        date_val = parse_date(date_raw, year)
        records.append({'place': place_orig, 'year': year, 'date': date_val})

date_df = pd.DataFrame(records).drop_duplicates(subset=['place', 'year'])
print(f'\n変換後レコード数: {len(date_df)}')
print(f'  日付あり: {date_df["date"].notna().sum()} 件')
print(f'  日付なし: {date_df["date"].isna().sum()} 件')
print('\n先頭20行:')
print(date_df[date_df['date'].notna()].head(20).to_string(index=False))

# =============================================================================
# 2. Solidテーブルへの適用
# =============================================================================
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# 既存の列を確認して date 列がなければ追加
existing_cols = [row[1] for row in cur.execute('PRAGMA table_info(Solid)').fetchall()]
print(f'\nSolid 既存列: {existing_cols}')

if 'date' not in existing_cols:
    cur.execute('ALTER TABLE Solid ADD COLUMN date TEXT')
    print('  → date 列を追加しました')
else:
    print('  → date 列は既に存在します（値を上書きします）')

# Solidテーブルの place × year の現状確認
solid_keys = pd.read_sql('SELECT DISTINCT place, year FROM Solid', conn)
solid_keys['place_norm'] = solid_keys['place'].apply(normalize_place)
print(f'\nSolid の (place, year) ユニーク数: {len(solid_keys)}')

# 突合してUPDATE（元のplace値 → ゼロ除去place値 の2段階で試みる）
updated = 0
not_matched = 0
for _, row in date_df.iterrows():
    if row['date'] is None:
        continue
    place_orig = row['place']                          # 例: A01（CSV元の値）
    place_norm = normalize_place(row['place'])         # 例: A1（ゼロ除去）
    year_val   = int(row['year'])

    # まず元の識別子でマッチ
    res = cur.execute(
        'UPDATE Solid SET date = ? WHERE place = ? AND year = ?',
        (row['date'], place_orig, year_val)
    )
    if res.rowcount > 0:
        updated += res.rowcount
        continue

    # 次にゼロ除去した識別子でマッチ
    if place_norm and place_norm != place_orig:
        res = cur.execute(
            'UPDATE Solid SET date = ? WHERE place = ? AND year = ?',
            (row['date'], place_norm, year_val)
        )
        if res.rowcount > 0:
            updated += res.rowcount
            continue

    not_matched += 1

conn.commit()
print(f'\nUPDATE 完了:')
print(f'  更新行数: {updated} 件')
print(f'  マッチしなかった日付データ: {not_matched} 件')

# 結果確認
result = pd.read_sql(
    "SELECT place, year, date FROM Solid WHERE date IS NOT NULL ORDER BY year, place LIMIT 30",
    conn
)
print('\n更新後の Solid (date列あり, 先頭30行):')
print(result.to_string(index=False))

# 年別更新数の確認
for yr in [2015, 2016, 2017, 2018]:
    n = pd.read_sql(
        f"SELECT COUNT(*) as n FROM Solid WHERE year={yr} AND date IS NOT NULL", conn
    ).iloc[0, 0]
    total = pd.read_sql(
        f"SELECT COUNT(*) as n FROM Solid WHERE year={yr}", conn
    ).iloc[0, 0]
    print(f'  {yr}年: {n}/{total} 件に日付あり')

conn.close()
print('\n完了。')
