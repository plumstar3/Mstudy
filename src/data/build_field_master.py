"""
build_field_master.py
=====================
FieldData.db と weather_database.db に field_id を付与するスクリプト。

処理内容:
  1. FieldData.db  → data/processed/FieldData_fieldid.db  にコピー
  2. field_master テーブルを作成し、(place, lat, lon) ごとに整数 field_id を採番
  3. FieldData.db 内の全8テーブルに field_id 列を追加・UPDATE
  4. weather_database.db → data/processed/weather_database_fieldid.db にコピー
  5. weather_data テーブルに field_id 列を追加・UPDATE

元ファイル (data/raw/) は一切変更しません。
"""
from __future__ import annotations
import os
import sys
import shutil
import sqlite3
import pandas as pd

# ── パス設定 ──────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR       = os.path.join(_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SRC_FIELD_DB   = os.path.join(RAW_DIR,       "FieldData.db")
SRC_WEATHER_DB = os.path.join(RAW_DIR,       "weather_database.db")
DST_FIELD_DB   = os.path.join(PROCESSED_DIR, "FieldData_fieldid.db")
DST_WEATHER_DB = os.path.join(PROCESSED_DIR, "weather_database_fieldid.db")

# FieldData.db 内の全テーブル（field_id を付与する対象）
FIELD_TABLES = [
    "Questionaire",
    "Harm",
    "Nutrition",
    "Solid",
    "SolidMoisture",
    "Refinement",
    "NormalWeather",
    "Weather",
]

SEP = "=" * 60


def section(msg: str) -> None:
    print(f"\n{SEP}\n  {msg}\n{SEP}")


# ── Step 0: ファイルのコピー ──────────────────────────────────────────────────
section("Step 0: ファイルコピー")

for src, dst in [(SRC_FIELD_DB, DST_FIELD_DB), (SRC_WEATHER_DB, DST_WEATHER_DB)]:
    if not os.path.exists(src):
        print(f"[ERROR] ソースファイルが見つかりません: {src}")
        sys.exit(1)
    print(f"  コピー: {os.path.basename(src)} → {os.path.relpath(dst, _ROOT)}")
    shutil.copy2(src, dst)

print("コピー完了。")


# ── Step 1: field_master の作成 ───────────────────────────────────────────────
section("Step 1: field_master テーブルの作成")

conn_f = sqlite3.connect(DST_FIELD_DB)
conn_f.execute("PRAGMA journal_mode=WAL")

# Questionaire から (place, lat, lon) のユニーク組み合わせを取得
# place のアルファベット順、同じ place 内では lat/lon 昇順で採番
master_df = pd.read_sql(
    """
    SELECT DISTINCT place,
           ROUND(lat, 6) AS lat,
           ROUND(lon, 6) AS lon
    FROM Questionaire
    WHERE lat IS NOT NULL AND lon IS NOT NULL
    ORDER BY place, lat, lon
    """,
    conn_f,
)
master_df.insert(0, "field_id", range(1, len(master_df) + 1))
print(f"  field_master: {len(master_df)} 件")

# 既存の field_master を削除して再作成
conn_f.execute("DROP TABLE IF EXISTS field_master")
conn_f.execute(
    """
    CREATE TABLE field_master (
        field_id  INTEGER PRIMARY KEY,
        place     TEXT    NOT NULL,
        lat       REAL    NOT NULL,
        lon       REAL    NOT NULL
    )
    """
)
master_df.to_sql("field_master", conn_f, if_exists="append", index=False)
conn_f.commit()
print("  field_master テーブル作成完了。")


# ── Step 2: FieldData_fieldid.db の各テーブルに field_id を追加 ───────────────
section("Step 2: FieldData_fieldid.db 各テーブルへ field_id 追加")

for table in FIELD_TABLES:
    # テーブルが存在するか確認
    exists = conn_f.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
    ).fetchone()
    if not exists:
        print(f"  [{table}] テーブルが存在しないためスキップ")
        continue

    # 既に field_id 列があれば DROP して再追加（べき等性のため）
    cols = [row[1] for row in conn_f.execute(f"PRAGMA table_info({table})").fetchall()]
    if "field_id" in cols:
        # SQLite は列削除をサポートしないため、テーブル再作成でリセット
        # ただし今回は上書き実行のため、単純に UPDATE し直す
        print(f"  [{table}] field_id 列が既に存在 → 値を上書き更新します")
    else:
        conn_f.execute(f"ALTER TABLE {table} ADD COLUMN field_id INTEGER")
        conn_f.commit()
        print(f"  [{table}] field_id 列を追加")

    # field_id の UPDATE
    if table == "Questionaire":
        # Questionaire は (place, lat, lon) から直接照合
        conn_f.execute(
            f"""
            UPDATE {table}
            SET field_id = (
                SELECT fm.field_id
                FROM field_master fm
                WHERE fm.place = {table}.place
                  AND fm.lat   = ROUND({table}.lat, 6)
                  AND fm.lon   = ROUND({table}.lon, 6)
            )
            """
        )
    else:
        # 他テーブルは (place, year) → Questionaire → field_id
        conn_f.execute(
            f"""
            UPDATE {table}
            SET field_id = (
                SELECT q.field_id
                FROM Questionaire q
                WHERE q.place = {table}.place
                  AND q.year  = {table}.year
                LIMIT 1
            )
            """
        )

    conn_f.commit()

    # NULL 残存チェック
    null_cnt = conn_f.execute(
        f"SELECT COUNT(*) FROM {table} WHERE field_id IS NULL"
    ).fetchone()[0]
    total    = conn_f.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    status   = "OK" if null_cnt == 0 else f"WARNING: NULL {null_cnt} rows remaining"
    print(f"  [{table}] 総行数={total:,}  field_id NULL={null_cnt}  {status}")

conn_f.close()


# ── Step 3: weather_database_fieldid.db に field_id を追加 ───────────────────
section("Step 3: weather_database_fieldid.db に field_id 追加")

conn_w = sqlite3.connect(DST_WEATHER_DB)
conn_w.execute("PRAGMA journal_mode=WAL")

# field_master を weather DB に ATTACH して照合
conn_w.execute(f"ATTACH DATABASE '{DST_FIELD_DB}' AS fdb")

cols_w = [row[1] for row in conn_w.execute("PRAGMA table_info(weather_data)").fetchall()]
if "field_id" in cols_w:
    print("  field_id 列が既に存在 → 値を上書き更新します")
else:
    conn_w.execute("ALTER TABLE weather_data ADD COLUMN field_id INTEGER")
    conn_w.commit()
    print("  field_id 列を追加")

print("  weather_data を更新中（約1013万行）... しばらくお待ちください")
conn_w.execute(
    """
    UPDATE weather_data
    SET field_id = (
        SELECT fm.field_id
        FROM fdb.field_master fm
        WHERE fm.place = weather_data.place
          AND fm.lat   = ROUND(weather_data.lat, 6)
          AND fm.lon   = ROUND(weather_data.lon, 6)
    )
    """
)
conn_w.commit()

null_cnt = conn_w.execute(
    "SELECT COUNT(*) FROM weather_data WHERE field_id IS NULL"
).fetchone()[0]
total    = conn_w.execute("SELECT COUNT(*) FROM weather_data").fetchone()[0]
# field_idがNULLとなるのは、field_masterに存在しない場所（緯度経度の不一致等）
status   = "OK" if null_cnt == 0 else f"WARNING: NULL {null_cnt} rows remaining"
print(f"  [weather_data] total={total:,}  field_id NULL={null_cnt}  {status}")

conn_w.execute("DETACH DATABASE fdb")
conn_w.close()


# ── 完了 ──────────────────────────────────────────────────────────────────────
section("完了")
print(f"  出力先:")
print(f"    {os.path.relpath(DST_FIELD_DB,   _ROOT)}")
print(f"    {os.path.relpath(DST_WEATHER_DB, _ROOT)}")
print("\n整備後の JOIN 例:")
print("""
  SELECT q.field_id, q.year, q.yield, h.drought, w.date, w.TMP_mea
  FROM   Questionaire q
  JOIN   Harm h         ON q.field_id = h.field_id
  JOIN   weather_data w ON q.field_id = w.field_id
  WHERE  w.date BETWEEN q.seed_date AND q.harvest_date
""")
