"""
weather_database.db 品質チェックスクリプト
------------------------------------------
以下の観点でデータ取得の成否を調べます:
  1. 基本情報     : テーブル構造・行数・カラム一覧
  2. 地点カバレッジ: 何地点分のデータがあるか (FieldData.db と比較)
  3. 期間カバレッジ: 各地点×各変数の日付範囲と欠損率
  4. 変数ごとの統計: mean/std/min/max で外れ値を確認
  5. 欠損マップ   : 地点×変数の欠損率をクロス集計
  6. 期待行数チェック: グループA(1981〜今日) / グループB(2008〜今日) の期待日数と比較
"""

import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from datetime import date

# ── パス設定 ─────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WEATHER_DB = os.path.join(_ROOT, "data", "raw", "weather_database.db")
FIELD_DB   = os.path.join(_ROOT, "data", "raw", "FieldData.db")

GROUP_A = ["TMP_mea", "TMP_max", "TMP_min", "OPR", "SSD", "GSR", "SD", "SWE", "SFW", "APCP"]
GROUP_B = ["APCPRA", "DLR", "RH", "WIND"]
ALL_VARS = GROUP_A + GROUP_B

START_A = pd.Timestamp("1981-01-01")
START_B = pd.Timestamp("2008-01-01")
END_DATE = pd.Timestamp(date.today())

EXPECTED_DAYS_A = (END_DATE - START_A).days + 1
EXPECTED_DAYS_B = (END_DATE - START_B).days + 1

SEP = "=" * 70


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ── 1. 基本情報 ──────────────────────────────────────────────
section("1. 基本情報")
conn_w = sqlite3.connect(WEATHER_DB)

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn_w)
print("テーブル一覧:", tables["name"].tolist())

# weather_data テーブルの行数・列数
n_rows = pd.read_sql("SELECT COUNT(*) as cnt FROM weather_data", conn_w)["cnt"][0]
cols_df = pd.read_sql("PRAGMA table_info(weather_data)", conn_w)
col_names = cols_df["name"].tolist()
print(f"行数       : {n_rows:,}")
print(f"カラム数   : {len(col_names)}")
print(f"カラム名   : {col_names}")


# ── 2. 地点カバレッジ ────────────────────────────────────────
section("2. 地点カバレッジ")

# FieldData.db から期待地点数を取得
if os.path.exists(FIELD_DB):
    conn_f = sqlite3.connect(FIELD_DB)
    places_expected = pd.read_sql(
        "SELECT DISTINCT place, lat, lon FROM Questionaire WHERE lat IS NOT NULL AND lon IS NOT NULL",
        conn_f
    )
    conn_f.close()
    n_expected = len(places_expected)
else:
    n_expected = None
    print("FieldData.db が見つかりません（期待地点数不明）")

places_actual = pd.read_sql("SELECT DISTINCT place FROM weather_data", conn_w)
n_actual = len(places_actual)

print(f"期待地点数 (FieldData.db): {n_expected}")
print(f"実際の地点数 (weather_db): {n_actual}")
if n_expected:
    coverage = n_actual / n_expected * 100
    print(f"地点カバレッジ           : {coverage:.1f}%")
    if n_expected and n_actual < n_expected:
        missing_places = set(places_expected["place"]) - set(places_actual["place"])
        print(f"未取得の地点 ({len(missing_places)}件):")
        for p in sorted(missing_places):
            print(f"  - {p}")


# ── 3. 期間カバレッジ (地点ごと) ─────────────────────────────
section("3. 期間カバレッジ (地点ごとの日付範囲)")

date_range = pd.read_sql(
    "SELECT place, MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as n_days FROM weather_data GROUP BY place",
    conn_w
)
date_range["min_date"] = pd.to_datetime(date_range["min_date"])
date_range["max_date"] = pd.to_datetime(date_range["max_date"])
date_range["n_days"] = date_range["n_days"].astype(int)

# 期待日数との比較 (グループA の開始日を基準)
date_range["expected_days"] = EXPECTED_DAYS_A
date_range["coverage_pct"] = (date_range["n_days"] / date_range["expected_days"] * 100).round(1)

pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 120)
print(date_range.to_string(index=False))
print(f"\n平均カバレッジ: {date_range['coverage_pct'].mean():.1f}%")
print(f"最低カバレッジ: {date_range['coverage_pct'].min():.1f}%  ({date_range.loc[date_range['coverage_pct'].idxmin(), 'place']})")


# ── 4. 変数ごとの基本統計 ────────────────────────────────────
section("4. 変数ごとの基本統計 (mean / std / min / max / null率)")

# 存在する列のみ対象
existing_vars = [v for v in ALL_VARS if v in col_names]
missing_vars  = [v for v in ALL_VARS if v not in col_names]
if missing_vars:
    print(f"⚠ weather_data に存在しない変数: {missing_vars}")

# サンプリング（全行読み込みは重いため最大50万行）
df_sample = pd.read_sql(
    f"SELECT {', '.join(existing_vars)} FROM weather_data LIMIT 500000",
    conn_w
)

stats = df_sample[existing_vars].describe().T
stats["null_pct"] = (df_sample[existing_vars].isna().mean() * 100).round(2)
print(stats[["count", "mean", "std", "min", "max", "null_pct"]].to_string())


# ── 5. 欠損マップ (地点 × 変数) ──────────────────────────────
section("5. 欠損マップ: 地点 × 変数 の欠損率 (%)")

# 各変数の地点ごとのNULL率
null_by_place = pd.read_sql(
    f"""
    SELECT place,
           {', '.join([f'ROUND(SUM(CASE WHEN {v} IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS {v}' for v in existing_vars])}
    FROM weather_data
    GROUP BY place
    """,
    conn_w
)
print(null_by_place.to_string(index=False))


# ── 6. 期待行数チェック ──────────────────────────────────────
section("6. 期待行数チェック")

print(f"期待日数 (グループA, {START_A.date()}〜{END_DATE.date()}): {EXPECTED_DAYS_A:,} 日")
print(f"期待日数 (グループB, {START_B.date()}〜{END_DATE.date()}): {EXPECTED_DAYS_B:,} 日")
print(f"実際の総行数 (全地点合計)               : {n_rows:,}")
if n_actual > 0:
    avg_rows_per_place = n_rows / n_actual
    print(f"地点あたり平均行数                     : {avg_rows_per_place:,.0f}")
    print(f"グループA との差 (地点あたり)           : {avg_rows_per_place - EXPECTED_DAYS_A:+,.0f} 日")


# ── 終了 ─────────────────────────────────────────────────────
conn_w.close()
section("チェック完了")
print("問題がなければ全地点でカバレッジ 90%+ が期待されます。")
print("欠損が多い地点・変数に注目して再取得を検討してください。")
