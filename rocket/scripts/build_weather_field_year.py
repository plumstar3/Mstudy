"""圃場×年ごとの日別気象データセットを作る。

rocket/data/raw/FieldData_fieldid.db の Questionaire から (field_id, year) を読み、
rocket/data/raw/weather_database_fieldid.db の weather_data から、同じ field_id で
date の年が year と一致する行(その年の1/1〜12/31)を取り出して1つの CSV にまとめる。

出力: rocket/data/processed/weather_field_year.csv(圃場×年×日、utf-8-sig)
DB は読み取り専用で開く。
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROCKET_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROCKET_DIR / "data" / "raw"
OUT_DIR = ROCKET_DIR / "data" / "processed"
FIELD_DB = RAW_DIR / "FieldData_fieldid.db"
WEATHER_DB = RAW_DIR / "weather_database_fieldid.db"
OUT_CSV = OUT_DIR / "weather_field_year.csv"


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def main() -> None:
    with connect_ro(FIELD_DB) as con:
        q = pd.read_sql("SELECT field_id, year FROM Questionaire", con)

    n_null = q["field_id"].isna().sum()
    if n_null:
        print(f"field_id が空の行を除外: {n_null} 件")
    pairs = q.dropna(subset=["field_id"]).astype({"field_id": int, "year": int})
    pairs = pairs.drop_duplicates().sort_values(["field_id", "year"])
    print(f"(field_id, year) の組: {len(pairs)} 件")

    frames = []
    with connect_ro(WEATHER_DB) as con:
        for year, g in pairs.groupby("year"):
            ids = g["field_id"].tolist()
            ph = ",".join("?" * len(ids))
            sql = (
                f"SELECT * FROM weather_data "
                f"WHERE date BETWEEN ? AND ? AND field_id IN ({ph})"
            )
            w = pd.read_sql(sql, con, params=[f"{year}-01-01", f"{year}-12-31", *ids])
            w.insert(1, "year", year)
            frames.append(w)
            print(f"  {year}: 圃場 {len(ids)} 件 → {len(w)} 行")

    df = pd.concat(frames, ignore_index=True)
    df["field_id"] = df["field_id"].astype(int)
    lead = ["field_id", "year", "date", "place", "lat", "lon"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df = df.sort_values(["field_id", "year", "date"]).reset_index(drop=True)

    # 気象が見つからなかった組と日数の確認
    days = df.groupby(["field_id", "year"]).size().rename("n_days").reset_index()
    chk = pairs.merge(days, how="left", on=["field_id", "year"])
    missing = chk[chk["n_days"].isna()]
    if len(missing):
        print(f"気象が見つからない組: {len(missing)} 件")
        print(missing.to_string(index=False))
    print("1組あたりの日数:", chk["n_days"].value_counts().to_dict())
    print("列ごとの欠損率:")
    print(df.isna().mean().round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"保存: {OUT_CSV} ({len(df)} 行, {df.shape[1]} 列)")


if __name__ == "__main__":
    main()
