"""
data/raw/FieldData.db に field_id 列だけを追加した DB を作る。

data/processed/FieldData_fieldid.db には field_id のほかに開花日推定・播種日/収穫日補完・
栽植間隔修正などが入っているため、それらを除いた「元データ + field_id」の版を作成する。
field_id は FieldData_fieldid.db の (place, year) -> field_id 対応から付与する
（対応の無い圃場×年は NULL のまま）。field_master テーブルも識別用にコピーする。

出力: data/processed/FieldData_fieldid_raw.db
リポジトリルートから実行:
    C:\\Users\\amilu\\anaconda3\\python build_fielddata_fieldid_raw.py
"""
import os
import shutil
import sqlite3

RAW_DB = "data/raw/FieldData.db"
FID_DB = "data/processed/FieldData_fieldid.db"
OUT_DB = "data/processed/FieldData_fieldid_raw.db"


def main():
    if os.path.exists(OUT_DB):
        raise SystemExit(f"{OUT_DB} は既に存在します。上書きする場合は削除してから実行してください。")

    shutil.copyfile(RAW_DB, OUT_DB)
    con = sqlite3.connect(OUT_DB)
    con.execute("ATTACH DATABASE ? AS fid", (FID_DB,))

    # 全テーブルから (place, year) -> field_id の対応表を作り、一意性を確認する
    tables = [r[0] for r in con.execute(
        "SELECT name FROM main.sqlite_master WHERE type='table' ORDER BY name")]
    union = " UNION ".join(
        f"SELECT DISTINCT place, year, field_id FROM fid.{t} WHERE field_id IS NOT NULL" for t in tables)
    con.execute(f"CREATE TEMP TABLE fid_map AS {union}")
    dup = con.execute(
        "SELECT COUNT(*) FROM (SELECT place, year FROM fid_map GROUP BY place, year HAVING COUNT(DISTINCT field_id) > 1)"
    ).fetchone()[0]
    if dup:
        raise SystemExit(f"(place, year) に複数の field_id が対応する組が {dup} 件あります")
    con.execute("CREATE UNIQUE INDEX temp.fid_map_idx ON fid_map(place, year)")

    for t in tables:
        con.execute(f"ALTER TABLE main.{t} ADD COLUMN field_id INTEGER")
        con.execute(f"""
            UPDATE main.{t} SET field_id = (
                SELECT m.field_id FROM fid_map m WHERE m.place = main.{t}.place AND m.year = main.{t}.year)
        """)
        n, n_null = con.execute(
            f"SELECT COUNT(*), SUM(field_id IS NULL) FROM main.{t}").fetchone()
        print(f"{t:15s} rows={n:7d}  field_id NULL={n_null}")

    sql =con.execute("SELECT sql FROM fid.sqlite_master WHERE name='field_master'").fetchone()[0]
    con.execute(sql)
    con.execute("INSERT INTO main.field_master SELECT * FROM fid.field_master")
    print(f"field_master    rows={con.execute('SELECT COUNT(*) FROM main.field_master').fetchone()[0]}")

    con.commit()
    con.execute("DETACH DATABASE fid")
    con.execute("VACUUM")
    con.close()
    print(f"-> {OUT_DB}")


if __name__ == "__main__":
    main()
