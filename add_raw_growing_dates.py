"""
FieldData_fieldid.db の Questionaire に、補完前の播種日・収穫日の列を追加する。

impute_growing_dates.py / fix_field5_harvest.py で seed_date・harvest_date の欠損が
近傍圃場から補完されているため、元の data/raw/FieldData.db の値を
seed_date_raw / harvest_date_raw として別列で持たせる（既存列は変更しない）。

例外: A4・2016年（field_id=5）は元データの harvest_date=2016-07-25 が開花日の誤入力で、
fix_field5_harvest.py で flower_date に移しているため、harvest_date_raw は NULL とする。

実行前に FieldData_fieldid.db.bak_before_rawdates を作成する。
リポジトリルートから実行:
    C:\\Users\\amilu\\anaconda3\\python add_raw_growing_dates.py
"""
import os
import shutil
import sqlite3

RAW_DB = "data/raw/FieldData.db"
FIELD_DB = "data/processed/FieldData_fieldid.db"
BACKUP = FIELD_DB + ".bak_before_rawdates"

# 元データの harvest_date が開花日の誤入力だった圃場×年
MISENTERED_HARVEST = [("A4", 2016)]


def main():
    con = sqlite3.connect(FIELD_DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(Questionaire)")]
    if "seed_date_raw" in cols or "harvest_date_raw" in cols:
        raise SystemExit("seed_date_raw / harvest_date_raw は既に存在します")
    con.close()

    if not os.path.exists(BACKUP):
        shutil.copyfile(FIELD_DB, BACKUP)
        print(f"バックアップ: {BACKUP}")

    con = sqlite3.connect(FIELD_DB)
    con.execute("ATTACH DATABASE ? AS raw", (RAW_DB,))
    con.execute("ALTER TABLE Questionaire ADD COLUMN seed_date_raw TEXT")
    con.execute("ALTER TABLE Questionaire ADD COLUMN harvest_date_raw TEXT")
    con.execute("""
        UPDATE Questionaire SET
            seed_date_raw = (SELECT r.seed_date FROM raw.Questionaire r
                             WHERE r.place = Questionaire.place AND r.year = Questionaire.year),
            harvest_date_raw = (SELECT r.harvest_date FROM raw.Questionaire r
                                WHERE r.place = Questionaire.place AND r.year = Questionaire.year)
    """)
    for place, year in MISENTERED_HARVEST:
        con.execute("UPDATE Questionaire SET harvest_date_raw = NULL WHERE place = ? AND year = ?",
                    (place, year))
    con.commit()

    n, s_null, h_null, s_imp, h_imp = con.execute("""
        SELECT COUNT(*), SUM(seed_date_raw IS NULL), SUM(harvest_date_raw IS NULL),
               SUM(seed_date_raw IS NULL AND seed_date IS NOT NULL),
               SUM(harvest_date_raw IS NULL AND harvest_date IS NOT NULL)
        FROM Questionaire
    """).fetchone()
    print(f"rows={n}  seed_date_raw NULL={s_null} (うち補完済み {s_imp})  "
          f"harvest_date_raw NULL={h_null} (うち補完済み {h_imp})")
    con.close()


if __name__ == "__main__":
    main()
