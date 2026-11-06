"""
rocket/data/raw/FieldData_fieldid.db の Questionaire に date_flag 列を追加し、
播種日・開花日・収穫日に要確認の点がある圃場×年にフラグを付ける。値そのものは変更しない。
経緯は rocket/docs/data_issues.md。

フラグ(複数該当するときは ';' でつなぐ。該当なしは NULL):
  harvest_was_flower  元データの harvest_date(2016-07-25)が開花日の誤入力。
                      flower_date に移し、harvest_date は近傍補完、harvest_date_raw は NULL(対応済み)
  harvest_next_year   収穫日が翌年 1/5(播種→収穫 164日)
  long_season         播種→収穫 193日・開花→収穫 121日(同品種・同播種日の D3/D4 は 11/05 収穫)
  long_seed_to_flower 播種→開花 72〜75日(全体の外れ値。品種特性の可能性あり)

実行前に FieldData_fieldid.db.bak_before_date_flag を作成する。
    rocket\\venv_rocket\\Scripts\\python rocket\\scripts\\add_date_flags.py
"""
import os
import shutil
import sqlite3

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD_DB = os.path.join(ROCKET_DIR, 'data', 'raw', 'FieldData_fieldid.db')
BACKUP = FIELD_DB + '.bak_before_date_flag'

FLAGS = {
    ('A4', 2016): ['harvest_was_flower'],
    ('O7', 2016): ['harvest_next_year'],
    ('O8', 2016): ['harvest_next_year'],
    ('D1', 2015): ['long_season', 'long_seed_to_flower'],
    ('D2', 2015): ['long_season', 'long_seed_to_flower'],
    ('D3', 2015): ['long_seed_to_flower'],
    ('D4', 2015): ['long_seed_to_flower'],
}


def main():
    con = sqlite3.connect(FIELD_DB)
    cols = [r[1] for r in con.execute('PRAGMA table_info(Questionaire)')]
    if 'date_flag' in cols:
        raise SystemExit('date_flag は既に存在します')
    con.close()

    if not os.path.exists(BACKUP):
        shutil.copyfile(FIELD_DB, BACKUP)
        print(f'バックアップ: {BACKUP}')

    con = sqlite3.connect(FIELD_DB)
    con.execute('ALTER TABLE Questionaire ADD COLUMN date_flag TEXT')
    for (place, year), flags in FLAGS.items():
        cur = con.execute('UPDATE Questionaire SET date_flag = ? WHERE place = ? AND year = ?',
                          (';'.join(flags), place, year))
        if cur.rowcount != 1:
            con.rollback()
            raise SystemExit(f'{place} {year}: 更新行数 {cur.rowcount}(1行のはず)')
    con.commit()

    for row in con.execute('''SELECT place, year, field_id, seed_date_raw, flower_date, harvest_date_raw, date_flag
                              FROM Questionaire WHERE date_flag IS NOT NULL ORDER BY year, place'''):
        print(row)
    con.close()


if __name__ == '__main__':
    main()
