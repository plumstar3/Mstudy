"""
fix_global_null_dates.py
=========================
2008-01-01 の APCPRA と 2008-12-21 の RH/WIND は
全611圃場でNULLとなっており、donor補完では埋められない。

前後日の線形補間（平均値）で該当圃場の残留1行NULLを解消する。
対象: field_id=39, 174, 215
"""

import sqlite3
import pandas as pd

WEATHER_DB  = 'data/processed/weather_database_fieldid.db'
TARGET_FIDS = [39, 174, 215]

# 全圃場でNULLの日付と列（データ仕様上の欠損）
NULL_DATES = {
    '2008-01-01': ['APCPRA'],
    '2008-12-21': ['RH', 'WIND'],
}


def interpolate_one(conn, fid: int, date: str, col: str) -> bool:
    """前後日の平均でNULLを補間。成功でTrue"""
    cur = conn.cursor()

    df_prev = pd.read_sql(
        f'SELECT {col} FROM weather_data WHERE field_id=? AND date<? AND {col} IS NOT NULL ORDER BY date DESC LIMIT 1',
        conn, params=(fid, date)
    )
    df_next = pd.read_sql(
        f'SELECT {col} FROM weather_data WHERE field_id=? AND date>? AND {col} IS NOT NULL ORDER BY date ASC LIMIT 1',
        conn, params=(fid, date)
    )

    prev_val = float(df_prev[col].iloc[0]) if not df_prev.empty else None
    next_val = float(df_next[col].iloc[0]) if not df_next.empty else None

    if prev_val is not None and next_val is not None:
        interp = (prev_val + next_val) / 2.0
        method = 'prev+next avg'
    elif prev_val is not None:
        interp = prev_val
        method = 'prev only'
    elif next_val is not None:
        interp = next_val
        method = 'next only'
    else:
        print(f'    !! {col}@{date}: 補間値なし → スキップ')
        return False

    cur.execute(
        f'UPDATE weather_data SET {col}=? WHERE field_id=? AND date=? AND {col} IS NULL',
        (interp, fid, date)
    )
    n = cur.rowcount
    conn.commit()
    print(f'    {col}@{date}: {interp:.4f}  [{method}]  {n}行UPDATE')
    return n > 0


def run():
    conn = sqlite3.connect(WEATHER_DB)
    print('=' * 60)
    print('  全圃場欠損日付 線形補間（残留1行NULLを解消）')
    print(f'  対象圃場: {TARGET_FIDS}')
    print('=' * 60)

    for fid in TARGET_FIDS:
        print(f'\nfield_id={fid}')
        for date, cols in NULL_DATES.items():
            for col in cols:
                interpolate_one(conn, fid, date, col)

    # 補完後確認
    print()
    print('=== 補完後 NULL 確認（2008年以降）===')
    all_ok = True
    check_cols = ['APCPRA', 'RH', 'WIND', 'DLR']
    for fid in TARGET_FIDS:
        exprs = ', '.join(
            [f'SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) as n_{c}' for c in check_cols]
        )
        df = pd.read_sql(
            f'SELECT {exprs} FROM weather_data WHERE field_id=? AND date>="2008-01-01"',
            conn, params=(fid,)
        )
        row = df.iloc[0]
        vals = {c: int(row[f'n_{c}']) for c in check_cols}
        ok = all(v == 0 for v in vals.values())
        if not ok:
            all_ok = False
        status = 'OK' if ok else 'NG'
        detail = '  '.join([f'{c}={v}' for c, v in vals.items()])
        print(f'  [{status}] field_id={fid}: {detail}')

    conn.close()

    if all_ok:
        print('\n全3圃場のNULLを解消しました。')
    else:
        print('\n残留NULLあり（全圃場共通の欠損日かデータ仕様）。')


if __name__ == '__main__':
    run()
