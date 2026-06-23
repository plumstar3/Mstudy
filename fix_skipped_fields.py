"""
fix_skipped_fields.py
======================
impute_weather_all_columns.py 実行時にdonor不在でスキップされた
field_id=39, 174, 215 の APCPRA/RH/WIND/DLR を再補完する。

donor条件: 2008年以降で対象列のNULL率 < 1% の最近傍圃場
（全圃場が2008年以降に1行だけNULLを持つため、= 0 の厳密条件では
  donorが存在しなかった。緩和条件で再実行する）
"""
import sqlite3
import pandas as pd
import sys

sys.path.insert(0, '.')
from impute_weather_all_columns import (
    find_donor, update_from_donor,
    WEATHER_DB, MAX_DIST_KM, GPU_AVAILABLE,
)

SKIP_FIDS = {
    39:  ['APCPRA', 'RH', 'DLR'],
    174: ['APCPRA', 'RH', 'WIND'],
    215: ['RH', 'WIND', 'DLR'],
}

def run():
    conn = sqlite3.connect(WEATHER_DB)
    print('=' * 60)
    print('  スキップ3圃場 再補完（NULL率 < 1% donor 条件）')
    print(f'  GPU: {GPU_AVAILABLE}')
    print('=' * 60)

    for fid, null_cols in SKIP_FIDS.items():
        print(f'\nfield_id={fid}  欠損列={null_cols}')

        # lat/lon 取得
        df_pos = pd.read_sql(
            'SELECT AVG(lat) as lat, AVG(lon) as lon FROM weather_data WHERE field_id=?',
            conn, params=(fid,)
        )
        lat, lon = float(df_pos.lat.iloc[0]), float(df_pos.lon.iloc[0])
        print(f'  lat={lat:.6f}  lon={lon:.6f}')

        # donor 検索（period_limited=True → 2008以降で判定）
        donor_fid, dist_km, note = find_donor(
            conn, fid, lat, lon, null_cols, MAX_DIST_KM, period_limited=True
        )

        if donor_fid is None:
            print(f'  !! donor なし → スキップ（手動対応が必要）')
            continue

        print(f'  → donor: field_id={donor_fid}  距離={dist_km:.3f} km  ({note})')

        # UPDATE 実行
        updated = update_from_donor(
            conn, fid, donor_fid, null_cols, period_limited=True
        )
        for col, n in updated.items():
            print(f'    {col}: {n:,} 行 UPDATE')
        print(f'  合計: {sum(updated.values()):,} 行')

    # ── 補完後 NULL 確認 ─────────────────────────────────────────────────
    print()
    print('=== 補完後 NULL 確認（2008年以降）===')
    all_ok = True
    for fid, null_cols in SKIP_FIDS.items():
        exprs = ', '.join(
            [f'SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) as n_{c}'
             for c in null_cols]
        )
        df = pd.read_sql(
            f'SELECT {exprs} FROM weather_data WHERE field_id=? AND date>="2008-01-01"',
            conn, params=(fid,)
        )
        row = df.iloc[0]
        vals = {c: int(row[f'n_{c}']) for c in null_cols}
        ok = all(v == 0 for v in vals.values())
        if not ok:
            all_ok = False
        status = 'OK' if ok else 'NG'
        detail = '  '.join([f'{c}={v}' for c, v in vals.items()])
        print(f'  [{status}] field_id={fid}: {detail}')

    conn.close()

    if all_ok:
        print('\n全3圃場の補完が完了しました。')
    else:
        print('\n一部 NULL が残っています。')


if __name__ == '__main__':
    run()
