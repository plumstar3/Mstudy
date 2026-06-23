"""
impute_weather_null_fields.py  (高速版)
============================================================
weather_database_fieldid.db で TMP_max / TMP_min が全行 NULL に
なっている圃場を、空間的最近傍圃場の値で補完するスクリプト。

【高速化手法】
  pandas で donor データを一括取得 → DataFrame でマージ →
  一時テーブルに INSERT → UPDATE ... FROM 一時テーブル
  （executemany による行ごとの UPDATE を回避）
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

WEATHER_DB  = 'data/processed/weather_database_fieldid.db'
LOG_PATH    = 'outputs/gdd/weather_impute_log.csv'
MAX_DIST_KM = 150.0

PROBLEM_FIELDS = {
    611: ['TMP_max'],
    284: ['TMP_max', 'TMP_min'],
    127: ['TMP_max', 'TMP_min'],
    433: ['TMP_max', 'TMP_min'],
    575: ['TMP_min'],
    131: ['TMP_max', 'TMP_min'],
    510: ['TMP_max', 'TMP_min'],
    307: ['TMP_max', 'TMP_min'],
    386: ['TMP_min'],
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def get_complete_fields(conn, null_cols):
    cond = ' AND '.join(
        [f'SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) = 0' for c in null_cols]
    )
    return pd.read_sql(f'''
        SELECT field_id, lat, lon FROM weather_data
        GROUP BY field_id HAVING {cond}
    ''', conn)


def run():
    Path('outputs/gdd').mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(WEATHER_DB)
    log_rows = []

    print('=' * 65)
    print('  weather NULL 気温データ 空間的最近傍補完（高速版）')
    print(f'  最大探索距離: {MAX_DIST_KM} km')
    print('=' * 65)

    for fid, null_cols in PROBLEM_FIELDS.items():
        print(f'\n── field_id={fid}  欠損列={null_cols} ──')

        # 対象圃場の lat/lon
        df_pos = pd.read_sql(
            'SELECT DISTINCT lat, lon FROM weather_data WHERE field_id = ?',
            conn, params=(fid,)
        ).dropna()
        if df_pos.empty:
            print('  !! lat/lon 不明 → スキップ')
            log_rows.append({'field_id': fid, 'null_cols': str(null_cols),
                             'donor_fid': None, 'dist_km': None, 'status': 'skip_no_geo'})
            continue
        lat, lon = float(df_pos['lat'].iloc[0]), float(df_pos['lon'].iloc[0])
        print(f'  lat={lat:.6f}  lon={lon:.6f}')

        # donor 候補取得
        df_donors = get_complete_fields(conn, null_cols)
        df_donors = df_donors[df_donors['field_id'] != fid].copy()
        if df_donors.empty:
            print('  !! donor 圃場なし → スキップ')
            log_rows.append({'field_id': fid, 'null_cols': str(null_cols),
                             'donor_fid': None, 'dist_km': None, 'status': 'skip_no_donor'})
            continue

        # 最近傍選択
        dists = haversine_km(lat, lon,
                              df_donors['lat'].to_numpy(),
                              df_donors['lon'].to_numpy())
        df_donors['dist_km'] = dists
        df_donors = df_donors.sort_values('dist_km')
        within = df_donors[df_donors['dist_km'] <= MAX_DIST_KM]
        best   = (within if not within.empty else df_donors).iloc[0]
        note   = 'within_150km' if not within.empty else 'nearest_fallback'

        donor_fid = int(best['field_id'])
        dist_km   = float(best['dist_km'])
        print(f'  → donor: field_id={donor_fid}  距離={dist_km:.1f} km  ({note})')

        # ── 高速補完: pandas マージ → to_sql 一時テーブル → UPDATE ──────────
        print(f'  donor データ取得中...')
        col_list = ', '.join(null_cols)
        df_donor_vals = pd.read_sql(
            f'SELECT date, {col_list} FROM weather_data WHERE field_id = ?',
            conn, params=(donor_fid,)
        )
        # donor 列をリネーム
        rename_map = {c: f'_d_{c}' for c in null_cols}
        df_donor_vals = df_donor_vals.rename(columns=rename_map)

        # 一時テーブルに書き込み（pandas to_sql は高速）
        tmp_table = f'_tmp_donor_{fid}'
        df_donor_vals.to_sql(tmp_table, conn, if_exists='replace', index=False)
        print(f'  一時テーブル "{tmp_table}" 作成: {len(df_donor_vals):,} 行')

        # 各列を UPDATE（SQLite の UPDATE ... FROM 構文）
        cursor = conn.cursor()
        total_updated = 0
        for col in null_cols:
            dcol = f'_d_{col}'
            sql = f'''
                UPDATE weather_data
                SET {col} = (
                    SELECT {dcol} FROM {tmp_table}
                    WHERE {tmp_table}.date = weather_data.date
                )
                WHERE field_id = {fid}
                  AND {col} IS NULL
                  AND EXISTS (
                    SELECT 1 FROM {tmp_table}
                    WHERE {tmp_table}.date = weather_data.date
                      AND {tmp_table}.{dcol} IS NOT NULL
                  )
            '''
            cursor.execute(sql)
            n = cursor.rowcount
            total_updated += n
            print(f'    {col}: {n:,} 行 UPDATE')

        conn.commit()

        # 一時テーブル削除
        cursor.execute(f'DROP TABLE IF EXISTS {tmp_table}')
        conn.commit()
        print(f'  完了: 合計 {total_updated:,} 行更新')

        log_rows.append({
            'field_id':     fid,
            'null_cols':    str(null_cols),
            'donor_fid':    donor_fid,
            'dist_km':      round(dist_km, 2),
            'updated_rows': total_updated,
            'status':       note,
        })

    conn.close()

    # ── ログ保存 ─────────────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False, encoding='utf-8-sig')
    print(f'\n補完ログ: {LOG_PATH}')
    print(log_df.to_string(index=False))

    # ── 補完後確認 ────────────────────────────────────────────────────────────
    print('\n=== 補完後 NULL 確認 ===')
    conn2 = sqlite3.connect(WEATHER_DB)
    all_ok = True
    for fid in PROBLEM_FIELDS:
        df = pd.read_sql('''
            SELECT SUM(CASE WHEN TMP_max IS NULL THEN 1 ELSE 0 END) as null_tmax,
                   SUM(CASE WHEN TMP_min IS NULL THEN 1 ELSE 0 END) as null_tmin
            FROM weather_data WHERE field_id = ?
        ''', conn2, params=(fid,))
        r = df.iloc[0]
        n_tmax, n_tmin = int(r.null_tmax), int(r.null_tmin)
        ok = 'OK' if n_tmax == 0 and n_tmin == 0 else 'NG'
        if ok == 'NG':
            all_ok = False
        print(f'  [{ok}] field_id={fid:3d}: null_tmax={n_tmax}  null_tmin={n_tmin}')
    conn2.close()

    if all_ok:
        print('\n全圃場の補完が完了しました。')
    else:
        print('\n一部の圃場で NULL が残っています。ログを確認してください。')


if __name__ == '__main__':
    run()
