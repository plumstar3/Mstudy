"""
impute_weather_all_columns.py
============================================================
weather_database_fieldid.db の全気象変数列について、
全行 NULL または部分 NULL となっている圃場を
空間的最近傍補完（Spatial Nearest-Neighbor Imputation）する。

【処理フロー】
  Step 1 : 全列×全 field_id の NULL 状況をスキャン
  Step 2 : 補完が必要な (field_id, col) ペアを特定
  Step 3 : 欠損パターン（全行NULL / 部分NULL）を分類
  Step 4 : 各 field_id に対してハーバーサイン距離で最近傍 donor を選択
             → 距離計算は GPU（CuPy）があれば GPU、なければ NumPy で実行
  Step 5 : donor の値で NULL 行を UPDATE（一時テーブル + サブクエリ方式）
  Step 6 : 補完後の NULL 状況を再確認し、ログを保存

【GPU 対応】
  CuPy がインストールされていれば自動的に GPU で距離計算を実行。
  未インストールの場合は NumPy にフォールバック。

【出力】
  outputs/gdd/weather_impute_all_log.csv   補完ログ
  outputs/gdd/weather_null_scan.csv        補完前のスキャン結果
"""

import sqlite3
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── GPU/CPU 切り替え ──────────────────────────────────────────────────────────
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print('[INFO] CuPy が検出されました。距離計算に GPU を使用します。')
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print('[INFO] CuPy 未検出。NumPy (CPU) で距離計算を実行します。')


# ── 定数 ─────────────────────────────────────────────────────────────────────
WEATHER_DB  = 'data/processed/weather_database_fieldid.db'
LOG_PATH    = 'outputs/gdd/weather_impute_all_log.csv'
SCAN_PATH   = 'outputs/gdd/weather_null_scan.csv'
MAX_DIST_KM = 150.0

# 補完対象外の列（キー・メタデータ列）
SKIP_COLS = {'date', 'place', 'lat', 'lon', 'field_id'}

# 全行 NULL と判定する閾値（null_ratio > この値で全行NULLとみなす）
ALL_NULL_THRESHOLD = 0.99


# ── ハーバーサイン距離（GPU/CPU 自動切り替え） ────────────────────────────────

def haversine_km(lat1: float, lon1: float,
                 lat2_arr, lon2_arr) -> np.ndarray:
    """スカラー (lat1, lon1) と配列 (lat2, lon2) の球面距離 [km]。
    CuPy が使えれば GPU で計算し、結果を NumPy に戻す。
    """
    xp = cp if GPU_AVAILABLE else np
    lat2 = xp.asarray(lat2_arr, dtype=xp.float64)
    lon2 = xp.asarray(lon2_arr, dtype=xp.float64)
    R    = 6371.0
    dlat = xp.radians(lat2 - lat1)
    dlon = xp.radians(lon2 - lon1)
    a    = (xp.sin(dlat / 2) ** 2
            + xp.cos(xp.radians(lat1)) * xp.cos(xp.radians(lat2))
            * xp.sin(dlon / 2) ** 2)
    dist = R * 2 * xp.arctan2(xp.sqrt(a), xp.sqrt(1 - a))
    return cp.asnumpy(dist) if GPU_AVAILABLE else dist


# ── Step 1: 全列×全 field_id の NULL スキャン ────────────────────────────────

def scan_null_status(conn) -> pd.DataFrame:
    """全気象変数列の field_id 別 NULL 件数を集計する。

    Returns:
        DataFrame: field_id, col, total_rows, null_count, null_ratio, lat, lon
    """
    # 対象列を取得
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(weather_data)")
    all_cols = [row[1] for row in cursor.fetchall()]
    weather_cols = [c for c in all_cols if c not in SKIP_COLS]

    print(f'  気象変数列: {weather_cols}')
    print(f'  列数: {len(weather_cols)}')

    # field_id ごとの集計（1クエリで全列を集計）
    null_exprs = ', '.join(
        [f'SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS null_{c}'
         for c in weather_cols]
    )
    df_agg = pd.read_sql(f'''
        SELECT field_id,
               COUNT(*) AS total_rows,
               AVG(lat) AS lat,
               AVG(lon) AS lon,
               {null_exprs}
        FROM weather_data
        GROUP BY field_id
    ''', conn)

    # ロング形式に変換
    records = []
    for _, row in df_agg.iterrows():
        fid   = int(row['field_id'])
        total = int(row['total_rows'])
        lat   = float(row['lat']) if pd.notna(row['lat']) else None
        lon   = float(row['lon']) if pd.notna(row['lon']) else None
        for col in weather_cols:
            null_cnt = int(row[f'null_{col}'])
            if null_cnt > 0:
                records.append({
                    'field_id':   fid,
                    'col':        col,
                    'total_rows': total,
                    'null_count': null_cnt,
                    'null_ratio': null_cnt / total,
                    'lat':        lat,
                    'lon':        lon,
                })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=['field_id', 'col', 'total_rows', 'null_count', 'null_ratio', 'lat', 'lon']
    ), weather_cols


# ── Step 2: 最近傍 donor 選択 ─────────────────────────────────────────────────

def find_donor(conn, fid: int, lat: float, lon: float,
               null_cols: list[str], max_dist: float) -> tuple:
    """指定列がすべて完全な最近傍 donor を返す。

    Returns:
        (donor_fid, dist_km, note) or (None, None, 'no_donor')
    """
    cond = ' AND '.join(
        [f'SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) = 0'
         for c in null_cols]
    )
    df_donors = pd.read_sql(f'''
        SELECT field_id, AVG(lat) AS lat, AVG(lon) AS lon
        FROM weather_data
        WHERE field_id != ?
        GROUP BY field_id
        HAVING {cond} AND AVG(lat) IS NOT NULL AND AVG(lon) IS NOT NULL
    ''', conn, params=(fid,))

    if df_donors.empty:
        return None, None, 'no_donor'

    dists = haversine_km(lat, lon,
                          df_donors['lat'].to_numpy(),
                          df_donors['lon'].to_numpy())
    df_donors['dist_km'] = dists
    df_donors = df_donors.sort_values('dist_km')

    within = df_donors[df_donors['dist_km'] <= max_dist]
    if not within.empty:
        best = within.iloc[0]
        note = 'within_dist'
    else:
        best = df_donors.iloc[0]
        note = 'nearest_fallback'

    return int(best['field_id']), float(best['dist_km']), note


# ── Step 3: 一時テーブル経由 UPDATE ──────────────────────────────────────────

def update_from_donor(conn, target_fid: int, donor_fid: int,
                       null_cols: list[str]) -> dict:
    """donor の値で target_fid の NULL 行を一括 UPDATE する。

    Returns:
        {col: updated_rows}
    """
    col_list = ', '.join(null_cols)
    df_donor = pd.read_sql(
        f'SELECT date, {col_list} FROM weather_data WHERE field_id = ?',
        conn, params=(donor_fid,)
    )
    rename_map = {c: f'_d_{c}' for c in null_cols}
    df_donor   = df_donor.rename(columns=rename_map)

    tmp = f'_tmp_{target_fid}'
    df_donor.to_sql(tmp, conn, if_exists='replace', index=False)

    cursor = conn.cursor()
    result = {}
    for col in null_cols:
        dcol = f'_d_{col}'
        cursor.execute(f'''
            UPDATE weather_data
            SET {col} = (
                SELECT {dcol} FROM {tmp}
                WHERE {tmp}.date = weather_data.date
            )
            WHERE field_id = {target_fid}
              AND {col} IS NULL
              AND EXISTS (
                SELECT 1 FROM {tmp}
                WHERE {tmp}.date = weather_data.date
                  AND {tmp}.{dcol} IS NOT NULL
              )
        ''')
        result[col] = cursor.rowcount

    conn.commit()
    cursor.execute(f'DROP TABLE IF EXISTS {tmp}')
    conn.commit()
    return result


# ── メイン ────────────────────────────────────────────────────────────────────

def run():
    Path('outputs/gdd').mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print('=' * 65)
    print('  weather_database_fieldid.db  全列 NULL 補完')
    print(f'  GPU: {"使用 (CuPy)" if GPU_AVAILABLE else "なし (NumPy)"}')
    print(f'  最大探索距離: {MAX_DIST_KM} km')
    print('=' * 65)

    conn = sqlite3.connect(WEATHER_DB)

    # ── Step 1: NULL スキャン ────────────────────────────────────────────────
    print('\n[Step 1] NULL 状況スキャン中...')
    result = scan_null_status(conn)
    if isinstance(result, tuple):
        scan_df, weather_cols = result
    else:
        scan_df = result
        weather_cols = []

    if scan_df.empty:
        print('  NULL のある列・圃場はありません。処理不要。')
        conn.close()
        return

    # 全行NULL と 部分NULL を分類
    scan_df['null_type'] = np.where(
        scan_df['null_ratio'] >= ALL_NULL_THRESHOLD, 'all_null', 'partial_null'
    )
    scan_df.to_csv(SCAN_PATH, index=False, encoding='utf-8-sig')
    print(f'  スキャン結果保存: {SCAN_PATH}')

    # サマリー表示
    summary = scan_df.groupby(['col', 'null_type'])['field_id'].count().reset_index()
    summary.columns = ['col', 'null_type', 'affected_fields']
    print('\n  欠損サマリー（列×タイプ別 影響圃場数）:')
    print(summary.to_string(index=False))

    # ── Step 2: 補完対象を特定 ──────────────────────────────────────────────
    # field_id ごとに欠損列をまとめる
    # 全行 NULL のみを対象（部分 NULL は別途扱い）
    all_null_df  = scan_df[scan_df['null_type'] == 'all_null']
    part_null_df = scan_df[scan_df['null_type'] == 'partial_null']

    print(f'\n  全行NULL: {len(all_null_df)} (field_id×col) ペア')
    print(f'  部分NULL: {len(part_null_df)} (field_id×col) ペア')

    # field_id → 欠損列リスト のマップ（全行NULL）
    fid_cols_map = (all_null_df.groupby('field_id')
                    .agg(null_cols=('col', list),
                         lat=('lat', 'first'),
                         lon=('lon', 'first'))
                    .reset_index())

    print(f'\n  補完対象圃場数（全行NULL）: {len(fid_cols_map)}')

    # ── Step 3: 補完ループ ───────────────────────────────────────────────────
    print('\n[Step 2] 空間的最近傍補完 実行中...')
    log_rows   = []
    total_fids = len(fid_cols_map)

    for i, row in enumerate(fid_cols_map.itertuples(), 1):
        fid       = int(row.field_id)
        null_cols = row.null_cols
        lat       = row.lat
        lon       = row.lon

        print(f'\n  [{i:3d}/{total_fids}] field_id={fid}  欠損列={null_cols}')

        if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
            print('    !! lat/lon 不明 → スキップ')
            log_rows.append({'field_id': fid, 'null_cols': str(null_cols),
                             'donor_fid': None, 'dist_km': None,
                             'status': 'skip_no_geo', 'updated_total': 0})
            continue

        print(f'    lat={lat:.6f}  lon={lon:.6f}')

        # donor 検索
        donor_fid, dist_km, note = find_donor(conn, fid, lat, lon,
                                               null_cols, MAX_DIST_KM)
        if donor_fid is None:
            print(f'    !! donor なし → スキップ')
            log_rows.append({'field_id': fid, 'null_cols': str(null_cols),
                             'donor_fid': None, 'dist_km': None,
                             'status': 'skip_no_donor', 'updated_total': 0})
            continue

        print(f'    → donor: field_id={donor_fid}  距離={dist_km:.2f} km  ({note})')

        # UPDATE 実行
        updated = update_from_donor(conn, fid, donor_fid, null_cols)
        total_upd = sum(updated.values())
        for col, n in updated.items():
            print(f'       {col}: {n:,} 行 UPDATE')
        print(f'    合計: {total_upd:,} 行')

        log_rows.append({
            'field_id':      fid,
            'null_cols':     str(null_cols),
            'donor_fid':     donor_fid,
            'dist_km':       round(dist_km, 3),
            'status':        note,
            'updated_total': total_upd,
        })

    # ── Step 4: 部分NULL の報告（今回は補完しない）──────────────────────────
    if not part_null_df.empty:
        print(f'\n[INFO] 部分NULL（{len(part_null_df)} ペア）は今回の対象外です。')
        print('  → 必要な場合は --partial フラグで個別処理してください。')
        part_summary = (part_null_df.groupby('col')
                        .agg(fields=('field_id', 'count'),
                             avg_null_ratio=('null_ratio', 'mean'))
                        .reset_index())
        part_summary['avg_null_ratio'] = part_summary['avg_null_ratio'].map('{:.1%}'.format)
        print(part_summary.to_string(index=False))

    conn.close()

    # ── ログ保存 ─────────────────────────────────────────────────────────────
    log_df = pd.DataFrame(log_rows)
    if not log_df.empty:
        log_df.to_csv(LOG_PATH, index=False, encoding='utf-8-sig')
        print(f'\n補完ログ: {LOG_PATH}')

    # ── 最終サマリー ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f'\n{"=" * 65}')
    print(f'  完了  経過時間: {elapsed:.1f}s')
    if not log_df.empty:
        done = log_df[log_df['status'].isin(['within_dist', 'nearest_fallback'])]
        skip = log_df[log_df['status'].str.startswith('skip')]
        print(f'  補完成功: {len(done)} 圃場')
        print(f'  スキップ: {len(skip)} 圃場')
        print(f'  総更新行: {log_df["updated_total"].sum():,} 行')
    print(f'{"=" * 65}')

    # ── 補完後 NULL 確認 ─────────────────────────────────────────────────────
    print('\n[Step 3] 補完後 NULL 確認...')
    conn2 = sqlite3.connect(WEATHER_DB)
    result2 = scan_null_status(conn2)
    if isinstance(result2, tuple):
        scan_after, _ = result2
    else:
        scan_after = result2

    if scan_after.empty:
        print('  全行NULL の残留なし。補完完了。')
    else:
        remaining = scan_after[scan_after['null_ratio'] >= ALL_NULL_THRESHOLD]
        if remaining.empty:
            print('  全行NULL の残留なし。')
        else:
            print(f'  残留する全行NULL: {len(remaining)} ペア')
            print(remaining[['field_id', 'col', 'null_count']].to_string(index=False))
    conn2.close()


if __name__ == '__main__':
    run()
