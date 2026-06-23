"""
fix_field5_harvest.py
============================================================
field_id=5 の harvest_date 修正スクリプト

【修正内容】
  1. field_id=5 の誤入力された harvest_date（2016-07-25）を
     flower_date 列に移し替える
  2. field_id=5 の harvest_date を NULL に戻した上で、
     impute_growing_dates.py と同じ空間的KNN補完法で再推定
  3. 両方の変更を FieldData_fieldid.db に書き戻す

【KNN補完パラメータ（impute_growing_dates.py と同一）】
  K_NEIGHBORS = 3
  MAX_DIST_KM = 100.0
  dayofyear の中央値を用いて対象年の日付に変換

【処理フロー】
  Step 1: 現在の field_id=5 のデータを確認・バックアップ表示
  Step 2: 誤入力値を flower_date に移し替え（harvest_date → flower_date）
  Step 3: 同年度の非欠損圃場（lat/lon あり）を donor pool として構築
  Step 4: ハーバーサイン距離で近傍 K 件を選択
  Step 5: 近傍の harvest_date の dayofyear 中央値で補完日を算出
  Step 6: DB に 2 件の UPDATE を実行（flower_date 更新、harvest_date 更新）
  Step 7: 変更結果を再読み込みして確認
"""

import sqlite3
import numpy as np
import pandas as pd

# ── パラメータ ─────────────────────────────────────────────────────────────────
FIELD_DB    = 'data/processed/FieldData_fieldid.db'
TARGET_FID  = 5
K_NEIGHBORS = 3
MAX_DIST_KM = 100.0

# デフォルト harvest_date（補完失敗時のフォールバック）
DEFAULT_HARVEST_MONTH = 10
DEFAULT_HARVEST_DAY   = 31


# ── ハーバーサイン距離 ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """スカラー (lat1,lon1) と配列 (lat2,lon2) 間の球面距離[km]を返す。"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── dayofyear 中央値 → 日付変換 ───────────────────────────────────────────────

def median_doy_to_date(doys: list[int], year: int) -> pd.Timestamp:
    """dayofyear リストの中央値を対象年の Timestamp に変換する。"""
    median_doy = int(np.median(doys))
    # うるう年対応
    max_doy = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    median_doy = min(median_doy, max_doy)
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=median_doy - 1)


# ── メイン処理 ────────────────────────────────────────────────────────────────

def main():
    print('=' * 62)
    print(f'  field_id={TARGET_FID} harvest_date 修正 + flower_date 移し替え')
    print('=' * 62)

    conn = sqlite3.connect(FIELD_DB)

    # ── Step 1: 現状確認 ────────────────────────────────────────────────────
    print(f'\n[Step 1] 現在の field_id={TARGET_FID} データ確認')
    df_target = pd.read_sql('''
        SELECT field_id, year, seed_date, flower_date, harvest_date, lat, lon
        FROM Questionaire
        WHERE field_id = ?
        ORDER BY year
    ''', conn, params=(TARGET_FID,))
    print(df_target.to_string(index=False))

    if df_target.empty:
        print(f'  !! field_id={TARGET_FID} のデータが見つかりません。終了します。')
        conn.close()
        return

    # field_id=5 の処理対象行を確認
    target_rows = df_target[df_target['harvest_date'].notna()].copy()
    if target_rows.empty:
        print('  !! harvest_date が既に NULL です。処理不要。')
        conn.close()
        return

    for _, row in target_rows.iterrows():
        year        = int(row['year'])
        wrong_hd    = row['harvest_date']   # 誤って入力された値（→ flower_date へ）
        lat, lon    = float(row['lat']), float(row['lon'])

        print(f'\n  対象: year={year}, 誤入力 harvest_date={wrong_hd}')
        print(f'  lat={lat:.6f}, lon={lon:.6f}')

        # ── Step 2: 同年の donor pool を取得 ──────────────────────────────
        print(f'\n[Step 2] year={year} の donor pool 構築...')
        df_donors = pd.read_sql('''
            SELECT field_id, year, harvest_date, lat, lon
            FROM Questionaire
            WHERE year = ?
              AND field_id != ?
              AND harvest_date IS NOT NULL
              AND lat IS NOT NULL
              AND lon IS NOT NULL
        ''', conn, params=(year, TARGET_FID))
        df_donors['harvest_date'] = pd.to_datetime(df_donors['harvest_date'],
                                                    errors='coerce')
        df_donors = df_donors.dropna(subset=['harvest_date'])
        print(f'  候補圃場数: {len(df_donors)} 件')

        if df_donors.empty:
            print('  !! 同年に参照可能な圃場がありません。')
            new_hd = pd.Timestamp(year, DEFAULT_HARVEST_MONTH, DEFAULT_HARVEST_DAY)
            impute_src = 'default_no_donors'
        else:
            # ── Step 3: ハーバーサイン距離で近傍選択 ─────────────────────
            dists  = haversine_km(lat, lon,
                                   df_donors['lat'].to_numpy(),
                                   df_donors['lon'].to_numpy())
            within = dists <= MAX_DIST_KM
            n_within = within.sum()
            print(f'  {MAX_DIST_KM}km 以内の圃場: {n_within} 件')

            if n_within == 0:
                print(f'  !! {MAX_DIST_KM}km 以内に参照圃場なし → デフォルト日付を使用')
                new_hd = pd.Timestamp(year, DEFAULT_HARVEST_MONTH, DEFAULT_HARVEST_DAY)
                impute_src = 'default_no_nearby'
            else:
                # 距離順ソート → 上位 K 件
                idx_sorted = np.argsort(dists[within])
                k_donors   = df_donors[within].iloc[idx_sorted[:K_NEIGHBORS]]

                print(f'  使用する近傍 {min(K_NEIGHBORS, n_within)} 件:')
                for _, d in k_donors.iterrows():
                    dist_km = haversine_km(lat, lon,
                                           np.array([float(d['lat'])]),
                                           np.array([float(d['lon'])]))[0]
                    print(f'    field_id={int(d["field_id"]):3d}  '
                          f'harvest_date={d["harvest_date"].date()}  '
                          f'距離={dist_km:.1f}km')

                # ── Step 4: dayofyear 中央値で補完日を算出 ──────────────
                doys   = k_donors['harvest_date'].apply(
                    lambda d: d.timetuple().tm_yday
                ).tolist()
                new_hd = median_doy_to_date(doys, year)
                impute_src = f'knn_{min(K_NEIGHBORS, n_within)}nearest'

        print(f'\n  補完結果:')
        print(f'    flower_date  ← {wrong_hd}  （誤入力値を移し替え）')
        print(f'    harvest_date ← {new_hd.date()}  （KNN補完: {impute_src}）')

        # ── Step 5: DB に書き戻し ────────────────────────────────────────
        print(f'\n[Step 3] DB 更新実行...')
        cursor = conn.cursor()

        # ① flower_date を更新（誤入力値を正しい列に移す）
        cursor.execute('''
            UPDATE Questionaire
            SET flower_date = ?
            WHERE field_id = ? AND year = ?
        ''', (wrong_hd, TARGET_FID, year))
        print(f'  flower_date 更新: {cursor.rowcount} 行')

        # ② harvest_date を補完値で更新
        cursor.execute('''
            UPDATE Questionaire
            SET harvest_date = ?
            WHERE field_id = ? AND year = ?
        ''', (new_hd.strftime('%Y-%m-%d'), TARGET_FID, year))
        print(f'  harvest_date 更新: {cursor.rowcount} 行')

        conn.commit()

    # ── Step 6: 更新後の確認 ────────────────────────────────────────────────
    print(f'\n[Step 4] 更新後の field_id={TARGET_FID} データ確認')
    df_after = pd.read_sql('''
        SELECT field_id, year, seed_date, flower_date, harvest_date, lat, lon
        FROM Questionaire
        WHERE field_id = ?
        ORDER BY year
    ''', conn, params=(TARGET_FID,))
    print(df_after.to_string(index=False))

    # 周辺圃場と並べて確認
    print(f'\n[Step 5] 近隣圃場との比較（更新後）')
    years_list = df_after['year'].tolist()
    yr_ph = ','.join([str(y) for y in years_list])
    df_compare = pd.read_sql(f'''
        SELECT field_id, year, seed_date, flower_date, harvest_date
        FROM Questionaire
        WHERE field_id IN (3,4,5,6,7)
          AND year IN ({yr_ph})
        ORDER BY field_id, year
    ''', conn)
    print(df_compare.to_string(index=False))

    conn.close()
    print('\n完了。FieldData_fieldid.db を更新しました。')


if __name__ == '__main__':
    main()
