"""
impute_growing_dates.py
========================
播種日（seed_date）・収穫日（harvest_date）の欠損値を
空間的近傍圃場の日付から補完するスクリプト。

【アルゴリズム】
  1. 同年度の非欠損圃場との球面距離（ハーバーサイン）を計算
  2. 距離が MAX_DIST_KM 以内の最近傍 K 件を取得
  3. dayofyear に変換して中央値を取り、その年の日付に戻す
  4. 同年度に参照圃場がなければ隣接年（±1年, ±2年）に拡張
  5. それでも補完できなければデフォルト（5/1〜10/31）を使用

【出力】
  - 補完済みの DataFrame を CSV と pickle で保存する
    outputs/imputed_dates/questionaire_imputed.csv
    outputs/imputed_dates/questionaire_imputed.pkl
  - --write-db フラグを付けると FieldData_fieldid.db の Questionaire テーブルも更新する
    （欠損だった seed_date / harvest_date のみ上書き。元から値のある行は変更しない）

【使い方】
  python impute_growing_dates.py                # 補完 → CSV/pkl 保存のみ
  python impute_growing_dates.py --write-db     # 補完 → CSV/pkl 保存 + DB 反映
  python impute_growing_dates.py --dry-run      # 統計のみ表示（DB更新なし）
"""

import argparse
import os
import sqlite3
import warnings
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── パラメータ ───────────────────────────────────────────────────────────────
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
OUTPUT_DIR  = os.path.join('outputs', 'imputed_dates')

TARGET_YEARS   = [2015, 2016, 2017, 2018]
K_NEIGHBORS    = 3          # 使用する近傍圃場数
MAX_DIST_KM    = 100.0      # 近傍探索の最大距離[km]
FALLBACK_YEARS = [1, -1, 2, -2]  # 隣接年の探索順序（オフセット）

# デフォルト日付（補完不可時）
DEFAULT_SEED_MONTH, DEFAULT_SEED_DAY         = 5,  1
DEFAULT_HARVEST_MONTH, DEFAULT_HARVEST_DAY   = 10, 31


# ── ハーバーサイン距離 ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """スカラー (lat1,lon1) と配列 (lat2,lon2) 間の球面距離[km]を返す。"""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ── 日付の中央値（dayofyear ベース） ─────────────────────────────────────────

def median_date_from_doys(doys: list[int], year: int) -> pd.Timestamp:
    """dayofyear のリストの中央値を取り、指定年の Timestamp に変換する。"""
    median_doy = int(np.median(doys))
    # うるう年対応: 366日を超えないようにクランプ
    max_doy = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    median_doy = min(median_doy, max_doy)
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=median_doy - 1)


# ── 1サンプルの補完ロジック ──────────────────────────────────────────────────

def impute_one(row: pd.Series, donor_pool: dict) -> dict:
    """1圃場1年度の seed_date / harvest_date を補完する。

    Args:
        row        : 対象行（field_id, year, seed_date, harvest_date, lat, lon）
        donor_pool : {year: DataFrame} 参照可能な圃場（非欠損）のプール

    Returns:
        dict with keys:
            seed_date_imp, harvest_date_imp  (pd.Timestamp)
            impute_source  (str)
    """
    year     = int(row['year'])
    need_sd  = pd.isna(row['seed_date'])
    need_hd  = pd.isna(row['harvest_date'])
    lat, lon = row['lat'], row['lon']

    # lat/lon 欠損 → デフォルトにフォールバック
    if pd.isna(lat) or pd.isna(lon):
        return _default_result(row, year, 'default_no_geo')

    # 探索年度リスト（同年 → 隣接年 の順）
    search_years = [year] + [year + d for d in FALLBACK_YEARS
                              if year + d in TARGET_YEARS]

    for search_year in search_years:
        donors = donor_pool.get(search_year)
        if donors is None or len(donors) == 0:
            continue

        # 距離計算
        dists = haversine_km(lat, lon,
                             donors['lat'].to_numpy(),
                             donors['lon'].to_numpy())
        within = dists <= MAX_DIST_KM
        if not within.any():
            continue

        # 距離順ソート → 上位 K 件
        idx_sorted = np.argsort(dists[within])
        k_donors   = donors[within].iloc[idx_sorted[:K_NEIGHBORS]]

        # seed_date
        sd_imp = row['seed_date']
        if need_sd:
            doys = k_donors['seed_date'].dropna().apply(lambda d: d.timetuple().tm_yday).tolist()
            if doys:
                sd_imp = median_date_from_doys(doys, year)

        # harvest_date
        hd_imp = row['harvest_date']
        if need_hd:
            doys = k_donors['harvest_date'].dropna().apply(lambda d: d.timetuple().tm_yday).tolist()
            if doys:
                hd_imp = median_date_from_doys(doys, year)

        # 補完できたか確認
        sd_ok = pd.notna(sd_imp)
        hd_ok = pd.notna(hd_imp)
        if (not need_sd or sd_ok) and (not need_hd or hd_ok):
            src = 'knn_same_year' if search_year == year else f'knn_year{search_year}'
            return {'seed_date_imp': sd_imp, 'harvest_date_imp': hd_imp,
                    'impute_source': src}

    # 全年度で失敗 → デフォルト
    return _default_result(row, year, 'default_fallback')


def _default_result(row, year, src):
    """デフォルト日付を返す。既存の値は保持する。"""
    sd = row['seed_date'] if pd.notna(row['seed_date']) \
        else pd.Timestamp(year, DEFAULT_SEED_MONTH, DEFAULT_SEED_DAY)
    hd = row['harvest_date'] if pd.notna(row['harvest_date']) \
        else pd.Timestamp(year, DEFAULT_HARVEST_MONTH, DEFAULT_HARVEST_DAY)
    return {'seed_date_imp': sd, 'harvest_date_imp': hd, 'impute_source': src}


# ── メイン処理 ───────────────────────────────────────────────────────────────

def build_donor_pool(df: pd.DataFrame) -> dict:
    """年度 → 非欠損かつ lat/lon ありの参照圃場 DataFrame を返す。"""
    pool = {}
    for year in TARGET_YEARS:
        sub = df[df['year'] == year]
        donors = sub[
            sub['seed_date'].notna() &
            sub['harvest_date'].notna() &
            sub['lat'].notna() &
            sub['lon'].notna()
        ].copy()
        pool[year] = donors.reset_index(drop=True)
    return pool


def run_imputation(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── データ読み込み ─────────────────────────────────────────────────
    print('Questionaire テーブル読み込み...', end=' ')
    conn = sqlite3.connect(args.field_db)
    df = pd.read_sql('''
        SELECT field_id, year, seed_date, harvest_date, yield, lat, lon
        FROM Questionaire
        WHERE year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()

    df['field_id']     = pd.to_numeric(df['field_id'], errors='coerce')
    df = df.dropna(subset=['field_id']).copy()
    df['field_id']     = df['field_id'].astype(int)
    df['year']         = df['year'].astype(int)
    df['seed_date']    = pd.to_datetime(df['seed_date'],    errors='coerce')
    df['harvest_date'] = pd.to_datetime(df['harvest_date'], errors='coerce')
    df['lat']          = pd.to_numeric(df['lat'], errors='coerce')
    df['lon']          = pd.to_numeric(df['lon'], errors='coerce')
    print(f'{len(df)} サンプル')

    # ── 補完前の統計 ───────────────────────────────────────────────────
    print('\n=== 補完前 欠損状況 ===')
    for year in TARGET_YEARS:
        sub = df[df['year'] == year]
        print(f'  {year}: n={len(sub):3d}  '
              f'seed_date欠損={sub["seed_date"].isna().sum()}  '
              f'harvest_date欠損={sub["harvest_date"].isna().sum()}')

    if args.dry_run:
        print('\n[dry-run] 補完は実行しません。')
        return

    # ── 参照プール構築 ─────────────────────────────────────────────────
    donor_pool = build_donor_pool(df)
    print('\n=== 参照プール ===')
    for year, donors in donor_pool.items():
        print(f'  {year}: {len(donors)} 件')

    # ── 補完ループ ─────────────────────────────────────────────────────
    print('\n補完処理中...')
    results = []
    source_counts = {}

    for _, row in df.iterrows():
        need_sd = pd.isna(row['seed_date'])
        need_hd = pd.isna(row['harvest_date'])

        if not need_sd and not need_hd:
            # 欠損なし → そのまま
            results.append({
                'seed_date_imp':   row['seed_date'],
                'harvest_date_imp': row['harvest_date'],
                'impute_source':   'original',
            })
        else:
            res = impute_one(row, donor_pool)
            results.append(res)
            src = res['impute_source']
            source_counts[src] = source_counts.get(src, 0) + 1

    result_df = pd.DataFrame(results)
    out = df.copy()
    out['seed_date_imp']    = result_df['seed_date_imp'].values
    out['harvest_date_imp'] = result_df['harvest_date_imp'].values
    out['impute_source']    = result_df['impute_source'].values

    # ── 補完後の統計 ───────────────────────────────────────────────────
    print('\n=== 補完後 統計 ===')
    print('  補完ソース内訳:')
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f'    {src:<25}: {cnt} 件')

    still_na_sd = out['seed_date_imp'].isna().sum()
    still_na_hd = out['harvest_date_imp'].isna().sum()
    print(f'  補完後 seed_date 欠損    : {still_na_sd} 件')
    print(f'  補完後 harvest_date 欠損 : {still_na_hd} 件')

    # ── CSV / pkl 保存 ────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'questionaire_imputed.csv')
    pkl_path = os.path.join(args.output_dir, 'questionaire_imputed.pkl')
    out.to_csv(csv_path, index=False)
    out.to_pickle(pkl_path)
    print(f'\n  CSV  → {csv_path}')
    print(f'  pkl  → {pkl_path}')

    # ── DB 書き戻し（--write-db 指定時のみ） ──────────────────────────
    if getattr(args, 'write_db', False):
        write_back_to_db(out, args.field_db)

    print('\nFinished.')
    return out


# ── DB 書き戻し ───────────────────────────────────────────────────────────────

def write_back_to_db(out: pd.DataFrame, field_db: str):
    """補完済み DataFrame の seed_date / harvest_date を DB に書き戻す。

    更新対象: impute_source が 'original' 以外の行のみ。
    更新内容: seed_date_imp / harvest_date_imp を Questionaire テーブルの
              seed_date / harvest_date に SET する。
    キー    : field_id AND year の一致する行を対象とする。

    Args:
        out      (pd.DataFrame): run_imputation() が返す補完済み DataFrame。
        field_db (str):          書き戻し先の SQLite DB パス。
    """
    imputed = out[out['impute_source'] != 'original'].copy()
    print(f'\n=== DB 書き戻し ===')
    print(f'  対象: {len(imputed)} 件 (impute_source != original)')

    # 日付を ISO 文字列に変換（SQLite は TEXT で保存）
    imputed['sd_str'] = imputed['seed_date_imp'].apply(
        lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else None
    )
    imputed['hd_str'] = imputed['harvest_date_imp'].apply(
        lambda d: d.strftime('%Y-%m-%d') if pd.notna(d) else None
    )

    conn = sqlite3.connect(field_db)
    cursor = conn.cursor()

    updated_sd = 0
    updated_hd = 0
    skipped    = 0

    for _, row in imputed.iterrows():
        fid  = int(row['field_id'])
        year = int(row['year'])
        src  = row['impute_source']

        # 元の値が NULL だったものだけ更新（元から値のある列は触らない）
        # seed_date の更新
        cursor.execute('''
            UPDATE Questionaire
            SET seed_date = ?
            WHERE field_id = ? AND year = ? AND seed_date IS NULL
        ''', (row['sd_str'], fid, year))
        if cursor.rowcount > 0:
            updated_sd += 1

        # harvest_date の更新
        cursor.execute('''
            UPDATE Questionaire
            SET harvest_date = ?
            WHERE field_id = ? AND year = ? AND harvest_date IS NULL
        ''', (row['hd_str'], fid, year))
        if cursor.rowcount > 0:
            updated_hd += 1

        if cursor.rowcount == 0:
            skipped += 1

    conn.commit()
    conn.close()

    print(f'  seed_date 更新    : {updated_sd} 件')
    print(f'  harvest_date 更新 : {updated_hd} 件')
    if skipped > 0:
        print(f'  スキップ（既に値あり）: {skipped} 件')
    print(f'  DB 更新完了: {field_db}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='播種日・収穫日の欠損を空間的近傍補完するスクリプト')
    p.add_argument('--field-db',   default=FIELD_DB,   dest='field_db')
    p.add_argument('--output-dir', default=OUTPUT_DIR, dest='output_dir')
    p.add_argument('--k',          type=int, default=K_NEIGHBORS,
                   help=f'近傍数 (default: {K_NEIGHBORS})')
    p.add_argument('--max-dist',   type=float, default=MAX_DIST_KM,
                   dest='max_dist_km',
                   help=f'最大探索距離 km (default: {MAX_DIST_KM})')
    p.add_argument('--dry-run',    action='store_true',
                   help='統計表示のみ、補完・保存・DB更新は実行しない')
    p.add_argument('--write-db',   action='store_true',
                   dest='write_db',
                   help='補完結果を FieldData_fieldid.db の Questionaire テーブルに書き戻す')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # CLI 引数をグローバルパラメータに反映
    K_NEIGHBORS = args.k
    MAX_DIST_KM = args.max_dist_km
    run_imputation(args)
