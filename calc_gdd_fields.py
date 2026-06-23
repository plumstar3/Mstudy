"""
calc_gdd_fields.py
============================================================
各圃場（field_id）の有効積算温度（GDD）を算出するスクリプト

【データソース】
  - FieldData_fieldid.db   / Questionaire テーブル
      → field_id, year, seed_date, harvest_date（生育期間の定義）
  - weather_database_fieldid.db / weather_data テーブル
      → field_id, date, TMP_max, TMP_min（気温データ）

【処理フロー】
  Step 1 : Questionaire から seed_date / harvest_date を取得
  Step 2 : weather_data から TMP_max / TMP_min を取得
  Step 3 : 気象データに seed_date・harvest_date をマージ
  Step 4 : 生育期間（seed_date ≤ date ≤ harvest_date）でフィルタリング
  Step 5 : 修正平均法でGDDをベクトル演算（forループ不使用）
  Step 6 : field_id × year グループ内で累積GDDを計算（groupby.cumsum）
  Step 7 : 詳細CSV・サマリーCSV・npy形式で保存

【出力ファイル】
  outputs/gdd/
    gdd_daily.csv     : 日別GDD詳細（field_id, year, date, 日次GDD, 累積GDD, ...）
    gdd_summary.csv   : 圃場×年の集計（播種〜収穫の総GDD、有効日数等）
    gdd_matrix.npy    : (field_id×year) × 累積GDD 最終値の行列

【使い方】
  python calc_gdd_fields.py
  python calc_gdd_fields.py --year-start 2015 --year-end 2018
  python calc_gdd_fields.py --dry-run        # データ読み込みのみ確認
"""

import argparse
import os
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 既存の calc_gdd モジュールから関数をインポート
from calc_gdd import calc_gdd, T_BASE, T_UPPER

warnings.filterwarnings('ignore')

# ── パス設定 ──────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).resolve().parent
FIELD_DB    = _BASE / 'data' / 'processed' / 'FieldData_fieldid.db'
WEATHER_DB  = _BASE / 'data' / 'processed' / 'weather_database_fieldid.db'
OUT_DIR     = _BASE / 'outputs' / 'gdd'

# 対象年度
DEFAULT_YEAR_START = 2015
DEFAULT_YEAR_END   = 2018


# ── Step 1: Questionaire から生育期間を取得 ───────────────────────────────────

def load_questionaire(field_db: Path, year_start: int, year_end: int) -> pd.DataFrame:
    """Questionaire テーブルから field_id・year・seed_date・harvest_date を取得。

    seed_date / harvest_date が両方 NULL の行は除外する。
    """
    conn = sqlite3.connect(field_db)
    df = pd.read_sql('''
        SELECT field_id, year, seed_date, harvest_date
        FROM Questionaire
        WHERE field_id IS NOT NULL
          AND year BETWEEN ? AND ?
          AND (seed_date IS NOT NULL OR harvest_date IS NOT NULL)
    ''', conn, params=(year_start, year_end))
    conn.close()

    df['field_id']     = pd.to_numeric(df['field_id'], errors='coerce').astype('Int64')
    df['year']         = df['year'].astype(int)
    df['seed_date']    = pd.to_datetime(df['seed_date'],    errors='coerce')
    df['harvest_date'] = pd.to_datetime(df['harvest_date'], errors='coerce')

    # seed_date が NaT の場合は year/5/1 をデフォルトとして補完
    mask_no_sd = df['seed_date'].isna()
    df.loc[mask_no_sd, 'seed_date'] = df.loc[mask_no_sd, 'year'].apply(
        lambda y: pd.Timestamp(y, 5, 1)
    )
    # harvest_date が NaT の場合は year/10/31 をデフォルトとして補完
    mask_no_hd = df['harvest_date'].isna()
    df.loc[mask_no_hd, 'harvest_date'] = df.loc[mask_no_hd, 'year'].apply(
        lambda y: pd.Timestamp(y, 10, 31)
    )

    return df.reset_index(drop=True)


# ── Step 2: 気象データ取得 ────────────────────────────────────────────────────

def load_weather(weather_db: Path, field_ids: list[int],
                 year_start: int, year_end: int) -> pd.DataFrame:
    """weather_data テーブルから TMP_max・TMP_min を取得。

    field_id が指定リストに含まれ、対象年度の範囲内のレコードのみ取得する。
    """
    fid_ph = ','.join(['?' for _ in field_ids])
    conn   = sqlite3.connect(weather_db)
    df = pd.read_sql(f'''
        SELECT field_id, date, TMP_max, TMP_min
        FROM weather_data
        WHERE field_id IN ({fid_ph})
          AND CAST(SUBSTR(date, 1, 4) AS INTEGER) BETWEEN ? AND ?
        ORDER BY field_id, date
    ''', conn, params=field_ids + [year_start, year_end])
    conn.close()

    df['field_id'] = df['field_id'].astype(int)
    df['date']     = pd.to_datetime(df['date'])
    # 年情報を付与（マージキーに使用）
    df['year']     = df['date'].dt.year
    return df


# ── Step 3〜6: GDD 計算（完全ベクトル演算）────────────────────────────────────

def compute_gdd_all_fields(quest_df: pd.DataFrame,
                           weather_df: pd.DataFrame) -> pd.DataFrame:
    """全 field_id × year の日別GDDと累積GDDをベクトル演算で一括算出する。

    処理フロー（forループなし）:
      1. weather_df に seed_date / harvest_date をマージ（field_id + year キー）
      2. 生育期間（seed_date ≤ date ≤ harvest_date）でフィルタリング
      3. TMP_max / TMP_min を生理学的範囲にクランプ（clip）
      4. 日次GDD = (補正T_max + 補正T_min) / 2 − T_base  → 負値は 0
      5. groupby(['field_id', 'year']).cumsum() で累積GDD
    """
    # ── Step 3: マージ ─────────────────────────────────────────────────────
    merged = weather_df.merge(
        quest_df[['field_id', 'year', 'seed_date', 'harvest_date']],
        on=['field_id', 'year'],
        how='inner'
    )

    # ── Step 4: 生育期間フィルタリング（ベクトル条件）─────────────────────
    in_season = (merged['date'] >= merged['seed_date']) & \
                (merged['date'] <= merged['harvest_date'])
    df = merged[in_season].copy()

    # ── Step 5: GDD 算出（clip によるベクトル演算）────────────────────────
    # T_max の補正: T_upper を上限、T_base を下限にクランプ
    df['補正後T_max'] = df['TMP_max'].clip(lower=T_BASE, upper=T_UPPER)
    # T_min の補正: T_base を下限にクランプ（上限なし）
    df['補正後T_min'] = df['TMP_min'].clip(lower=T_BASE)
    # 日次GDD = 平均補正気温 − 基準温度（負値は 0）
    df['日次GDD'] = ((df['補正後T_max'] + df['補正後T_min']) / 2.0 - T_BASE).clip(lower=0.0)

    # ── Step 6: 累積GDD（field_id × year グループ内で日付順 cumsum）─────
    df = df.sort_values(['field_id', 'year', 'date'])
    df['累積GDD'] = df.groupby(['field_id', 'year'])['日次GDD'].cumsum()

    # 何日目の栽培日かも付与（デバッグ・可視化に便利）
    df['栽培日数'] = df.groupby(['field_id', 'year']).cumcount() + 1

    return df.reset_index(drop=True)


# ── Step 7: サマリー集計 ──────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """field_id × year ごとの集計サマリーを作成する。

    Returns:
        DataFrame with columns:
            field_id, year, seed_date, harvest_date,
            有効日数（GDD>0の日数）, 総GDD（累積GDDの最終値）,
            平均日次GDD, 最大日次GDD
    """
    grp = df.groupby(['field_id', 'year'])

    summary = grp.agg(
        seed_date       = ('seed_date',    'first'),
        harvest_date    = ('harvest_date', 'first'),
        栽培日数        = ('栽培日数',     'max'),
        有効日数        = ('日次GDD',      lambda x: (x > 0).sum()),
        総GDD           = ('累積GDD',      'last'),   # cumsum の最終値 = 合計
        平均日次GDD     = ('日次GDD',      'mean'),
        最大日次GDD     = ('日次GDD',      'max'),
    ).reset_index()

    summary['平均日次GDD'] = summary['平均日次GDD'].round(2)
    summary['総GDD']       = summary['総GDD'].round(2)
    summary['最大日次GDD'] = summary['最大日次GDD'].round(2)
    return summary


# ── メイン処理 ────────────────────────────────────────────────────────────────

def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 62)
    print('  圃場別 有効積算温度（GDD）算出')
    print(f'  T_base={T_BASE}℃  /  T_upper={T_UPPER}℃')
    print(f'  対象年: {args.year_start}〜{args.year_end}')
    print('=' * 62)

    # ── Step 1: Questionaire 読み込み ───────────────────────────────────
    print('\n[Step 1] 生育期間データ読み込み (Questionaire)...')
    quest_df = load_questionaire(args.field_db, args.year_start, args.year_end)
    n_samples = len(quest_df)
    n_fields  = quest_df['field_id'].nunique()
    print(f'  {n_samples} レコード  ({n_fields} 圃場 × {quest_df["year"].nunique()} 年)')
    print(f'  seed_date 欠損補完  : {quest_df["seed_date"].isna().sum()} 件 → 5/1 デフォルト')
    print(f'  harvest_date 欠損補完: {quest_df["harvest_date"].isna().sum()} 件 → 10/31 デフォルト')

    if args.dry_run:
        print('\n[dry-run] ここで終了。')
        return

    # ── Step 2: 気象データ読み込み ─────────────────────────────────────
    print(f'\n[Step 2] 気象データ読み込み (weather_data)...')
    all_fids   = sorted(quest_df['field_id'].dropna().astype(int).unique().tolist())
    weather_df = load_weather(args.weather_db, all_fids,
                              args.year_start, args.year_end)
    print(f'  {len(weather_df):,} 行取得  '
          f'({weather_df["field_id"].nunique()} 圃場 × '
          f'{weather_df["date"].dt.year.nunique()} 年)')
    print(f'  TMP_max 欠損: {weather_df["TMP_max"].isna().sum():,} 件  '
          f'TMP_min 欠損: {weather_df["TMP_min"].isna().sum():,} 件')

    # ── Step 3〜6: GDD 一括計算 ────────────────────────────────────────
    print('\n[Step 3-6] GDD 算出（ベクトル演算）...')
    gdd_df = compute_gdd_all_fields(quest_df, weather_df)
    print(f'  生育期間内レコード: {len(gdd_df):,} 行')
    print(f'  処理圃場数        : {gdd_df["field_id"].nunique()} 圃場')
    print(f'  日次GDD 統計      : mean={gdd_df["日次GDD"].mean():.2f}  '
          f'max={gdd_df["日次GDD"].max():.1f}  '
          f'0の割合={( gdd_df["日次GDD"] == 0).mean()*100:.1f}%')

    # ── Step 7: サマリー集計 ────────────────────────────────────────────
    print('\n[Step 7] サマリー集計...')
    summary_df = build_summary(gdd_df)
    print(f'  サマリー行数: {len(summary_df)}')
    print(f'\n  総GDD 統計:')
    print(f'    min   = {summary_df["総GDD"].min():.1f}')
    print(f'    mean  = {summary_df["総GDD"].mean():.1f}')
    print(f'    median= {summary_df["総GDD"].median():.1f}')
    print(f'    max   = {summary_df["総GDD"].max():.1f}')

    # ── 保存 ────────────────────────────────────────────────────────────
    print('\n[Step 8] 保存中...')

    # 出力列を整理（日別詳細）
    out_cols = ['field_id', 'year', 'date', 'seed_date', 'harvest_date',
                'TMP_max', 'TMP_min', '補正後T_max', '補正後T_min',
                '日次GDD', '累積GDD', '栽培日数']
    daily_path   = OUT_DIR / 'gdd_daily.csv'
    summary_path = OUT_DIR / 'gdd_summary.csv'
    npy_path     = OUT_DIR / 'gdd_summary.npy'

    # 日別CSV
    save_df = gdd_df[out_cols].copy()
    save_df['date']         = pd.to_datetime(save_df['date']).dt.strftime('%Y-%m-%d')
    save_df['seed_date']    = pd.to_datetime(save_df['seed_date']).dt.strftime('%Y-%m-%d')
    save_df['harvest_date'] = pd.to_datetime(save_df['harvest_date']).dt.strftime('%Y-%m-%d')
    for col in ['補正後T_max', '補正後T_min', '日次GDD', '累積GDD']:
        save_df[col] = save_df[col].round(3)
    save_df.to_csv(daily_path, index=False, encoding='utf-8-sig')
    print(f'  日別GDD CSV   : {daily_path}  ({len(save_df):,} 行)')

    # サマリーCSV
    sum_save = summary_df.copy()
    sum_save['seed_date']    = pd.to_datetime(sum_save['seed_date']).dt.strftime('%Y-%m-%d')
    sum_save['harvest_date'] = pd.to_datetime(sum_save['harvest_date']).dt.strftime('%Y-%m-%d')
    sum_save.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f'  サマリーCSV   : {summary_path}  ({len(sum_save)} 行)')

    # numpy 保存（field_id, year, 総GDD の3列）
    npy_data = summary_df[['field_id', 'year', '総GDD']].to_numpy(dtype=np.float32)
    np.save(npy_path, npy_data)
    print(f'  GDD numpy     : {npy_path}  shape={npy_data.shape}')

    # ── プレビュー ─────────────────────────────────────────────────────
    print('\n=== サマリー プレビュー（先頭10行）===')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary_df.head(10).to_string(index=False))

    print('\n=== 日別GDD プレビュー（先頭20行）===')
    print(save_df.head(20).to_string(index=False))

    print(f'\n完了  出力先: {OUT_DIR}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='圃場別 有効積算温度（GDD）算出スクリプト'
    )
    p.add_argument('--field-db',    type=Path, default=FIELD_DB,   dest='field_db')
    p.add_argument('--weather-db',  type=Path, default=WEATHER_DB, dest='weather_db')
    p.add_argument('--out-dir',     type=Path, default=OUT_DIR,    dest='out_dir')
    p.add_argument('--year-start',  type=int,  default=DEFAULT_YEAR_START,
                   dest='year_start', help=f'開始年 (default: {DEFAULT_YEAR_START})')
    p.add_argument('--year-end',    type=int,  default=DEFAULT_YEAR_END,
                   dest='year_end',   help=f'終了年 (default: {DEFAULT_YEAR_END})')
    p.add_argument('--dry-run',     action='store_true',
                   help='データ読み込みまで実行して終了（GDD計算・保存を行わない）')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
