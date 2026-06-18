"""
generate_pretrain_dataset_v2.py
================================
ts2vec 事前学習用データセット生成スクリプト（動的クロッピング版）

【気象データの対象変数について】
  WEATHER_COLS に列挙した 9 変数を対象とします。
  ─ 1981 年から存在するデータのみを対象にしたい場合は、
    WEATHER_COLS_1981 を使用してください（SFW など一部変数は後年から）。
  ─ デフォルトは WEATHER_COLS（全 9 変数）です。
    DB にデータが存在しない期間の値は NaN → 標準化後に 0.0（平均）に置換されます。

【標準化について】
  全サンプル・全期間の値を使い、特徴量（変数）ごと（単変量ごと）に
  mean / std を計算して標準化します（nanmean/nanstd の axis=0 を参照）。
  パディング箇所には 0.0（標準化後の「平均」）が入ります。

【field_id による紐づけとクロッピングについ て】
  ・すべての気象データは field_id を持ちます。
  ・Questionaire の seed_date / harvest_date をもとに各圃場×年の
    栽培期間を決定し、気象データをクロッピングします。
  ・Questionaire に存在する年（QUESTIONAIRE_YEARS: 2015〜2018）は
    実際の seed_date / harvest_date を使用します。
  ・それ以外の年（1981〜2014）はデフォルト期間（5/1〜10/31）を使用します。

【GPU 対応について】
  CuPy が利用可能な場合、標準化・テンソル構築を GPU 上で実行します。
  利用不可の場合は自動的に CPU (NumPy) にフォールバックします。

【処理フロー】
  Step 1: Questionaire から播種日・収穫日を取得
  Step 2: 気象データ全体を取得（field_id で紐づけ）
  Step 3: 事前標準化パラメータ計算（特徴量ごとに mean / std を算出）
  Step 4: 栽培期間テーブル構築（seed_date / harvest_date → start / end）
  Step 5: 動的クロッピング + ゼロパディング（GPU 対応）
  Step 6: 保存

【出力】
  data/processed/soybean_ts2vec_v2/
    pretrain_X_v2.npy       shape: (N, max_len, 9)  float32
    pretrain_meta_v2.csv    field_id, year, seed_date, harvest_date,
                            grow_days, impute_source 等
    pretrain_norm_stats.npy shape: (2, 9)  [mean, std] per feature

【使い方】
  python generate_pretrain_dataset_v2.py
  python generate_pretrain_dataset_v2.py --field-db PATH --weather-db PATH
  python generate_pretrain_dataset_v2.py --dry-run   # 統計確認のみ
  python generate_pretrain_dataset_v2.py --no-gpu    # CPU 強制
  python generate_pretrain_dataset_v2.py --cols 1981 # 1981年から存在する変数のみ
"""

import argparse
import sqlite3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── GPU 可用性チェック ────────────────────────────────────────────────────────
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def get_xp(use_gpu: bool):
    """GPU モードなら cupy、それ以外は numpy を返す。"""
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np


# ── パス設定 ────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).resolve().parent
FIELD_DB    = _BASE / 'data' / 'processed' / 'FieldData_fieldid.db'
WEATHER_DB  = _BASE / 'data' / 'processed' / 'weather_database_fieldid.db'
OUT_DIR     = _BASE / 'data' / 'processed' / 'soybean_ts2vec_v2'

# ── 気象変数の定義 ────────────────────────────────────────────────────────────
# 全 9 変数（DB に存在する全種類）
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# 1981 年から確実に存在する変数のみ（SFW / SWE などは後年から整備された場合に使用）
# ※ 実際に 1981 年から存在するかは DB の内容に依存します。
WEATHER_COLS_1981 = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD']

N_FEAT = len(WEATHER_COLS)  # デフォルト: 9

# デフォルト栽培期間（seed_date / harvest_date 欠損時）
DEFAULT_SEED_MD    = (5, 1)   # 5月1日
DEFAULT_HARVEST_MD = (10, 31) # 10月31日
DEFAULT_GROW_DAYS  = 180      # seed_dateのみある場合の補完日数

# 事前学習の対象年（1981〜2018）
PRETRAIN_YEAR_START = 1981
PRETRAIN_YEAR_END   = 2018

# Questionaire から取得する対象年（播種日補完に使う年）
QUESTIONAIRE_YEARS  = [2015, 2016, 2017, 2018]


# ── 播種日・収穫日の解決 ──────────────────────────────────────────────────────

def resolve_period(seed_date, harvest_date, year: int):
    """seed_date / harvest_date から栽培開始日・終了日を決定する。

    優先順位:
      1. 両方あり → そのまま使用
      2. seed_date のみ → harvest_date = seed_date + DEFAULT_GROW_DAYS
      3. 両方なし → year/5/1 〜 year/10/31 をデフォルトとして使用

    Returns:
        start (pd.Timestamp), end (pd.Timestamp), source (str)
    """
    if pd.notna(seed_date) and pd.notna(harvest_date):
        return seed_date, harvest_date, 'both'
    elif pd.notna(seed_date):
        return seed_date, seed_date + pd.Timedelta(days=DEFAULT_GROW_DAYS), 'seed_only'
    else:
        return (pd.Timestamp(year, *DEFAULT_SEED_MD),
                pd.Timestamp(year, *DEFAULT_HARVEST_MD),
                'default')


# ── Step 1: Questionaire から播種日・収穫日を取得 ─────────────────────────────

def load_questionaire(field_db: Path) -> pd.DataFrame:
    """Questionaire テーブルから (field_id, year, seed_date, harvest_date) を取得。

    DB には既に impute_growing_dates.py による補完済みの日付が入っている前提。
    対象年: QUESTIONAIRE_YEARS（2015〜2018）の field_id をもとに
            事前学習の全年（1981〜2018）の圃場リストを決定する。
    """
    conn = sqlite3.connect(field_db)
    df = pd.read_sql('''
        SELECT field_id, year, seed_date, harvest_date
        FROM Questionaire
        WHERE field_id IS NOT NULL
          AND year BETWEEN ? AND ?
    ''', conn, params=(min(QUESTIONAIRE_YEARS), max(QUESTIONAIRE_YEARS)))
    conn.close()

    df['field_id']     = pd.to_numeric(df['field_id'], errors='coerce')
    df = df.dropna(subset=['field_id']).copy()
    df['field_id']     = df['field_id'].astype(int)
    df['year']         = df['year'].astype(int)
    df['seed_date']    = pd.to_datetime(df['seed_date'],    errors='coerce')
    df['harvest_date'] = pd.to_datetime(df['harvest_date'], errors='coerce')
    return df.reset_index(drop=True)


# ── Step 2: 気象データ取得（field_id で紐づけ）───────────────────────────────

def load_weather_raw(weather_db: Path,
                     field_ids: list[int],
                     year_start: int,
                     year_end: int,
                     weather_cols: list[str]) -> pd.DataFrame:
    """weather_data テーブルから指定 field_id・年度範囲の気象データを取得。

    ・field_id で圃場と紐づけられています。
    ・全 field_id に field_id が存在することを前提としています。
    ・指定した weather_cols のみ取得します（1981 年から存在する変数のみにも対応）。
    """
    fid_ph = ','.join(['?' for _ in field_ids])
    conn   = sqlite3.connect(weather_db)
    df = pd.read_sql(f'''
        SELECT field_id, date, {", ".join(weather_cols)}
        FROM weather_data
        WHERE field_id IN ({fid_ph})
          AND CAST(SUBSTR(date, 1, 4) AS INTEGER) BETWEEN ? AND ?
        ORDER BY field_id, date
    ''', conn, params=field_ids + [year_start, year_end])
    conn.close()

    df['field_id'] = df['field_id'].astype(int)
    df['date']     = pd.to_datetime(df['date'])
    return df


# ── Step 3: 事前標準化パラメータ計算（特徴量ごと・単変量ごと）────────────────

def compute_norm_stats(weather_df: pd.DataFrame,
                       weather_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """全サンプルの気象データから特徴量ごと（単変量ごと）の mean / std を計算する。

    ・axis=0 で全サンプル・全日付にわたる統計量を列（変数）ごとに算出します。
    ・NaN を除外して計算し、std が極端に小さい場合は 1.0 に置き換えます。
    ・shape: mean (n_feat,), std (n_feat,)

    Returns:
        mean (np.ndarray): shape (n_feat,)  各変数の全期間平均
        std  (np.ndarray): shape (n_feat,)  各変数の全期間標準偏差
    """
    data = weather_df[weather_cols].to_numpy(dtype=np.float64)
    mean = np.nanmean(data, axis=0)  # (n_feat,) 変数ごとの平均
    std  = np.nanstd(data,  axis=0)  # (n_feat,) 変数ごとの標準偏差
    std[std < 1e-8] = 1.0            # ゼロ除算防止
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_gpu(arr, mean, std, xp):
    """(T, F) の配列を標準化する（GPU/CPU 共通）。NaN は 0.0（平均）に置換。

    Args:
        arr  : xp.ndarray shape (T, F)  未標準化の気象データ
        mean : xp.ndarray shape (F,)    各特徴量の平均
        std  : xp.ndarray shape (F,)    各特徴量の標準偏差
        xp   : numpy または cupy
    """
    norm = (arr - mean) / std
    if xp is np:
        return np.nan_to_num(norm, nan=0.0).astype(np.float32)
    else:
        # CuPy: nan_to_num は cupy でも使用可能
        return cp.nan_to_num(norm, nan=0.0).astype(cp.float32)


# ── Step 4: 各圃場の栽培期間テーブルを構築 ───────────────────────────────────

def _compute_field_representative_md(questionaire_df: pd.DataFrame) -> dict:
    """field_id ごとの代表的な播種月日・収穫月日を Questionaire から計算する。

    Questionaire に存在する年（2015〜2018）の seed_date / harvest_date の
    月・日それぞれの中央値を取り、その field_id の代表月日とする。

    Returns:
        {field_id: {'seed': (month, day), 'harvest': (month, day)}}
    """
    field_md = {}
    for fid, grp in questionaire_df.groupby('field_id'):
        fid = int(fid)
        valid_seed    = grp['seed_date'].dropna()
        valid_harvest = grp['harvest_date'].dropna()

        seed_md = DEFAULT_SEED_MD
        if len(valid_seed) > 0:
            s_month = int(np.median(valid_seed.dt.month.values))
            s_day   = int(np.median(valid_seed.dt.day.values))
            seed_md = (s_month, s_day)

        harvest_md = DEFAULT_HARVEST_MD
        if len(valid_harvest) > 0:
            h_month = int(np.median(valid_harvest.dt.month.values))
            h_day   = int(np.median(valid_harvest.dt.day.values))
            harvest_md = (h_month, h_day)

        field_md[fid] = {'seed': seed_md, 'harvest': harvest_md}

    return field_md


def build_period_table(questionaire_df: pd.DataFrame,
                       all_field_ids: list[int],
                       all_years: list[int]) -> pd.DataFrame:
    """(field_id, year) ごとの栽培期間 (start, end, grow_days) を構築する。

    【期間の決定ルール】
      ① Questionaire にその (field_id, year) が存在する場合
          → そのレコードの seed_date / harvest_date を使用
      ② Questionaire にその field_id は存在するが年が異なる場合
          → その field_id の Questionaire 記録から計算した
            「代表播種月日・収穫月日」（各年の中央値）を、
            対象年に適用してクロッピング期間を決定
          ※ これにより field_id=1 の 1981〜2014 年も、
            2015〜2018 年の実績に沿った月日でクロッピングされます。
      ③ その field_id が Questionaire に一切存在しない場合
          → グローバルデフォルト（5/1〜10/31）を使用

    Returns:
        DataFrame with columns:
            field_id, year, start_date, end_date, grow_days, period_source
    """
    # Questionaire データを (field_id, year) でインデックス化
    q_index = {(int(r.field_id), int(r.year)): r
               for _, r in questionaire_df.iterrows()}

    # field_id ごとの代表月日を事前計算
    field_md = _compute_field_representative_md(questionaire_df)

    rows = []
    for fid in all_field_ids:
        for year in all_years:
            key = (fid, year)
            if key in q_index:
                # ① その year の実績値を使用
                r = q_index[key]
                start, end, src = resolve_period(r['seed_date'], r['harvest_date'], year)
            elif fid in field_md:
                # ② その field_id の代表月日を対象年に適用
                seed_md    = field_md[fid]['seed']
                harvest_md = field_md[fid]['harvest']
                try:
                    start = pd.Timestamp(year, *seed_md)
                    end   = pd.Timestamp(year, *harvest_md)
                    src   = 'field_representative'
                except ValueError:
                    # うるう年などで日付が無効な場合はデフォルトにフォールバック
                    start = pd.Timestamp(year, *DEFAULT_SEED_MD)
                    end   = pd.Timestamp(year, *DEFAULT_HARVEST_MD)
                    src   = 'default_date_error'
            else:
                # ③ Questionaire にまったく存在しない field_id → グローバルデフォルト
                start = pd.Timestamp(year, *DEFAULT_SEED_MD)
                end   = pd.Timestamp(year, *DEFAULT_HARVEST_MD)
                src   = 'default_no_questionaire'

            grow_days = (end - start).days + 1
            if grow_days <= 0:
                grow_days = (pd.Timestamp(year, *DEFAULT_HARVEST_MD)
                             - pd.Timestamp(year, *DEFAULT_SEED_MD)).days + 1
                start = pd.Timestamp(year, *DEFAULT_SEED_MD)
                end   = pd.Timestamp(year, *DEFAULT_HARVEST_MD)
                src   = 'default_invalid_period'

            rows.append({
                'field_id':      fid,
                'year':          year,
                'start_date':    start,
                'end_date':      end,
                'grow_days':     grow_days,
                'period_source': src,
            })

    return pd.DataFrame(rows)


# ── Step 5: 動的クロッピング + ゼロパディング（GPU 対応）─────────────────────

def crop_and_pad(weather_by_fid: dict,
                 period_df: pd.DataFrame,
                 mean: np.ndarray,
                 std: np.ndarray,
                 max_len: int,
                 n_feat: int,
                 weather_cols: list[str],
                 use_gpu: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """各サンプルを seed_date〜harvest_date で切り出し、標準化 → ゼロパディングする。

    ・field_id をキーに各圃場の気象データを参照します。
    ・標準化は特徴量（変数）ごとに実施します（mean/std は shape (n_feat,)）。
    ・GPU 利用可能かつ use_gpu=True の場合、CuPy テンソルで処理します。

    Args:
        weather_by_fid : {field_id: DataFrame} 気象データ（未標準化）
        period_df      : build_period_table() の出力
        mean, std      : 事前標準化パラメータ shape (n_feat,)
        max_len        : パディング後のテンソル長
        n_feat         : 特徴量数
        weather_cols   : 使用する気象変数のリスト
        use_gpu        : GPU を使用するかどうか

    Returns:
        X_out    (np.ndarray): shape (N, max_len, n_feat)  float32  （常に CPU numpy）
        meta_out (pd.DataFrame): 各サンプルのメタ情報
    """
    xp = get_xp(use_gpu)
    using_gpu = (xp is not np)

    if using_gpu:
        print(f'  GPU モード: {cp.cuda.Device().id} 番 GPU を使用')
        mean_xp = cp.asarray(mean)  # (n_feat,) on GPU
        std_xp  = cp.asarray(std)   # (n_feat,) on GPU
    else:
        print('  CPU モード (NumPy)')
        mean_xp = mean
        std_xp  = std

    N     = len(period_df)
    # 出力テンソルは GPU 上に確保（後で CPU に転送）
    if using_gpu:
        X_out_gpu = cp.zeros((N, max_len, n_feat), dtype=cp.float32)
    else:
        X_out_gpu = np.zeros((N, max_len, n_feat), dtype=np.float32)

    meta_rows = []
    n_ok      = 0
    n_nodata  = 0
    n_short   = 0

    for i, (_, row) in enumerate(period_df.iterrows()):
        fid   = int(row['field_id'])
        start = row['start_date']
        end   = row['end_date']
        gdays = int(row['grow_days'])

        wdf = weather_by_fid.get(fid)
        if wdf is None:
            # この field_id の気象データが存在しない
            n_nodata += 1
            meta_rows.append({**row.to_dict(), 'actual_days': 0, 'status': 'no_weather'})
            continue

        # seed_date〜harvest_date でスライス（field_id で紐づけ済み）
        mask   = (wdf['date'] >= start) & (wdf['date'] <= end)
        slice_ = wdf.loc[mask, weather_cols].to_numpy(dtype=np.float32)  # (n_days, n_feat)
        n_days = len(slice_)

        if n_days == 0:
            n_nodata += 1
            meta_rows.append({**row.to_dict(), 'actual_days': 0, 'status': 'no_data_in_period'})
            continue

        if n_days < 10:
            n_short += 1

        # GPU に転送して標準化
        if using_gpu:
            slice_gpu = cp.asarray(slice_)
        else:
            slice_gpu = slice_

        slice_norm = standardize_gpu(slice_gpu, mean_xp, std_xp, xp)  # (n_days, n_feat)

        # 先頭から代入（Post-padding: 残りは 0 のまま）
        copy_len = min(n_days, max_len)
        X_out_gpu[i, :copy_len, :] = slice_norm[:copy_len]

        n_ok += 1
        meta_rows.append({**row.to_dict(), 'actual_days': n_days, 'status': 'ok'})

    # GPU → CPU に転送
    if using_gpu:
        X_out = cp.asnumpy(X_out_gpu)
    else:
        X_out = X_out_gpu

    meta_out = pd.DataFrame(meta_rows)

    print(f'  成功: {n_ok} / {N} サンプル')
    if n_nodata > 0:
        print(f'  気象データなし: {n_nodata} 件 → ゼロ行列のまま')
    if n_short > 0:
        print(f'  栽培期間 10日未満: {n_short} 件（警告）')

    return X_out, meta_out


# ── メイン処理 ───────────────────────────────────────────────────────────────

def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # GPU 使用可否の決定
    use_gpu = (not args.no_gpu) and _CUPY_AVAILABLE

    # 気象変数の選択
    if args.cols == '1981':
        weather_cols = WEATHER_COLS_1981
        print('気象変数: 1981 年から存在する変数のみ使用')
    else:
        weather_cols = WEATHER_COLS
        print('気象変数: 全 9 変数使用')
    n_feat = len(weather_cols)

    print('=' * 60)
    print('  事前学習データセット生成 v2（動的クロッピング版）')
    print('=' * 60)
    if use_gpu:
        print(f'  GPU: CuPy 利用可 → GPU モードで実行')
    else:
        if _CUPY_AVAILABLE and args.no_gpu:
            print('  GPU: --no-gpu 指定 → CPU モードで実行')
        else:
            print('  GPU: CuPy 未インストール → CPU (NumPy) で実行')
    print(f'  気象変数: {weather_cols}')
    print(f'  変数数  : {n_feat}')

    # ── Step 1: Questionaire から播種日・収穫日取得 ──────────────────────
    print('\n[Step 1] Questionaire 読み込み...')
    q_df = load_questionaire(args.field_db)
    print(f'  {len(q_df)} サンプル')
    print(f'  seed_date 欠損  : {q_df["seed_date"].isna().sum()} 件')
    print(f'  harvest_date 欠損: {q_df["harvest_date"].isna().sum()} 件')

    # ── Step 2: 気象データ取得（field_id で紐づけ）──────────────────────
    print(f'\n[Step 2] 気象データ取得 ({args.year_start}〜{args.year_end})...')
    print(f'  ※ すべての気象データは field_id を持ちます。field_id で圃場と紐づけます。')
    all_fids = sorted(q_df['field_id'].unique().tolist())
    weather_df = load_weather_raw(args.weather_db, all_fids,
                                  args.year_start, args.year_end,
                                  weather_cols)
    print(f'  {len(weather_df):,} 行取得  '
          f'({weather_df["field_id"].nunique()} 圃場 × '
          f'{weather_df["date"].dt.year.nunique()} 年)')

    if args.dry_run:
        print('\n[dry-run] ここで終了。')
        return

    # ── Step 3: 事前標準化パラメータ計算（特徴量ごと・単変量ごと）────────
    print('\n[Step 3] 事前標準化パラメータ計算（特徴量ごと・単変量ごと）...')
    print(f'  ※ nanmean / nanstd を axis=0 で計算 → 各変数独立に標準化')
    mean, std = compute_norm_stats(weather_df, weather_cols)
    print(f'  {"変数":<12} {"mean":>9} {"std":>9}')
    print(f'  {"-" * 32}')
    for i, col in enumerate(weather_cols):
        print(f'  {col:<12} {mean[i]:>9.3f} {std[i]:>9.3f}')

    # ── Step 4: 栽培期間テーブル構築 ────────────────────────────────────
    print('\n[Step 4] 栽培期間テーブル構築...')
    print(f'  ※ seed_date / harvest_date をもとにクロッピング期間を決定')
    all_years  = sorted(weather_df['date'].dt.year.unique().tolist())
    period_df  = build_period_table(q_df, all_fids, all_years)
    print(f'  総サンプル数  : {len(period_df):,}  '
          f'({len(all_fids)} 圃場 × {len(all_years)} 年)')

    # 栽培日数の統計
    gd = period_df['grow_days']
    print(f'  栽培日数統計  : min={gd.min()}  median={gd.median():.0f}  '
          f'mean={gd.mean():.1f}  max={gd.max()}')

    # max_len を動的決定
    max_len = int(gd.max())
    print(f'  max_len (動的): {max_len} 日')

    # 期間ソース内訳
    src_counts = period_df['period_source'].value_counts()
    print('  期間ソース内訳:')
    for src, cnt in src_counts.items():
        print(f'    {src:<30}: {cnt:,} 件')

    # ── Step 5: 動的クロッピング + ゼロパディング（GPU 対応）────────────
    print(f'\n[Step 5] 動的クロッピング + ゼロパディング (max_len={max_len})...')
    weather_by_fid = {fid: grp.reset_index(drop=True)
                      for fid, grp in weather_df.groupby('field_id')}

    X_out, meta_out = crop_and_pad(
        weather_by_fid, period_df, mean, std,
        max_len, n_feat, weather_cols, use_gpu
    )
    print(f'  X_out.shape : {X_out.shape}')
    print(f'  ゼロ率      : {(X_out == 0).mean() * 100:.1f}%  '
          f'（パディング0 + 標準化後に本来0の値を含む）')

    # ── Step 6: 保存 ────────────────────────────────────────────────────
    print('\n[Step 6] 保存中...')

    # メタデータ列を整形
    meta_out['start_date'] = meta_out['start_date'].astype(str)
    meta_out['end_date']   = meta_out['end_date'].astype(str)

    # パディング情報を追加（CSVから確認できるように）
    #   max_len     : 全サンプル共通のテンソル長（最大grow_days）
    #   padded_days : 0パディングされたステップ数 = max_len - actual_days
    #                 （actual_days が有効データ、残りは 0 で埋められている）
    meta_out['max_len']     = max_len
    meta_out['padded_days'] = meta_out['actual_days'].apply(
        lambda d: max(0, max_len - int(d)) if d > 0 else max_len
    )

    # 保存先
    x_path    = OUT_DIR / 'pretrain_X_v2.npy'
    meta_path = OUT_DIR / 'pretrain_meta_v2.csv'
    norm_path = OUT_DIR / 'pretrain_norm_stats.npy'

    np.save(x_path, X_out)
    np.save(norm_path, np.stack([mean, std], axis=0))  # shape: (2, n_feat)
    meta_out.to_csv(meta_path, index=False)

    size_mb = x_path.stat().st_size / 1024 ** 2

    # ── サマリー ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print('  生成完了')
    print(f'{"=" * 60}')
    print(f'  pretrain_X_v2.npy')
    print(f'    shape   : {X_out.shape}  (N, max_len, features)')
    print(f'    dtype   : {X_out.dtype}')
    print(f'    サイズ  : {size_mb:.1f} MB')
    print(f'  pretrain_norm_stats.npy')
    print(f'    shape   : (2, {n_feat})  [mean, std] per feature')
    print(f'  pretrain_meta_v2.csv : {len(meta_out):,} 行')
    print(f'  max_len（動的）      : {max_len} 日')
    print(f'  気象変数             : {weather_cols}')
    print(f'  総処理時間           : {elapsed:.1f} 秒')
    print(f'\n  出力ディレクトリ: {OUT_DIR}')
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in ('.npy', '.csv'):
            mb = f.stat().st_size / 1024 ** 2
            print(f'    {f.name:<40} {mb:7.1f} MB')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='ts2vec 事前学習データセット生成（動的クロッピング版）')
    p.add_argument('--field-db',   type=Path, default=FIELD_DB,   dest='field_db')
    p.add_argument('--weather-db', type=Path, default=WEATHER_DB, dest='weather_db')
    p.add_argument('--out-dir',    type=Path, default=OUT_DIR,    dest='out_dir')
    p.add_argument('--year-start', type=int,  default=PRETRAIN_YEAR_START,
                   dest='year_start', help=f'事前学習開始年 (default: {PRETRAIN_YEAR_START})')
    p.add_argument('--year-end',   type=int,  default=PRETRAIN_YEAR_END,
                   dest='year_end',   help=f'事前学習終了年 (default: {PRETRAIN_YEAR_END})')
    p.add_argument('--dry-run',    action='store_true',
                   help='気象データ取得まで実行して終了（配列生成・保存を行わない）')
    p.add_argument('--no-gpu',     action='store_true',
                   help='CPU (NumPy) 強制モード。GPU (CuPy) を使わない。')
    p.add_argument('--cols',       type=str, default='all',
                   choices=['all', '1981'],
                   help=(
                       'all: 全 9 変数を使用 (default) | '
                       '1981: 1981 年から確実に存在する変数のみ使用'
                   ))
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
