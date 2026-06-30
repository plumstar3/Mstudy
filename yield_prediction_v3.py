"""
yield_prediction_v3.py  (v3.0 — 累積GDD期間分割対応)
============================================================
【変更点（v2.2 からの差分）】
  - 特徴量抽出方法を「固定3期間（栽培期間の1/3ずつ）」から
    「累積GDD閾値に基づく動的3期間」に変更
      期間1: 播種日 〜 累積GDD が 600 を超える日の前日
      期間2: 累積GDD > 600 〜 累積GDD が 1000 を超える日の前日
      期間3: 累積GDD > 1000 〜 収穫日
  - 各期間の気象変数統計量は「平均値」のみ
      特徴量次元: 9変数 × 3期間 × 1統計 = 27次元（+ geo 2次元）

【GDD 期間ソース】
  outputs/gdd/gdd_daily.csv
    (field_id, year, date, 累積GDD 列を使用)

【気象データ取得先】
  weather_database_fieldid.db の weather_data テーブル
    (field_id, date, TMP_mea, TMP_max, TMP_min, APCPRA, SSD, GSR, WIND, SWE, RH)

【CV方式】
  kfold  : KFold(n_splits=5, shuffle=True, random_state=42)
  loyo   : Leave-One-Year-Out

【モデル】
  Ridge(alpha=100)  /  LightGBM(n_est=200, lr=0.05, leaves=31)

【使い方】
  python yield_prediction_v3.py --cv-mode kfold
  python yield_prediction_v3.py --cv-mode loyo
  python yield_prediction_v3.py --cv-mode kfold --add-geo
  python yield_prediction_v3.py --cv-mode kfold --iqr
"""

import argparse
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# ── 定数 ──────────────────────────────────────────────────────────────────────

WEATHER_DB  = os.path.join('data', 'processed', 'weather_database_fieldid.db')
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
GDD_CSV     = os.path.join('outputs', 'gdd', 'gdd_daily.csv')
OUTPUT_DIR  = os.path.join('outputs', 'yield_pred_v3')

WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
#WEATHER_COLS = ['TMP_mea', 'APCPRA',  'GSR', 'WIND']
N_VARS   = len(WEATHER_COLS)   # 9

# 累積GDD 期間閾値
GDD_THRESHOLDS = [600, 1000]   # [th1, th2] → 期間1: [0,th1], 期間2: (th1,th2], 期間3: (th2,∞)
N_PERIODS      = 3

# 統計量（v2 と同じ 5 種）
STAT_FUNCS = ['mean', 'std', 'min', 'max', 'median']
#STAT_FUNCS = ['mean']
N_STATS    = len(STAT_FUNCS)   # 5

# 特徴量次元: 9変数 × 3期間 × 5統計 = 135
N_FEATURES_GDD = N_VARS * N_PERIODS * N_STATS

TARGET_YEARS  = [2015, 2016, 2017, 2018]

# Harm テーブル追加特徴量
# typhoon: 'TRUE'/'FALSE' 文字列 → 1/0 に変換
# sick/weed/wet/unripen: 1〜4 の順序尺度（欠損は most_frequent で補完）
HARM_COLS = ['sick', 'weed', 'wet', 'typhoon', 'unripen']

# 作付管理特徴量
# between_lines  : 条間距離 [cm] — 欠損率 3.5%  → 固定値 75 で補完
# between_stocks : 株間距離 [cm] — 欠損率 29.5% + 文字列混入 → 固定値 18 で補完
SPACING_COLS = ['between_lines', 'between_stocks']
SPACING_FILL = {'between_lines': 75.0, 'between_stocks': 18.0}

# 土壌水分特徴量 (SolidMoisture テーブル)
# VWC_mean  : 測定期間全体の平均VWC —— 欠損の場合: 同一field_idの他年平均 → 全体平均
# has_vwc   : VWCセンサーが設置されていた場合 1，ない場合 0
VWC_COLS = ['VWC_mean', 'has_vwc']

# ベースライン固定ハイパーパラメータ
RIDGE_ALPHA       = 100
LGBM_N_ESTIMATORS = 200
LGBM_LR           = 0.05
LGBM_NUM_LEAVES   = 31

N_SPLITS      = 5
RANDOM_STATE  = 42
PCA_N_DEFAULT = 30
FIXED_TEST_YEAR = 2018


# ── Step 1: gdd_daily.csv を読み込んで期間ラベルを付与 ─────────────────────────

def load_gdd_with_periods(gdd_csv: str) -> pd.DataFrame:
    """gdd_daily.csv を読み込み、累積GDD に基づいて期間ラベルを付与する。

    期間1: 累積GDD ≤ 600
    期間2: 600 < 累積GDD ≤ 1000
    期間3: 累積GDD > 1000

    Returns:
        DataFrame: field_id, year, date, 累積GDD, period (1/2/3)
    """
    df = pd.read_csv(gdd_csv, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])

    # 累積GDD 列名（日本語 or ASCII どちらでも対応）
    cum_col = '累積GDD' if '累積GDD' in df.columns else 'cumulative_gdd'
    if cum_col not in df.columns:
        raise ValueError(f'累積GDD 列が見つかりません。列: {df.columns.tolist()}')

    th1, th2 = GDD_THRESHOLDS
    df['period'] = 1
    df.loc[df[cum_col] > th1, 'period'] = 2
    df.loc[df[cum_col] > th2, 'period'] = 3

    return df[['field_id', 'year', 'date', cum_col, 'period']].rename(
        columns={cum_col: '累積GDD'}
    )


# ── Step 2: 気象データ取得 ────────────────────────────────────────────────────

def load_weather(weather_db: str, field_ids: list[int], years: list[int]) -> pd.DataFrame:
    """weather_data テーブルから 9 気象変数を一括取得する。"""
    fid_ph  = ','.join(['?'] * len(field_ids))
    yr_ph   = ','.join([f"'{y}'" for y in years])
    col_str = ', '.join(WEATHER_COLS)
    conn    = sqlite3.connect(weather_db)
    df = pd.read_sql(f'''
        SELECT field_id, date, {col_str}
        FROM weather_data
        WHERE field_id IN ({fid_ph})
          AND CAST(SUBSTR(date, 1, 4) AS INTEGER) IN ({yr_ph})
        ORDER BY field_id, date
    ''', conn, params=field_ids)
    conn.close()
    df['field_id'] = df['field_id'].astype(int)
    df['date']     = pd.to_datetime(df['date'])
    return df


# ── Step 3: 収量データ取得 ────────────────────────────────────────────────────

def load_questionaire(field_db: str) -> pd.DataFrame:
    """Questionaire テーブルから field_id, year, yield, lat, lon を取得。"""
    conn = sqlite3.connect(field_db)
    df = pd.read_sql('''
        SELECT field_id, year, yield, lat, lon
        FROM Questionaire
        WHERE field_id IS NOT NULL AND yield IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()
    df['field_id'] = df['field_id'].astype(int)
    df['year']     = df['year'].astype(int)
    df['yield']    = df['yield'].astype(float)
    df['lat']      = pd.to_numeric(df['lat'], errors='coerce')
    df['lon']      = pd.to_numeric(df['lon'], errors='coerce')
    return df.reset_index(drop=True)


def load_harm(field_db: str) -> pd.DataFrame:
    """Harm テーブルから sick, weed, wet, typhoon, unripen を取得して数値化する。

    変換ルール:
      typhoon : 'TRUE' -> 1, 'FALSE' -> 0  (欠損なし)
      sick / weed / wet / unripen : 順序尺度の整数値（欠損は NaN のまま残す）
                                    → パイプライン内の SimpleImputer で最頻値補完
    """
    conn = sqlite3.connect(field_db)
    col_str = ', '.join(HARM_COLS)
    df = pd.read_sql(f'''
        SELECT field_id, year, {col_str}
        FROM Harm
        WHERE field_id IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()

    df['field_id'] = df['field_id'].astype(int)
    df['year']     = df['year'].astype(int)

    # typhoon: 文字列 TRUE/FALSE → 1/0
    df['typhoon'] = df['typhoon'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    df['typhoon'] = pd.to_numeric(df['typhoon'], errors='coerce')

    # 残りの列: float に変換（欠損は NaN）
    for col in ['sick', 'weed', 'wet', 'unripen']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f'  Harm 読み込み: {len(df)} 行')
    for col in HARM_COLS:
        n_miss = df[col].isna().sum()
        print(f'    {col}: 欠損 {n_miss} 件 ({n_miss/len(df)*100:.1f}%)')

    return df.reset_index(drop=True)


# ── 作付管理特徴量: 条間・株間（固定値補完） ─────────────────────────────────

def load_spacing(field_db: str) -> pd.DataFrame:
    """Questionaire テーブルから between_lines / between_stocks を取得して固定値補完する。

    補完ルール:
      between_lines  : 非数値または NaN → 75.0 で補完（全体最頻値）
      between_stocks : 文字列または NaN → 18.0 で補完（全体中央値と最頻値の近傍）

    Returns:
        DataFrame: field_id, year, between_lines, between_stocks（補完済み）
    """
    conn = sqlite3.connect(field_db)
    df = pd.read_sql('''
        SELECT field_id, year, between_lines, between_stocks
        FROM Questionaire
        WHERE field_id IS NOT NULL AND yield IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()

    df['field_id'] = df['field_id'].astype(int)
    df['year']     = df['year'].astype(int)

    # between_lines: 数値変換し、NaN → 75
    df['between_lines'] = pd.to_numeric(df['between_lines'], errors='coerce')
    n_miss_l = df['between_lines'].isna().sum()
    df['between_lines'] = df['between_lines'].fillna(SPACING_FILL['between_lines'])

    # between_stocks: 文字列は先に coerce で NaN 化し、NaN → 18
    df['between_stocks'] = pd.to_numeric(df['between_stocks'], errors='coerce')
    n_miss_s = df['between_stocks'].isna().sum()
    df['between_stocks'] = df['between_stocks'].fillna(SPACING_FILL['between_stocks'])

    print(f'  between_lines  : 欠損/非数値 {n_miss_l} 件 → {SPACING_FILL["between_lines"]} に補完')
    print(f'  between_stocks : 欠損/文字列 {n_miss_s} 件 → {SPACING_FILL["between_stocks"]} に補完')

    return df[['field_id', 'year'] + SPACING_COLS].reset_index(drop=True)


# ── 土壌水分特徴量: VWC（SolidMoisture） ───────────────────────────────────────

def load_vwc(field_db: str) -> pd.DataFrame:
    """SolidMoisture テーブルから VWC を取得し、(field_id, year) 単位の平均VWCと
    has_vwc フラグを返す。

    【補完ロジック】
      プライオリティ順に以下を適用する。
      1. (field_id, year) にデータあり → その年の測定期間全体の平均
      2. field_id はあるが対象年にデータなし
         （例: 2015年は全圆場がない）→ その field_id の全年平均
      3. field_id 自体が SolidMoisture にない → 全体平均 (0.347)

    注意:
      has_vwc = 1 : その field_id が SolidMoisture に少なくとも 1年分のデータあり
      has_vwc = 0 : SolidMoisture に全く記録なし

    Returns:
        DataFrame: field_id, year, VWC_mean, has_vwc
    """
    conn = sqlite3.connect(field_db)
    sm  = pd.read_sql('SELECT field_id, year, VWC FROM SolidMoisture', conn)
    qst = pd.read_sql('''
        SELECT CAST(field_id AS INTEGER) AS field_id, year
        FROM Questionaire
        WHERE field_id IS NOT NULL AND yield IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()

    sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce').astype('Int64')
    sm['year']     = pd.to_numeric(sm['year'], errors='coerce').astype('Int64')
    sm['VWC']      = pd.to_numeric(sm['VWC'], errors='coerce')
    sm = sm.dropna(subset=['field_id', 'VWC'])

    # 全体平均 VWC（最終フォールバック）
    global_mean = float(sm['VWC'].mean())

    # field_id 別平均 VWC（年度跨ぎ；2015年などのフォールバック用）
    field_mean = sm.groupby('field_id')['VWC'].mean().to_dict()

    # (field_id, year) 別平均 VWC
    field_year_mean = (
        sm.groupby(['field_id', 'year'])['VWC'].mean()
    ).to_dict()  # key = (fid, yr)

    # VWC 測定有りの field_id 集合（かつ has_vwc フラグの山）
    vwc_fids = set(sm['field_id'].dropna().astype(int))

    # Questionaire の全 (field_id, year) に対して補完適用
    qst['field_id'] = qst['field_id'].astype(int)
    qst['year']     = qst['year'].astype(int)

    vwc_list, has_vwc_list = [], []
    n_case = {1: 0, 2: 0, 3: 0}  # 補完ケースのカウント

    for _, row in qst.iterrows():
        fid = int(row['field_id'])
        yr  = int(row['year'])
        key = (fid, yr)

        if key in field_year_mean:              # Case 1: その年のデータあり
            v = field_year_mean[key]
            n_case[1] += 1
        elif fid in field_mean:                 # Case 2: 別年のデータで補完
            v = field_mean[fid]
            n_case[2] += 1
        else:                                   # Case 3: 全体平均
            v = global_mean
            n_case[3] += 1

        vwc_list.append(v)
        has_vwc_list.append(1 if fid in vwc_fids else 0)

    qst = qst.copy()
    qst['VWC_mean'] = vwc_list
    qst['has_vwc']  = has_vwc_list

    print(f'  VWC 補完内訳:')
    print(f'    Case1 (当年データあり)     : {n_case[1]} 件')
    print(f'    Case2 (別年データで補完)  : {n_case[2]} 件')
    print(f'    Case3 (全体平均フォールバック): {n_case[3]} 件  (global_mean={global_mean:.4f})')
    print(f'    has_vwc=1: {sum(has_vwc_list)} 件  has_vwc=0: {len(has_vwc_list)-sum(has_vwc_list)} 件')

    return qst[['field_id', 'year'] + VWC_COLS].reset_index(drop=True)


# ── Step 4: GDD期間ごとの気象特徴量を構築 ────────────────────────────────────

def build_gdd_features(gdd_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """GDD 期間（1/2/3）ごとの気象変数 5 統計量を計算し、特徴量行列を返す。

    処理:
      1. gdd_df（期間ラベル付き）と weather_df を (field_id, date) でマージ
      2. (field_id, year, period) でグループ化し各変数の 5 統計を計算
         統計量: mean, std, min, max, median
      3. period をピボットして 9×3×5=135 次元の特徴量に変換

    Returns:
        DataFrame: index=(field_id, year), columns=変数名_p1_mean 等 （135列）
                   空の期間は NaN で埋める（SimpleImputer が後段で補完）
    """
    # ① マージ：gdd の各日に気象値を付与
    merged = gdd_df.merge(
        weather_df[['field_id', 'date'] + WEATHER_COLS],
        on=['field_id', 'date'],
        how='left'
    )

    # ② グループ集計: (field_id, year, period) × 変数 × 5統計
    # agg の結果: MultiIndex columns (var, stat)
    grp = (merged
           .groupby(['field_id', 'year', 'period'])[WEATHER_COLS]
           .agg(STAT_FUNCS))  # shape: (n_groups, 9×5)

    # ③ ピボット: period をカラムに展開
    # unstack('period') → MultiIndex columns (var, stat, period)
    grp_pivot = grp.unstack('period')  # shape: (n_fid_year, 9×5×3)

    # ④ カラム名をフラット化: 'TMP_mea_p1_mean', 'TMP_mea_p1_std', ...
    grp_pivot.columns = [
        f'{var}_p{int(period)}_{stat}'
        for var, stat, period in grp_pivot.columns
    ]

    # ⑤ 欠損期間（存在しない期間は NaN）→ SimpleImputer が補完
    for p in [1, 2, 3]:
        for var in WEATHER_COLS:
            for stat in STAT_FUNCS:
                col = f'{var}_p{p}_{stat}'
                if col not in grp_pivot.columns:
                    grp_pivot[col] = np.nan

    # 列順を統一: var_p1_mean, var_p1_std, ..., var_p3_median の順
    ordered_cols = [
        f'{var}_p{p}_{stat}'
        for p in [1, 2, 3]
        for var in WEATHER_COLS
        for stat in STAT_FUNCS
    ]
    grp_pivot = grp_pivot[ordered_cols]

    return grp_pivot.reset_index()  # field_id, year, 135 特徴量列


# ── Step 5: データセット構築 ──────────────────────────────────────────────────

def build_dataset(field_db: str, weather_db: str, gdd_csv: str,
                  add_harm: bool = False,
                  add_spacing: bool = False,
                  add_vwc: bool = False):
    """GDD 期間分割に基づく特徴量 X と目的変数 y を構築する。

    Args:
        add_harm    : True のとき Harm テーブルの 5 変数を追加する。
        add_spacing : True のとき between_lines / between_stocks を追加する。
        add_vwc     : True のとき VWC_mean / has_vwc を追加する。

    Returns:
        X    (N, 135 [+5] [+2] [+2])  特徴量行列
        y    (N,)                     収量
        geo  (N, 2)                   lat, lon
        meta DataFrame
        feat_cols list
    """
    # ── 収量データ ─────────────────────────────────────────────────────────
    print('Questionaire 読み込み...', end=' ')
    quest_df = load_questionaire(field_db)
    print(f'{len(quest_df)} サンプル')

    # ── GDD期間ラベル ───────────────────────────────────────────────────────
    print('gdd_daily.csv 読み込み...', end=' ')
    gdd_df = load_gdd_with_periods(gdd_csv)
    print(f'{len(gdd_df):,} 行')

    # ── 有効な (field_id, year) の絞り込み ─────────────────────────────────
    # quest_df と gdd_df の両方に存在するものだけを使う
    q_keys   = set(zip(quest_df['field_id'], quest_df['year']))
    g_keys   = set(zip(gdd_df['field_id'],   gdd_df['year']))
    valid_fy = q_keys & g_keys
    if not valid_fy:
        raise RuntimeError('収量データと GDD データに共通の (field_id, year) がありません。')

    fids  = sorted(set(f for f, _ in valid_fy))
    years = sorted(set(y for _, y in valid_fy))

    # ── 気象データ ─────────────────────────────────────────────────────────
    print(f'気象データ読み込み ({len(fids)} 圃場)...', end=' ')
    weather_df = load_weather(weather_db, fids, years)
    print(f'{len(weather_df):,} 行')

    # ── GDD期間ごとの特徴量計算 ────────────────────────────────────────────
    print('GDD期間別気象特徴量計算...')
    feat_df = build_gdd_features(gdd_df, weather_df)
    # 135 特徴量列名 (var_p{p}_{stat})
    feat_cols = [
        f'{var}_p{p}_{stat}'
        for p in [1, 2, 3]
        for var in WEATHER_COLS
        for stat in STAT_FUNCS
    ]

    # ── 結合: 収量 + 特徴量 ───────────────────────────────────────────────
    merged = quest_df.merge(feat_df, on=['field_id', 'year'], how='inner')
    print(f'  結合後サンプル数: {len(merged)}')

    # 期間カバレッジ情報（どの期間まで GDD が到達したか）
    max_period = (gdd_df.groupby(['field_id', 'year'])['period']
                  .max()
                  .reset_index()
                  .rename(columns={'period': 'max_period'}))
    merged = merged.merge(max_period, on=['field_id', 'year'], how='left')

    period_counts = merged['max_period'].value_counts().sort_index()
    print(f'  到達期間の内訳:')
    for p, cnt in period_counts.items():
        label = {1: f'期間1のみ (GDD < {GDD_THRESHOLDS[0]})',
                 2: f'期間2まで ({GDD_THRESHOLDS[0]} <= GDD < {GDD_THRESHOLDS[1]})',
                 3: f'期間3まで (GDD >= {GDD_THRESHOLDS[1]})'}.get(int(p), str(p))
        print(f'    {label}: {cnt} サンプル')

    # ── Harm 特徴量の追加（任意） ──────────────────────────────────────────
    if add_harm:
        print('Harm 特徴量読み込み...')
        harm_df = load_harm(field_db)
        merged  = merged.merge(harm_df[['field_id', 'year'] + HARM_COLS],
                               on=['field_id', 'year'], how='left')
        all_feat_cols = feat_cols + HARM_COLS
        print(f'  Harm 追加後サンプル数: {len(merged)}  特徴量次元: {len(all_feat_cols)}')
    else:
        all_feat_cols = feat_cols

    if add_spacing:
        print('条間・株間特徴量読み込み...')
        spacing_df = load_spacing(field_db)
        merged = merged.merge(spacing_df[['field_id', 'year'] + SPACING_COLS],
                              on=['field_id', 'year'], how='left')
        all_feat_cols = all_feat_cols + SPACING_COLS
        print(f'  Spacing 追加後サンプル数: {len(merged)}  特徴量次元: {len(all_feat_cols)}')

    if add_vwc:
        print('VWC (土壌水分) 特徴量読み込み...')
        vwc_df = load_vwc(field_db)
        merged = merged.merge(vwc_df[['field_id', 'year'] + VWC_COLS],
                              on=['field_id', 'year'], how='left')
        all_feat_cols = all_feat_cols + VWC_COLS
        print(f'  VWC 追加後サンプル数: {len(merged)}  特徴量次元: {len(all_feat_cols)}')

    X   = merged[all_feat_cols].to_numpy(dtype=np.float32)
    y   = merged['yield'].to_numpy(dtype=np.float32)
    geo = merged[['lat', 'lon']].to_numpy(dtype=np.float32)
    meta = merged[['field_id', 'year', 'yield', 'max_period']].reset_index(drop=True)

    return X, y, geo, meta, all_feat_cols


# ── IQR 外れ値除去 ────────────────────────────────────────────────────────────

def apply_iqr(X, y, geo, meta):
    q1  = float(np.percentile(y, 25))
    q3  = float(np.percentile(y, 75))
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    keep = (y >= lb) & (y <= ub)
    print(f'\nIQR 外れ値除去: Q1={q1:.1f} Q3={q3:.1f} 範囲=[{lb:.1f}, {ub:.1f}]')
    print(f'  除外 {(~keep).sum()} 件 → 残り {keep.sum()} 件')
    return (X[keep], y[keep], geo[keep],
            meta[keep].reset_index(drop=True))


# ── 評価指標 ──────────────────────────────────────────────────────────────────

def calc_metrics(pred, target):
    rmse   = float(np.sqrt(mean_squared_error(target, pred)))
    mae    = float(np.abs(pred - target).mean())
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    nz     = np.abs(target) > 0
    mape   = float(np.mean(np.abs((pred[nz] - target[nz]) / target[nz])) * 100) \
             if nz.any() else float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


# ── 予測値 vs 実測値 散布図 ───────────────────────────────────────────────────

def plot_pred_vs_actual(pred_records, cv_label, pca_label, output_dir):
    model_names = sorted(set(r['model'] for r in pred_records))
    fold_labels  = sorted(set(r['fold'] for r in pred_records), key=str)
    n_models     = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6),
                             facecolor='#f8f9fa')
    if n_models == 1:
        axes = [axes]

    n_colors   = max(len(fold_labels), 1)
    palette    = [cm.tab10(i / max(n_colors - 1, 1)) for i in range(n_colors)]
    fold_color = {fl: palette[i] for i, fl in enumerate(fold_labels)}

    for ax, model_name in zip(axes, model_names):
        recs     = [r for r in pred_records if r['model'] == model_name]
        all_true = np.concatenate([r['y_true'] for r in recs])
        all_pred = np.concatenate([r['y_pred'] for r in recs])

        for fold_label in fold_labels:
            fold_recs = [r for r in recs if r['fold'] == fold_label]
            if not fold_recs:
                continue
            yt = np.concatenate([r['y_true'] for r in fold_recs])
            yp = np.concatenate([r['y_pred'] for r in fold_recs])
            ax.scatter(yt, yp, alpha=0.65, s=45,
                       color=fold_color[fold_label],
                       label=str(fold_label), zorder=3,
                       edgecolors='white', linewidths=0.4)

        lims_min = min(all_true.min(), all_pred.min())
        lims_max = max(all_true.max(), all_pred.max())
        margin   = (lims_max - lims_min) * 0.06
        lims     = [lims_min - margin, lims_max + margin]
        ax.plot(lims, lims, '--', color='#333333', lw=1.5,
                label='Perfect prediction', zorder=2)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        rmse   = float(np.sqrt(((all_pred - all_true) ** 2).mean()))
        mae    = float(np.abs(all_pred - all_true).mean())
        ss_res = ((all_true - all_pred) ** 2).sum()
        ss_tot = ((all_true - all_true.mean()) ** 2).sum()
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        textstr = f'RMSE = {rmse:.3f}\nMAE  = {mae:.3f}\nR2   = {r2:.4f}'
        ax.text(0.04, 0.96, textstr, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.87, edgecolor='#cccccc'))

        ax.set_xlabel('Actual Yield', fontsize=12)
        ax.set_ylabel('Predicted Yield', fontsize=12)
        ax.set_title(model_name, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=9, title='Fold / Year', loc='lower right', framealpha=0.85)
        ax.set_facecolor('#fdfdfd')
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle(f'Predicted vs Actual Yield\n[{cv_label}  /  {pca_label}]',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    def _safe(s):
        return str(s).replace('(', '').replace(')', '').replace(' ', '_')

    fpath = os.path.join(output_dir, f'pred_vs_actual_{_safe(cv_label)}_{_safe(pca_label)}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 散布図保存: {fpath}')


# ── モデル定義 ────────────────────────────────────────────────────────────────

def make_models(use_pca=False, pca_n=PCA_N_DEFAULT):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer

    def _build(model_obj):
        steps = [
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler',  StandardScaler()),
        ]
        if use_pca:
            steps.append(('pca', PCA(n_components=pca_n, random_state=RANDOM_STATE)))
        steps.append(('model', model_obj))
        return Pipeline(steps)

    ridge = _build(Ridge(alpha=RIDGE_ALPHA))
    lgbm  = _build(lgb.LGBMRegressor(
        n_estimators=LGBM_N_ESTIMATORS, learning_rate=LGBM_LR,
        num_leaves=LGBM_NUM_LEAVES, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    ))
    return {'Ridge': ridge, 'LightGBM': lgbm}


# ── fold ループ共通処理 ───────────────────────────────────────────────────────

def _run_folds(X, y, splits_iter, n_folds_label, models,
               pred_store=None, fold_labels=None):
    metrics = {name: [] for name in models}

    for fold_idx, (tr_idx, va_idx) in enumerate(splits_iter):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        fold_label = fold_labels[fold_idx] if fold_labels is not None else fold_idx + 1
        print(f'  Fold {fold_idx + 1}  (train={len(y_tr)} val={len(y_va)})')

        for model_name, pipeline in models.items():
            t0   = time.time()
            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            m    = calc_metrics(pred, y_va)
            metrics[model_name].append(m)
            if pred_store is not None:
                pred_store.append({
                    'model':  model_name,
                    'fold':   fold_label,
                    'y_true': y_va.copy(),
                    'y_pred': pred.copy(),
                })
            print(f'    {model_name:<10} RMSE={m["RMSE"]:7.3f}  MAE={m["MAE"]:7.3f}  '
                  f'MAPE={m["MAPE"]:6.2f}%  R2={m["R2"]:7.4f}  ({time.time()-t0:.1f}s)')

    return metrics


def _print_summary(metrics, cv_label, use_pca, pca_n):
    pca_label = f'PCA({pca_n}d)' if use_pca else 'NoPCA'
    print(f'\n{"=" * 65}')
    print(f'  SUMMARY  [{cv_label} / {pca_label}]  mean ± std')
    print(f'{"=" * 65}')
    summary_rows = []
    for model_name, fold_data in metrics.items():
        stats = {k: (np.mean([f[k] for f in fold_data]),
                     np.std( [f[k] for f in fold_data]))
                 for k in ('RMSE', 'MAE', 'MAPE', 'R2')}
        print(f'  {model_name:<12} '
              f'RMSE={stats["RMSE"][0]:>7.3f}±{stats["RMSE"][1]:<5.3f}  '
              f'MAE={stats["MAE"][0]:>7.3f}±{stats["MAE"][1]:<5.3f}  '
              f'MAPE={stats["MAPE"][0]:>5.2f}%  '
              f'R2={stats["R2"][0]:>6.4f}±{stats["R2"][1]:.4f}')
        for fold_idx, m in enumerate(fold_data):
            summary_rows.append({'cv_mode': cv_label, 'pca': use_pca,
                                  'model': model_name, 'fold': fold_idx + 1, **m})
    print(f'{"=" * 65}')
    return summary_rows


# ── CV方式ごとの実行関数 ──────────────────────────────────────────────────────

def run_kfold(X, y, use_pca, pca_n, output_dir=None):
    print(f'\n{"─" * 65}')
    print(f'  [5-Fold CV]  PCA={"あり(" + str(pca_n) + "d)" if use_pca else "なし"}')
    print(f'{"─" * 65}')
    kf         = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models     = make_models(use_pca=use_pca, pca_n=pca_n)
    pred_store = [] if output_dir else None
    metrics    = _run_folds(X, y, kf.split(X), N_SPLITS, models, pred_store=pred_store)
    rows       = _print_summary(metrics, 'kfold', use_pca, pca_n)
    if output_dir and pred_store:
        pca_label = f'PCA({pca_n}d)' if use_pca else 'NoPCA'
        plot_pred_vs_actual(pred_store, 'kfold', pca_label, output_dir)
    return rows


def run_loyo(X, y, meta, use_pca, pca_n, output_dir=None):
    years = sorted(meta['year'].unique().tolist())
    print(f'\n{"─" * 65}')
    print(f'  [Leave-One-Year-Out]  テスト年: {years}')
    print(f'{"─" * 65}')
    year_arr = meta['year'].to_numpy()

    def _splits():
        for test_year in years:
            va_mask = year_arr == test_year
            tr_idx  = np.where(~va_mask)[0]
            va_idx  = np.where( va_mask)[0]
            print(f'  [test_year={test_year}]  train={len(tr_idx)}  val={len(va_idx)}')
            yield tr_idx, va_idx

    models     = make_models(use_pca=use_pca, pca_n=pca_n)
    pred_store = [] if output_dir else None
    metrics    = _run_folds(X, y, _splits(), len(years), models,
                            pred_store=pred_store, fold_labels=years)
    rows       = _print_summary(metrics, 'loyo', use_pca, pca_n)
    if output_dir and pred_store:
        pca_label = f'PCA({pca_n}d)' if use_pca else 'NoPCA'
        plot_pred_vs_actual(pred_store, 'loyo', pca_label, output_dir)
    return rows


def run_single(X, y, meta, cv_mode, use_pca, pca_n, output_dir=None):
    if cv_mode == 'kfold':
        return run_kfold(X, y, use_pca, pca_n, output_dir=output_dir)
    elif cv_mode == 'loyo':
        return run_loyo(X, y, meta, use_pca, pca_n, output_dir=output_dir)
    else:
        raise ValueError(f'不明な cv_mode: {cv_mode}')


# ── メインエントリ ────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    th1, th2 = GDD_THRESHOLDS
    print('=' * 65)
    print('  Yield Prediction v3.0  (累積GDD期間分割)')
    print(f'  気象変数  : {WEATHER_COLS}')
    print(f'  GDD閾値   : 期間1=[0,{th1}]  期間2=({th1},{th2}]  期間3=({th2},∞)')
    print(f'  特徴量    : {N_VARS}変数 × {N_PERIODS}期間 × {N_STATS}統計({"、".join(STAT_FUNCS)}) = {N_FEATURES_GDD}次元'
          + (' + lat/lon 2次元' if args.add_geo else ''))
    print(f'  CV方式    : {args.cv_mode}')
    print(f'  IQR除外   : {"ON" if args.iqr else "OFF"}')
    if args.add_harm:
        print(f'  Harm      : {HARM_COLS}')
    if args.add_spacing:
        print(f'  Spacing   : {SPACING_COLS}  (between_lines->75, between_stocks->18)')
    if args.add_vwc:
        print(f'  VWC       : {VWC_COLS}  (SolidMoisture: field/year-mean imputation)')
    print('=' * 65)

    # データセット構築
    X, y, geo, meta, feat_cols = build_dataset(
        args.field_db, args.weather_db, args.gdd_csv,
        add_harm=args.add_harm,
        add_spacing=args.add_spacing,
        add_vwc=args.add_vwc
    )

    # ── IQR 外れ値除去 ───────────────────────────────────────────────────
    if args.iqr:
        X, y, geo, meta = apply_iqr(X, y, geo, meta)

    # ── lat/lon を特徴量に追加（任意） ──────────────────────────────────
    if args.add_geo:
        geo_filled = np.where(np.isnan(geo), np.nanmean(geo, axis=0), geo).astype(np.float32)
        X = np.concatenate([X, geo_filled], axis=1)

    n_total, n_feat = X.shape
    print(f'\n総サンプル数: {n_total}  特徴量次元: {n_feat}')
    print(f'yield: min={y.min():.1f}  max={y.max():.1f}  '
          f'mean={y.mean():.1f}  std={y.std():.1f}')

    # ── CV 実行 ──────────────────────────────────────────────────────────
    rows = run_single(X, y, meta,
                      cv_mode=args.cv_mode,
                      use_pca=args.pca,
                      pca_n=args.pca_n,
                      output_dir=args.output_dir)

    # ── 結果保存 ─────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, f'cv_results_{args.cv_mode}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n  Fold 詳細 CSV → {csv_path}')
    print('\nFinished.')


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Yield prediction v3.0: 累積GDD期間分割',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--cv-mode', choices=['kfold', 'loyo'], default='kfold',
                   help='CV方式 (kfold / loyo)')
    p.add_argument('--pca',   action='store_true', help='PCA を有効にする')
    p.add_argument('--pca-n', type=int, default=PCA_N_DEFAULT, dest='pca_n')
    p.add_argument('--add-geo',  action='store_true',
                   help='lat/lon を特徴量ベクトルに追加')
    p.add_argument('--add-harm', action='store_true', dest='add_harm',
                   help='Harm テーブルの sick/weed/wet/typhoon/unripen を追加')
    p.add_argument('--add-spacing', action='store_true', dest='add_spacing',
                   help='between_lines(NaN->75) / between_stocks(non-num/NaN->18) を追加')
    p.add_argument('--add-vwc', action='store_true', dest='add_vwc',
                   help='SolidMoisture VWC を追加 (VWC_mean + has_vwc; 2015は別年平均で補完)')
    p.add_argument('--iqr', action='store_true', help='IQR 外れ値除去を適用')
    p.add_argument('--weather-db', default=WEATHER_DB, dest='weather_db')
    p.add_argument('--field-db',   default=FIELD_DB,   dest='field_db')
    p.add_argument('--gdd-csv',    default=GDD_CSV,    dest='gdd_csv')
    p.add_argument('--output-dir', default=OUTPUT_DIR, dest='output_dir')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
