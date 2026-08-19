"""
eval_yield_deviation.py
================================================================
[目的]
  農林水産省の市町村別大豆収量データ（summary-soy-2010-2018.csv）の
  過去5年移動平均（Y-5～Y-1）をベースラインとして差し引き、
  目的変数を「絶対収量」から「分散（Anomaly）」に変更して LightGBM LOYO を実施する。
  例）対象年度 2015 → 2010～2014 の 5 年間平均がベースライン

[入力特徴量]
  気象変数 9種 × 3 GDD期間 × mean = 27次元
  病害変数 5種                = 5次元
  条間・株間（標準値補完）       = 2次元
  合計 34次元

[バイアス補正]
  なし（LightGBMが内部的に学習データの平均を吸収するため不要と判断）

[出力]
  - コンソール: LOYO RMSE, MAPE, R2（絶対収量ベース・偏差ベース両方）
  - 散布図: outputs/yield_pred_v3/eval_deviation_loyo.png
  - 除外圃場リスト: outputs/yield_pred_v3/excluded_fields_deviation.csv
"""

import sqlite3, os, warnings, json, sys, re
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.interpolate import interp1d  # 分位点マッピング用
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── フォント設定 ───────────────────────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定 ──────────────────────────────────────────────────────────────────────
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB  = os.path.join('data', 'processed', 'weather_database_fieldid.db')
GDD_CSV     = os.path.join('outputs', 'gdd', 'gdd_daily.csv')
MAFF_CSV    = os.path.join('data', 'raw', 'summary-soy-2010-2018.csv')  # 2010～2018
MAFF_WINDOW          = 1     # 移動平均の年数（Y-5～Y-1）
USE_QUANTILE_MAPPING = False  # True: LOYOループ内で分位点マッピング（QM）を適用する
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_v5')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS   = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
# 病害スコア列（1～4の段階評価）: NaNは「未記入」として扱い、LightGBMにNaNのまま渡す
HARM_SCORE_COLS = ['sick', 'wet', 'unripen', 'weed', 'bug', 'lay', 'loss']
# 病害フラグ列（TRUE/FALSE）: NaN→被害なし（0）として扱う
HARM_FLAG_COLS  = ['typhoon', 'long_rain', 'heavy_rain', 'drought', 'gale', 'few_solar']
HARM_COLS       = HARM_SCORE_COLS + HARM_FLAG_COLS  # 全列（後続利用用）
GDD_THRESHOLDS = [600, 1000]
RANDOM_STATE   = 42

# ── 1. 収量・位置データ ──────────────────────────────────────────────────────
print('=' * 60)
print(f'  大豆収量分散予測モデル（農水省{MAFF_WINDOW}年移動平均ベースライン）')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon, between_lines, between_stocks, breed
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn)
conn.close()
quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)
quest_df['yield']    = quest_df['yield'].astype(float)
quest_df['lat']      = pd.to_numeric(quest_df['lat'], errors='coerce')
quest_df['lon']      = pd.to_numeric(quest_df['lon'], errors='coerce')
quest_df['between_lines'] = pd.to_numeric(quest_df['between_lines'], errors='coerce').fillna(75.0)
quest_df['between_stocks'] = pd.to_numeric(quest_df['between_stocks'], errors='coerce').fillna(18.0)
# breed: 欠損を 'Unknown' で埋め、pandasのcategory型に変換
quest_df['breed'] = quest_df['breed'].fillna('Unknown').astype(str).str.strip()
quest_df['breed'] = quest_df['breed'].astype('category')
quest_df = quest_df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)
print(f'質問票データ: {len(quest_df)} 件')
print(f'品種（breed）: {quest_df["breed"].nunique()} 種類')

# ── 2. 逆ジオコーディング結果（field_id → city, muniCd）──────────────────
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city', 'muniCd']].drop_duplicates('field_id')
geo_df['muniCd'] = pd.to_numeric(geo_df['muniCd'], errors='coerce').astype('Int64')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')
n_no_city = quest_df['city'].isna().sum()
print(f'市町村紐付け: {quest_df["city"].notna().sum()} 件成功 / {n_no_city} 件失敗')

# ── 3. 農林水産省データ読み込み（2010～2018）とクリーニング ─────────────────
maff_raw = pd.read_csv(MAFF_CSV, encoding='cp932')
maff_raw.columns = ['year', 'city', 'maff_yield', 'muniCd']
maff_raw['year']       = maff_raw['year'].astype(int)
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['city']       = maff_raw['city'].astype(str).str.strip()
maff_raw['muniCd']     = pd.to_numeric(maff_raw['muniCd'], errors='coerce').astype('Int64')
maff_valid = maff_raw.dropna(subset=['maff_yield'])
print(f'\n農水省データ: {len(maff_raw)} 行 / 有効（数値）: {len(maff_valid)} 行')
print(f'  対象年度: {sorted(maff_valid["year"].unique())}')

# ── 4. muniCdを主キーに移動平均をマッチング ─────────────────────────
TARGET_YEARS = [2015, 2016, 2017, 2018]
quest_df['muniCd'] = pd.to_numeric(quest_df['muniCd'], errors='coerce').astype('Int64')

maff_rows = []
for target_year in TARGET_YEARS:
    hist_years = list(range(target_year - MAFF_WINDOW, target_year))  # Y-5～Y-1
    hist = maff_valid[maff_valid['year'].isin(hist_years)]
    # muniCd ごとに有効年度の平均（同名の市町村混同を防止）
    muni_avg = hist.groupby('muniCd')['maff_yield'].mean().reset_index()
    muni_avg.rename(columns={'maff_yield': 'maff_yield'}, inplace=True)
    muni_avg['join_year'] = target_year
    maff_rows.append(muni_avg)
maff_5yr = pd.concat(maff_rows, ignore_index=True)
maff_5yr['muniCd'] = maff_5yr['muniCd'].astype('Int64')

quest_df = quest_df.merge(
    maff_5yr[['join_year', 'muniCd', 'maff_yield']],
    left_on=['year', 'muniCd'],
    right_on=['join_year', 'muniCd'],
    how='left'
).drop(columns='join_year')

# マッチング結果レポート
n_matched = quest_df['maff_yield'].notna().sum()
n_unmatch = quest_df['maff_yield'].isna().sum()
print(f'\n{MAFF_WINDOW}年移動平均マッチング結果（muniCd完全一致ベース）:')
print(f'  マッチ成功: {n_matched} 件')
print(f'  マッチ失敗（除外予定）: {n_unmatch} 件')

# ── 5. 除外圃場を記録して保存 ──────────────────────────────────────────────
excluded_df = quest_df[quest_df['maff_yield'].isna()].copy()

# 除外理由を詳細化
maff_municds = set(maff_valid['muniCd'].dropna().astype(int).tolist())
def get_reason(row):
    mcd = row.get('muniCd')
    if pd.isna(mcd):
        return 'muniCdなし（逆ジオコーディング失敗等）'
    if int(mcd) not in maff_municds:
        return f'MAFF未掲載市町村（muniCd={int(mcd)}）: {row.get("city", "")}'
    return (f'MAFF過去{MAFF_WINDOW}年間データに有効値なし: '
            f'muniCd={int(mcd)} ({row.get("city", "")}), 対象年={row["year"]}, '
            f'必要年度={row["year"]-MAFF_WINDOW}～{row["year"]-1}')

excluded_df['reason'] = excluded_df.apply(get_reason, axis=1)
excl_path = os.path.join(OUT_DIR, 'excluded_fields_deviation.csv')
excluded_df.to_csv(excl_path, index=False, encoding='utf-8-sig')
print(f'  除外リスト保存: {excl_path}')
print('  除外理由の内訳:')
for reason, cnt in excluded_df['reason'].value_counts().items():
    print(f'    [{cnt:3d}件] {reason}')

# ── 6. 有効サンプルで偏差を計算 ───────────────────────────────────────────
valid_df = quest_df.dropna(subset=['maff_yield']).reset_index(drop=True)
valid_df['y_diff'] = valid_df['yield'] - valid_df['maff_yield']
print(f'\n有効サンプル: {len(valid_df)} 件')
print(f'年別内訳: {valid_df.groupby("year").size().to_dict()}')
print(f'y_diff (偏差) 統計:')
print(f'  mean={valid_df["y_diff"].mean():.1f}  '
      f'std={valid_df["y_diff"].std():.1f}  '
      f'min={valid_df["y_diff"].min():.1f}  '
      f'max={valid_df["y_diff"].max():.1f}')
print(f'maff_yield ({MAFF_WINDOW}年移動平均) 統計:')
print(f'  mean={valid_df["maff_yield"].mean():.1f}  '
      f'std={valid_df["maff_yield"].std():.1f}')

# ── 7. GDD期間別気象特徴量の生成（9変数×3期間×mean = 27次元）──────────────
fids  = sorted(valid_df['field_id'].unique().tolist())
years = sorted(valid_df['year'].unique().tolist())

print(f'\nGDD読み込み...', end=' ', flush=True)
gdd_df = pd.read_csv(GDD_CSV, encoding='utf-8-sig')
gdd_df['date'] = pd.to_datetime(gdd_df['date'])
cum_col = [c for c in gdd_df.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
th1, th2 = GDD_THRESHOLDS
gdd_df['period'] = 1
gdd_df.loc[gdd_df[cum_col] > th1, 'period'] = 2
gdd_df.loc[gdd_df[cum_col] > th2, 'period'] = 3
gdd_df = gdd_df[['field_id', 'year', 'date', 'period']]
print(f'{len(gdd_df):,} 行')

print(f'気象データ読み込み ({len(fids)} 圃場)...', end=' ', flush=True)
conn_w = sqlite3.connect(WEATHER_DB)
fid_ph  = ','.join(['?'] * len(fids))
yr_ph   = ','.join(f"'{y}'" for y in years)
col_str = ', '.join(WEATHER_COLS)
weather_df = pd.read_sql(f'''
    SELECT field_id, date, {col_str} FROM weather_data
    WHERE field_id IN ({fid_ph})
      AND CAST(SUBSTR(date,1,4) AS INTEGER) IN ({yr_ph})
    ORDER BY field_id, date''', conn_w, params=fids)
conn_w.close()
weather_df['field_id'] = weather_df['field_id'].astype(int)
weather_df['date']     = pd.to_datetime(weather_df['date'])
print(f'{len(weather_df):,} 行')

print('GDD期間別特徴量計算...')
merged_gdd = gdd_df.merge(weather_df[['field_id', 'date'] + WEATHER_COLS],
                          on=['field_id', 'date'], how='left')
grp = (merged_gdd.groupby(['field_id', 'year', 'period'])[WEATHER_COLS]
       .agg('mean'))
grp_pivot = grp.unstack('period')
grp_pivot.columns = [f'{v}_p{int(p)}_mean' for v, p in grp_pivot.columns]
gdd_feat_cols = [f'{v}_p{p}_mean' for p in [1, 2, 3] for v in WEATHER_COLS]
for col in gdd_feat_cols:
    if col not in grp_pivot.columns:
        grp_pivot[col] = np.nan
feat_df = grp_pivot[gdd_feat_cols].reset_index()
print(f'気象特徴量: {len(gdd_feat_cols)} 次元')

# 有効サンプルに気象特徴量を結合
all_data = valid_df.merge(feat_df, on=['field_id', 'year'], how='inner')
print(f'気象結合後: {len(all_data)} 件')

# ── 7b. 当年病害データの読み込みと結合 ──────────────────────────────────────
# 変数の意味を考慮した正確な変換:
#   sick  : 1=なかった, 2=あった, 3=不明 → 3はNaN
#   bug   : 2015年: 1=目立たなかった, 2=目立った, 3=不明 → 3はNaN
#           2016+年: 1=目立たなかった, 2=ほ場で, 3=収穫物で, 4=不明 → 4はNaN
#   weed  : 1=なかった, 2=あった（バイナリ、そのままOK）
#   wet   : 1=なかった, 2=あった（バイナリ、そのままOK）
#   unripen/lay/loss: 1=無, 2=少, 3=中, 4=多（順序変数、そのままOK）
print('当年病害データ読み込み...', end=' ', flush=True)
conn_h = sqlite3.connect(FIELD_DB)
harm_col_str = ', '.join(HARM_SCORE_COLS + HARM_FLAG_COLS)
harm_df = pd.read_sql(f'''
    SELECT field_id, year, {harm_col_str} FROM harm
    WHERE field_id IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn_h)
conn_h.close()
harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)

# ── スコア列の数値変換 ────────────────────────────────────────────────────────
for c in HARM_SCORE_COLS:
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce')

# sick（病害）: 3=不明 → NaN に変換
harm_df['sick'] = harm_df['sick'].where(harm_df['sick'] != 3, other=np.nan)

# bug（虫害）: 年によってコード体系が異なる
#   2015年: 3=不明 → NaN
harm_df.loc[harm_df['year'] == 2015, 'bug'] = (
    harm_df.loc[harm_df['year'] == 2015, 'bug']
    .where(harm_df.loc[harm_df['year'] == 2015, 'bug'] != 3, other=np.nan))
#   2016年以降: 4=不明 → NaN (3=収穫物で目立った は有効な情報として保持)
harm_df.loc[harm_df['year'] >= 2016, 'bug'] = (
    harm_df.loc[harm_df['year'] >= 2016, 'bug']
    .where(harm_df.loc[harm_df['year'] >= 2016, 'bug'] != 4, other=np.nan))

# ── フラグ列: TRUE/FALSE → 1/0 変換（NaN → 被害なし=0）──────────────────────
for c in HARM_FLAG_COLS:
    harm_df[c] = harm_df[c].replace({'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0,
                                     True: 1, False: 0})
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce').fillna(0).astype(int)

# ── 圃場×年単位で集計 ─────────────────────────────────────────────────────────
# フラグ=max（1件でも被害あればTRUE相当）、スコア=mean（平均的な被害度、不明除外済み）
agg_dict = {c: 'max'  for c in HARM_FLAG_COLS}
agg_dict.update({c: 'mean' for c in HARM_SCORE_COLS})
harm_df = harm_df.groupby(['field_id', 'year'])[HARM_SCORE_COLS + HARM_FLAG_COLS].agg(agg_dict).reset_index()
print(f'{len(harm_df):,} 行')

all_data = all_data.merge(harm_df, on=['field_id', 'year'], how='left')

# ── マージ後のNaN処理 ─────────────────────────────────────────────────────────
for c in HARM_FLAG_COLS:
    all_data[c] = all_data[c].fillna(0)   # 記録なし → 被害なし
# スコア列のNaN（未記入 or 不明）はそのままLightGBMへ（Imputer経由で平均補完）
print(f'病害結合後: {len(all_data)} 件')
n_score_nan = {c: int(all_data[c].isna().sum()) for c in HARM_SCORE_COLS}
print(f'  スコアNaN件数（不明値+未回答）: { {k:v for k,v in n_score_nan.items() if v>0} }')



# ── 8. LightGBM モデル定義 ──────────────────────────────────────────────────────────────
CAT_COLS = ['breed']  # LightGBMのネイティブカテゴリカル指定列

def make_lgb_model(num_feat_cols: list, params: dict = None):
    """数値列のはImputerを通し、breedはそのままLGBMに渡すモデル生成関数。
    戻り値: (imputer, lgb_model) のタプル。
    """
    if params is None:
        params = {}
    imputer = SimpleImputer(strategy='mean')
    model   = lgb.LGBMRegressor(
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        verbose      = -1,
        **params)
    return imputer, model

def fit_predict(imputer, model, X_df_tr, y_tr, X_df_va, num_feat_cols):
    """訓練・推論ヘルパー。
    - 数値列だけはImputerで缺損補完
    - breed (category型) はそのままDataFrameとしてLGBMに渡す。
    """
    # 数値列の缺損補完
    X_num_tr = imputer.fit_transform(X_df_tr[num_feat_cols])
    X_num_va = imputer.transform(X_df_va[num_feat_cols])

    # category列を希望の列順にソートした DataFrame を準備
    X_cat_tr = X_df_tr[CAT_COLS].reset_index(drop=True)
    X_cat_va = X_df_va[CAT_COLS].reset_index(drop=True)

    # 数値 + カテゴリカルを DataFrame にまとめる
    X_tr_full = pd.concat(
        [pd.DataFrame(X_num_tr, columns=num_feat_cols), X_cat_tr], axis=1)
    X_va_full = pd.concat(
        [pd.DataFrame(X_num_va, columns=num_feat_cols), X_cat_va], axis=1)

    model.fit(
        X_tr_full, y_tr,
        categorical_feature=CAT_COLS
    )
    return model.predict(X_va_full)

def fit_quantile_map(y_maff_tr: np.ndarray, y_true_tr: np.ndarray,
                     n_quantiles: int = 100):
    """
    訓練データ（train fold）のみを使って Quantile Mapping 関数を構築する。
    MAFF の分布 → 現場（Questionaire）の分布 に変換する interp1d 関数を返す。

    Args:
        y_maff_tr   : train fold の前年MAFF収量配列
        y_true_tr   : train fold の実測収量配列
        n_quantiles : パーセンタイルの分割数（デフォルト100）
    Returns:
        interp1d 関数（MAFFの値 → 現場スケールの値）
    """
    quantiles = np.linspace(0, 100, n_quantiles)
    maff_q = np.percentile(y_maff_tr, quantiles)
    true_q = np.percentile(y_true_tr, quantiles)
    # 単調増加を保証するため重複パーセンタイルを除去してから線形補間
    _, idx = np.unique(maff_q, return_index=True)
    return interp1d(maff_q[idx], true_q[idx],
                    kind='linear', fill_value='extrapolate')

def _hpo_objective(trial, X_all, y_diff_all, y_true_all, y_maff_all, year_all, num_feat_cols):
    """Optunaオブジェクティブ関数: LOYO構造（年グループCV）で AbsRMSE を返す"""
    params = {
        'num_leaves'       : trial.suggest_int(  'num_leaves',        10,  150),
        'min_child_samples': trial.suggest_int(  'min_child_samples',  5,   60),
        'learning_rate'    : trial.suggest_float('learning_rate',  0.01,  0.3, log=True),
        'n_estimators'     : trial.suggest_int(  'n_estimators',     50,  500),
        'reg_lambda'       : trial.suggest_float('reg_lambda',      0.0, 30.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha',       0.0, 10.0),
        'subsample'        : trial.suggest_float('subsample',       0.5,  1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.4,  1.0),
    }
    test_years = sorted(np.unique(year_all))
    rmse_list = []
    for test_year in test_years:
        tr = np.where(year_all != test_year)[0]
        va = np.where(year_all == test_year)[0]
        if len(tr) == 0 or len(va) == 0:
            continue

        if USE_QUANTILE_MAPPING:
            # ★ 訓練データのみでQMを構築し、訓練・バリデーション両方に適用（データリーク防止）
            qm_func       = fit_quantile_map(y_maff_all[tr], y_true_all[tr])
            y_maff_tr_qm  = qm_func(y_maff_all[tr]).astype(np.float32)
            y_maff_va_qm  = qm_func(y_maff_all[va]).astype(np.float32)
            y_diff_tr     = (y_true_all[tr] - y_maff_tr_qm).astype(np.float32)
            maff_for_pred = y_maff_va_qm
        else:
            y_diff_tr     = y_diff_all[tr]
            maff_for_pred = y_maff_all[va]

        imputer, model = make_lgb_model(num_feat_cols, params)
        pred_diff = fit_predict(
            imputer, model,
            X_all.iloc[tr], y_diff_tr,
            X_all.iloc[va], num_feat_cols)
        pred_abs = pred_diff + maff_for_pred
        rmse_list.append(float(np.sqrt(((y_true_all[va] - pred_abs) ** 2).mean())))
    return float(np.mean(rmse_list))


# ── 9. LOYO 実行 ───────────────────────────────────────────────────────────────────
all_feat_cols     = gdd_feat_cols + HARM_COLS + ['between_lines', 'between_stocks']
num_feat_cols     = all_feat_cols  # 数値特徴量列（27次元気象+5次元病害+2次元=34次元）
all_feat_cols_cat = all_feat_cols + CAT_COLS   # breed を含む全特徴量（35次元）
# 27次元気象 + 5次元病害 + 2次元（条間・株間） + 1次元（breed） = 35次元

# breedのpandascategory型を全データ共通のカテゴリに統一（LOYO分割時にカテゴリが欠けるのを防ぐ）
all_data['breed'] = all_data['breed'].astype('category')

# DataFrame形式で保持（breedはcategory型を維持）
X_df     = all_data[all_feat_cols_cat].copy()
y_diff   = all_data['y_diff'].to_numpy(dtype=np.float32)
y_true   = all_data['yield'].to_numpy(dtype=np.float32)
y_maff   = all_data['maff_yield'].to_numpy(dtype=np.float32)
year_arr = all_data['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))

# ── 9a. ハイパーパラメータ: JSONから読み込み or Optuna探索 ────────────────────
N_TRIALS = 100
bp_path = os.path.join(OUT_DIR, f'best_params_deviation{"_qm" if USE_QUANTILE_MAPPING else ""}.json')

if os.path.exists(bp_path):
    # ── JSONが存在する場合: 保存済みパラメータを使用 ─────────────────
    with open(bp_path) as f:
        saved = json.load(f)
    best_params = saved['best_params']
    best_rmse   = saved.get('best_rmse', float('nan'))
    print(f'\n{"=" * 60}')
    print(f'  パラメータ読み込み: {bp_path}')
    print(f'  保存済み HPO RMSE: {best_rmse:.4f}')
    print(f'  パラメータ:')
    for k, v in best_params.items():
        print(f'    {k:<22} = {v}')
    print(f'{"=" * 60}')
else:
    # ── JSONが存在しない場合: Optuna で探索して保存 ───────────────
    print(f'\n{"=" * 60}')
    print(f'  Optuna ハイパーパラメータ探索 ({N_TRIALS} trials)')
    print(f'{"=" * 60}')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _hpo_objective(
            trial, X_df, y_diff, y_true, y_maff, year_arr, num_feat_cols),
        n_trials=N_TRIALS,
        show_progress_bar=False
    )
    best_params = study.best_params
    best_rmse   = study.best_value
    print(f'  最良 RMSE: {best_rmse:.4f}')
    print(f'  最良パラメータ:')
    for k, v in best_params.items():
        print(f'    {k:<22} = {v}')
    with open(bp_path, 'w') as f:
        json.dump({'best_rmse': best_rmse, 'best_params': best_params}, f, indent=2)
    print(f'  パラメータ保存: {bp_path}')
    sys.stdout.flush()  # Optuna完了後に必ずフラッシュ

print(f'\n{"=" * 60}')
print(f'  LOYO 評価（分散予測モデル ・ {MAFF_WINDOW}年移動平均ベースライン）')
print(f'  サンプル数: {len(all_data)} 件 / 特徴量: {len(all_feat_cols_cat)} 次元')
print(f'  テスト年: {test_years}')
print(f'{"=" * 60}')

all_true_abs, all_pred_abs, all_years = [], [], []
all_true_diff, all_pred_diff, all_maff  = [], [], []
per_year_results = []

for test_year in test_years:
    tr = np.where(year_arr != test_year)[0]
    va = np.where(year_arr == test_year)[0]
    if len(tr) == 0 or len(va) == 0:
        continue

    if USE_QUANTILE_MAPPING:
        # ★ 訓練データのみでQMを構築（データリーク防止）し、訓練・テスト両方に適用
        qm_func      = fit_quantile_map(y_maff[tr], y_true[tr])
        y_maff_tr_qm = qm_func(y_maff[tr]).astype(np.float32)
        y_maff_va_qm = qm_func(y_maff[va]).astype(np.float32)
        y_diff_tr    = (y_true[tr] - y_maff_tr_qm).astype(np.float32)
        y_diff_va    = (y_true[va] - y_maff_va_qm).astype(np.float32)
        maff_for_restore = y_maff_va_qm   # 復元・ナイーブ評価に使うQM補正済みMAFF
    else:
        y_diff_tr        = y_diff[tr]
        y_diff_va        = y_diff[va]
        maff_for_restore = y_maff[va]

    imputer, model = make_lgb_model(num_feat_cols, best_params)
    pred_diff_va = fit_predict(
        imputer, model,
        X_df.iloc[tr], y_diff_tr,
        X_df.iloc[va], num_feat_cols)

    # 絶対収量に復元（予測偏差 + QM補正済みMAFF市町村平均）
    pred_abs  = pred_diff_va + maff_for_restore
    true_abs  = y_true[va]
    true_diff = y_diff_va

    rmse_abs = float(np.sqrt(((true_abs - pred_abs) ** 2).mean()))
    mape_abs = float(np.abs((pred_abs - true_abs) / true_abs).mean() * 100)
    ss_r  = ((true_abs - pred_abs) ** 2).sum()
    ss_t  = ((true_abs - true_abs.mean()) ** 2).sum()
    r2    = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
    rmse_diff = float(np.sqrt(((true_diff - pred_diff_va) ** 2).mean()))

    print(f'  test={test_year}(n={len(va):2d}) train={len(tr):2d}  '
          f'AbsRMSE={rmse_abs:7.3f}  MAPE={mape_abs:6.2f}%  '
          f'R2={r2:6.4f}  DiffRMSE={rmse_diff:.3f}')
    per_year_results.append({'year': test_year, 'n': len(va),
                             'AbsRMSE': rmse_abs, 'MAPE': mape_abs, 'R2': r2,
                             'DiffRMSE': rmse_diff})

    all_true_abs.extend(true_abs.tolist())
    all_pred_abs.extend(pred_abs.tolist())
    all_years.extend([test_year] * len(va))
    all_true_diff.extend(true_diff.tolist())
    all_pred_diff.extend(pred_diff_va.tolist())
    all_maff.extend(maff_for_restore.tolist())   # ナイーブベース用（QM補正済みMAFF）

# ── プール計算 ────────────────────────────────────────────────────────────────
yt = np.array(all_true_abs)
yp = np.array(all_pred_abs)
rmse_pool = float(np.sqrt(((yt - yp) ** 2).mean()))
mape_pool = float(np.abs((yp - yt) / yt).mean() * 100)
ss_r = ((yt - yp) ** 2).sum()
ss_t = ((yt - yt.mean()) ** 2).sum()
r2_pool = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')

yd  = np.array(all_true_diff)
ypd = np.array(all_pred_diff)
rmse_d_pool = float(np.sqrt(((yd - ypd) ** 2).mean()))
ss_r2 = ((yd - ypd) ** 2).sum()
ss_t2 = ((yd - yd.mean()) ** 2).sum()
r2_d_pool = float(1 - ss_r2 / ss_t2) if ss_t2 > 0 else float('nan')

# ── ナイーブベース（偏差=0予測、つまり前年市町村平均を予測値として使う）────────
ym  = np.array(all_maff)        # 各テストサンプルの前年MAFF平均（ナイーブ予測値）
# AbsRMSE/MAPE/R2（絶対収量空間）
rmse_naive  = float(np.sqrt(((yt - ym) ** 2).mean()))  # = sqrt(mean(y_diff^2))
mape_naive  = float(np.abs((ym - yt) / yt).mean() * 100)
ss_r_n = ((yt - ym) ** 2).sum()
r2_naive = float(1 - ss_r_n / ss_t) if ss_t > 0 else float('nan')
# R2（偏差空間: 予測偏差=0 vs 実際の偏差）
ss_r2_n = (yd ** 2).sum()   # pred_diff=0 なので残差=y_diff
r2_naive_d = float(1 - ss_r2_n / ss_t2) if ss_t2 > 0 else float('nan')

print(f'\n{"=" * 60}')
print('  プール計算 サマリ')
print(f'{"=" * 60}')
print(f'  {"条件":<25} {"RMSE":>8} {"MAPE":>8} {"R2(絶対)":>10} {"R2(偏差)":>10}')
print(f'  {"-"*63}')
_naive_label = f'ナイーブ（{MAFF_WINDOW}年移動平均）'
print(f'  {_naive_label:<25} {rmse_naive:>8.3f} {mape_naive:>7.2f}% {r2_naive:>10.4f} {r2_naive_d:>10.4f}')
print(f'  {"LightGBM（偏差予測→復元）":<25} {rmse_pool:>8.3f} {mape_pool:>7.2f}% {r2_pool:>10.4f} {r2_d_pool:>10.4f}')
print(f'  {"-"*63}')
rmse_imp  = rmse_naive  - rmse_pool
mape_imp  = mape_naive  - mape_pool
print(f'  {"モデルの改善量":<25} {rmse_imp:>+8.3f} {mape_imp:>+7.2f}%')
print(f'  評価方法: LOYO年グループCV')

# ── 10. 散布図 ────────────────────────────────────────────────────────────────
YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60', 2015: '#8e44ad'}
yrs = np.array(all_years)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f8f9fa')
fig.suptitle(f'LightGBM LOYO: 分散予測モデル\n（農水省{MAFF_WINDOW}年移動平均をベースライン）',
             fontsize=13, fontweight='bold')

# 左: 絶対収量での評価
for yr, col in YEAR_COLORS.items():
    m = yrs == yr
    if m.any():
        ax1.scatter(yt[m], yp[m], alpha=0.8, s=80, c=col,
                    edgecolors='white', linewidths=0.8, zorder=3,
                    label=f'{yr}year (n={m.sum()})')
mn = min(yt.min(), yp.min()) - 20
mx = max(yt.max(), yp.max()) + 20
ax1.plot([mn, mx], [mn, mx], '--', color='#555555', lw=1.5, zorder=2)
ax1.set_xlim(mn, mx); ax1.set_ylim(mn, mx)
ax1.set_xlabel('Observed yield (kg/10a)', fontsize=12)
ax1.set_ylabel('Predicted yield (kg/10a)', fontsize=12)
ax1.set_title('Absolute yield (deviation -> restored)', fontsize=11, fontweight='bold')
ax1.text(0.04, 0.96,
         f'RMSE={rmse_pool:.2f}\nMAPE={mape_pool:.2f}%\nR2={r2_pool:.4f}',
         transform=ax1.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.87, edgecolor='#cccccc'))
ax1.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax1.grid(True, alpha=0.25); ax1.set_facecolor('#fdfdfd'); ax1.set_axisbelow(True)

# 右: 偏差での評価
for yr, col in YEAR_COLORS.items():
    m = yrs == yr
    if m.any():
        ax2.scatter(yd[m], ypd[m], alpha=0.8, s=80, c=col,
                    edgecolors='white', linewidths=0.8, zorder=3,
                    label=f'{yr}year (n={m.sum()})')
mn2 = min(yd.min(), ypd.min()) - 20
mx2 = max(yd.max(), ypd.max()) + 20
ax2.plot([mn2, mx2], [mn2, mx2], '--', color='#555555', lw=1.5, zorder=2)
ax2.set_xlim(mn2, mx2); ax2.set_ylim(mn2, mx2)
ax2.set_xlabel('Observed deviation (kg/10a)', fontsize=12)
ax2.set_ylabel('Predicted deviation (kg/10a)', fontsize=12)
ax2.set_title('Deviation only', fontsize=11, fontweight='bold')
ax2.text(0.04, 0.96,
         f'RMSE={rmse_d_pool:.2f}\nR2={r2_d_pool:.4f}',
         transform=ax2.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.87, edgecolor='#cccccc'))
ax2.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax2.grid(True, alpha=0.25); ax2.set_facecolor('#fdfdfd'); ax2.set_axisbelow(True)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'eval_deviation_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  散布図 -> {out_path}')
print('\n完了')
