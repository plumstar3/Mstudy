"""
eval_yield_deviation_sameyear.py
================================================================
[目的]
  農林水産省の市町村別大豆収量データ（summary-soy-2010-2018.csv）を用いて、
  偏差予測モデルを構築する。

  【eval_yield_deviation.py との違い】
  - 学習サンプル（train）: 当年の市町村平均収量（maff_current）を偏差のベースラインとする
      y_diff_train = yield - maff_current  ← より精度の高いベースライン
  - テストサンプル（test）: 前年の市町村平均収量（maff_prev）を偏差のベースラインとする
      ※当年のMAFFをテストに使うとデータリークとなるため、前年で代替
      y_diff_test  = yield - maff_prev     ← リーク防止

  LOYOループ内でサンプルの役割（train/test）に応じてベースラインを切り替える。
  その他の処理（気象特徴量・病害・HPO・評価方法）はeval_yield_deviation.pyと同一。

[入力特徴量]
  気象変数 9種 × 3 GDD期間 × mean = 27次元
  病害変数 5種                = 5次元
  条間・株間（標準値補完）       = 2次元
  品種（breed / カテゴリカル）   = 1次元
  合計 35次元

[出力]
  - コンソール: LOYO RMSE, MAPE, R2（絶対収量ベース・偏差ベース両方）
  - 散布図: outputs/yield_pred_v4/eval_deviation_sameyear_loyo.png
  - 除外圃場リスト: outputs/yield_pred_v4/excluded_fields_sameyear.csv
"""

import sqlite3, os, warnings, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_v4')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS   = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']
GDD_THRESHOLDS = [600, 1000]
RANDOM_STATE   = 42

# ── 1. 収量・位置データ ──────────────────────────────────────────────────────
print('=' * 60)
print('  大豆収量分散予測モデル（当年MAFFベース / test年は前年で代替）')
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
quest_df['between_lines']  = pd.to_numeric(quest_df['between_lines'],  errors='coerce').fillna(75.0)
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

# ── 4. 当年 / 前年 のMAFF市町村平均を両方マッチング ────────────────────────
#
# [設計思想]
#   - maff_current: 当年(Y年)の市町村平均収量  → 学習時のベースライン
#   - maff_prev   : 前年(Y-1年)の市町村平均収量 → テスト時のベースライン（リーク防止）
#
TARGET_YEARS = [2015, 2016, 2017, 2018]
quest_df['muniCd'] = pd.to_numeric(quest_df['muniCd'], errors='coerce').astype('Int64')

# 当年MAFF（muniCd × year でそのまま結合）
maff_current_df = (maff_valid[maff_valid['year'].isin(TARGET_YEARS)]
                   [['year', 'muniCd', 'maff_yield']]
                   .rename(columns={'maff_yield': 'maff_current'}))
maff_current_df['muniCd'] = maff_current_df['muniCd'].astype('Int64')

# 前年MAFF（Y-1年のデータをY年として結合できるよう year+1 してから結合）
maff_prev_df = maff_valid[['year', 'muniCd', 'maff_yield']].copy()
maff_prev_df['year']    = maff_prev_df['year'] + 1   # 2014→2015, ..., 2017→2018
maff_prev_df['muniCd']  = maff_prev_df['muniCd'].astype('Int64')
maff_prev_df = (maff_prev_df[maff_prev_df['year'].isin(TARGET_YEARS)]
                .rename(columns={'maff_yield': 'maff_prev'}))

# quest_df に当年 / 前年 のMAFFを結合
quest_df = quest_df.merge(maff_current_df, on=['year', 'muniCd'], how='left')
quest_df = quest_df.merge(maff_prev_df,    on=['year', 'muniCd'], how='left')

# マッチング結果レポート（当年 / 前年 ともにNaNでない件数を報告）
n_matched_curr = quest_df['maff_current'].notna().sum()
n_matched_prev = quest_df['maff_prev'].notna().sum()
n_unmatch      = quest_df[quest_df['maff_current'].isna() | quest_df['maff_prev'].isna()].shape[0]
print(f'\n当年MAFFマッチング:  {n_matched_curr} 件成功 / {quest_df["maff_current"].isna().sum()} 件失敗')
print(f'前年MAFFマッチング:  {n_matched_prev} 件成功 / {quest_df["maff_prev"].isna().sum()} 件失敗')

# ── 5. 除外圃場を記録して保存 ──────────────────────────────────────────────
# 当年・前年どちらかでもNaNがある場合は除外
excluded_df = quest_df[quest_df['maff_current'].isna() | quest_df['maff_prev'].isna()].copy()
maff_municds = set(maff_valid['muniCd'].dropna().astype(int).tolist())

def get_reason(row):
    mcd = row.get('muniCd')
    if pd.isna(mcd):
        return 'muniCdなし（逆ジオコーディング失敗等）'
    if int(mcd) not in maff_municds:
        return f'MAFF未掲載市町村（muniCd={int(mcd)}）: {row.get("city", "")}'
    if pd.isna(row['maff_current']):
        return f'MAFF当年データなし: muniCd={int(mcd)} ({row.get("city", "")}), 対象年={row["year"]}'
    if pd.isna(row['maff_prev']):
        return f'MAFF前年データなし: muniCd={int(mcd)} ({row.get("city", "")}), 対象年={row["year"]} (前年={row["year"]-1})'
    return '不明'

excluded_df['reason'] = excluded_df.apply(get_reason, axis=1)
excl_path = os.path.join(OUT_DIR, 'excluded_fields_sameyear.csv')
excluded_df.to_csv(excl_path, index=False, encoding='utf-8-sig')
print(f'  除外リスト保存: {excl_path}')
print('  除外理由の内訳:')
for reason, cnt in excluded_df['reason'].value_counts().items():
    print(f'    [{cnt:3d}件] {reason}')

# ── 6. 有効サンプル確定・統計表示 ────────────────────────────────────────────
valid_df = quest_df[quest_df['maff_current'].notna() & quest_df['maff_prev'].notna()].copy()
valid_df = valid_df.reset_index(drop=True)

# 参考のため、当年ベースラインの偏差を事前計算（統計表示のみ。LOYOでは使い分ける）
valid_df['y_diff_curr'] = valid_df['yield'] - valid_df['maff_current']
valid_df['y_diff_prev'] = valid_df['yield'] - valid_df['maff_prev']

print(f'\n有効サンプル: {len(valid_df)} 件')
print(f'年別内訳: {valid_df.groupby("year").size().to_dict()}')
print(f'y_diff 統計（当年ベースライン）:')
print(f'  mean={valid_df["y_diff_curr"].mean():.1f}  '
      f'std={valid_df["y_diff_curr"].std():.1f}  '
      f'min={valid_df["y_diff_curr"].min():.1f}  '
      f'max={valid_df["y_diff_curr"].max():.1f}')
print(f'y_diff 統計（前年ベースライン）:')
print(f'  mean={valid_df["y_diff_prev"].mean():.1f}  '
      f'std={valid_df["y_diff_prev"].std():.1f}  '
      f'min={valid_df["y_diff_prev"].min():.1f}  '
      f'max={valid_df["y_diff_prev"].max():.1f}')
print(f'maff_current（当年平均）統計:')
print(f'  mean={valid_df["maff_current"].mean():.1f}  std={valid_df["maff_current"].std():.1f}')
print(f'maff_prev（前年平均）統計:')
print(f'  mean={valid_df["maff_prev"].mean():.1f}  std={valid_df["maff_prev"].std():.1f}')

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

all_data = valid_df.merge(feat_df, on=['field_id', 'year'], how='inner')
print(f'気象結合後: {len(all_data)} 件')

# ── 7b. 当年病害データの読み込みと結合 ──────────────────────────────────────
print('当年病害データ読み込み...', end=' ', flush=True)
conn_h = sqlite3.connect(FIELD_DB)
harm_col_str = ', '.join(HARM_COLS)
harm_df = pd.read_sql(f'''
    SELECT field_id, year, {harm_col_str} FROM harm
    WHERE field_id IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn_h)
conn_h.close()
harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)
for c in HARM_COLS:
    harm_df[c] = harm_df[c].replace({'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0,
                                     True: 1, False: 0})
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce').fillna(0)
harm_df = harm_df.groupby(['field_id', 'year'])[HARM_COLS].sum().reset_index()
print(f'{len(harm_df):,} 行')
all_data = all_data.merge(harm_df, on=['field_id', 'year'], how='left')
for c in HARM_COLS:
    all_data[c] = all_data[c].fillna(0)
print(f'病害結合後: {len(all_data)} 件')

# ── 8. LightGBM モデル定義 ─────────────────────────────────────────────────
CAT_COLS = ['breed']  # LightGBMのネイティブカテゴリカル指定列

def make_lgb_model(num_feat_cols: list, params: dict = None):
    """数値列はImputerを通し、breedはそのままLGBMに渡すモデル生成関数。"""
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
    """訓練・推論ヘルパー。数値列のみImputer、breed列はそのままLGBMへ渡す。"""
    X_num_tr = imputer.fit_transform(X_df_tr[num_feat_cols])
    X_num_va = imputer.transform(X_df_va[num_feat_cols])
    X_cat_tr = X_df_tr[CAT_COLS].reset_index(drop=True)
    X_cat_va = X_df_va[CAT_COLS].reset_index(drop=True)
    X_tr_full = pd.concat([pd.DataFrame(X_num_tr, columns=num_feat_cols), X_cat_tr], axis=1)
    X_va_full = pd.concat([pd.DataFrame(X_num_va, columns=num_feat_cols), X_cat_va], axis=1)
    model.fit(X_tr_full, y_tr, categorical_feature=CAT_COLS)
    return model.predict(X_va_full)

def _hpo_objective(trial, X_df_all, y_diff_curr_all, y_diff_prev_all,
                   y_true_all, y_maff_prev_all, year_all, num_feat_cols):
    """
    Optunaオブジェクティブ関数。
    - 学習: y_diff_curr（当年ベース）でモデルを訓練
    - 評価: pred_diff + maff_prev（前年）で絶対収量を復元してAbsRMSEを計算
    """
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
        imputer, model = make_lgb_model(num_feat_cols, params)
        # 訓練: 当年ベースの偏差を目的変数に使用
        pred_diff = fit_predict(
            imputer, model,
            X_df_all.iloc[tr], y_diff_curr_all[tr],
            X_df_all.iloc[va], num_feat_cols)
        # テスト復元: 前年MAFFを加えて絶対収量を復元（リーク防止）
        pred_abs = pred_diff + y_maff_prev_all[va]
        rmse_list.append(float(np.sqrt(((y_true_all[va] - pred_abs) ** 2).mean())))
    return float(np.mean(rmse_list))


# ── 9. LOYO 実行 ─────────────────────────────────────────────────────────────
all_feat_cols     = gdd_feat_cols + HARM_COLS + ['between_lines', 'between_stocks']
num_feat_cols     = all_feat_cols   # 数値特徴量列（34次元）
all_feat_cols_cat = all_feat_cols + CAT_COLS  # breed含む全特徴量（35次元）

# breedのcategory型をLOYO全体で統一
all_data['breed'] = all_data['breed'].astype('category')

# DataFrameとNumpy配列の準備
X_df          = all_data[all_feat_cols_cat].copy()
y_diff_curr   = all_data['y_diff_curr'].to_numpy(dtype=np.float32)  # 当年ベース偏差
y_diff_prev   = all_data['y_diff_prev'].to_numpy(dtype=np.float32)  # 前年ベース偏差
y_true        = all_data['yield'].to_numpy(dtype=np.float32)
y_maff_curr   = all_data['maff_current'].to_numpy(dtype=np.float32)
y_maff_prev   = all_data['maff_prev'].to_numpy(dtype=np.float32)
year_arr      = all_data['year'].to_numpy(dtype=int)
test_years    = sorted(np.unique(year_arr))

# ── 9a. ハイパーパラメータ: JSONから読み込み or Optuna探索 ────────────────────
N_TRIALS = 100
bp_path  = os.path.join(OUT_DIR, 'best_params_sameyear.json')

if os.path.exists(bp_path):
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
    print(f'\n{"=" * 60}')
    print(f'  Optuna ハイパーパラメータ探索 ({N_TRIALS} trials)')
    print(f'{"=" * 60}')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _hpo_objective(
            trial, X_df, y_diff_curr, y_diff_prev,
            y_true, y_maff_prev, year_arr, num_feat_cols),
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
    sys.stdout.flush()

print(f'\n{"=" * 60}')
print(f'  LOYO 評価（当年MAFFベース / test年は前年で代替）')
print(f'  サンプル数: {len(all_data)} 件 / 特徴量: {len(all_feat_cols_cat)} 次元')
print(f'  テスト年: {test_years}')
print(f'{"=" * 60}')

all_true_abs, all_pred_abs, all_years  = [], [], []
all_true_diff, all_pred_diff, all_maff = [], [], []
per_year_results = []

for test_year in test_years:
    tr = np.where(year_arr != test_year)[0]
    va = np.where(year_arr == test_year)[0]
    if len(tr) == 0 or len(va) == 0:
        continue

    imputer, model = make_lgb_model(num_feat_cols, best_params)

    # ★ 訓練: 当年ベース偏差（y_diff_curr）を目的変数として学習
    pred_diff_va = fit_predict(
        imputer, model,
        X_df.iloc[tr], y_diff_curr[tr],
        X_df.iloc[va], num_feat_cols)

    # ★ テスト復元: 前年MAFFを使って絶対収量を復元（データリーク防止）
    pred_abs  = pred_diff_va + y_maff_prev[va]
    true_abs  = y_true[va]
    true_diff = y_diff_prev[va]   # テスト偏差も前年ベースで比較

    rmse_abs  = float(np.sqrt(((true_abs - pred_abs) ** 2).mean()))
    mape_abs  = float(np.abs((pred_abs - true_abs) / true_abs).mean() * 100)
    ss_r      = ((true_abs - pred_abs) ** 2).sum()
    ss_t      = ((true_abs - true_abs.mean()) ** 2).sum()
    r2        = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
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
    all_maff.extend(y_maff_prev[va].tolist())   # ナイーブベース用（前年MAFF）

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

# ── ナイーブベース（前年MAFFをそのまま予測値として使用）─────────────────────
ym = np.array(all_maff)     # 各テストサンプルの前年MAFF平均（ナイーブ予測値）
rmse_naive  = float(np.sqrt(((yt - ym) ** 2).mean()))
mape_naive  = float(np.abs((ym - yt) / yt).mean() * 100)
ss_r_n      = ((yt - ym) ** 2).sum()
r2_naive    = float(1 - ss_r_n / ss_t) if ss_t > 0 else float('nan')
ss_r2_n     = (yd ** 2).sum()
r2_naive_d  = float(1 - ss_r2_n / ss_t2) if ss_t2 > 0 else float('nan')

print(f'\n{"=" * 60}')
print('  プール計算 サマリ')
print(f'{"=" * 60}')
print(f'  {"条件":<25} {"RMSE":>8} {"MAPE":>8} {"R2(絶対)":>10} {"R2(偏差)":>10}')
print(f'  {"-"*63}')
print(f'  {"ナイーブ（前年MAFF）":<25} {rmse_naive:>8.3f} {mape_naive:>7.2f}% {r2_naive:>10.4f} {r2_naive_d:>10.4f}')
print(f'  {"LightGBM（偏差予測→復元）":<25} {rmse_pool:>8.3f} {mape_pool:>7.2f}% {r2_pool:>10.4f} {r2_d_pool:>10.4f}')
print(f'  {"-"*63}')
rmse_imp = rmse_naive - rmse_pool
mape_imp = mape_naive - mape_pool
print(f'  {"モデルの改善量":<25} {rmse_imp:>+8.3f} {mape_imp:>+7.2f}%')
print(f'  評価方法: LOYO年グループCV')
print(f'  学習ベースライン: 当年MAFF市町村平均')
print(f'  テストベースライン: 前年MAFF市町村平均（リーク防止）')

# ── 10. 散布図 ────────────────────────────────────────────────────────────────
YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60', 2015: '#8e44ad'}
yrs = np.array(all_years)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f8f9fa')
fig.suptitle('LightGBM LOYO: 分散予測モデル\n（学習=当年MAFF / テスト=前年MAFFで復元）',
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
out_path = os.path.join(OUT_DIR, 'eval_deviation_sameyear_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  散布図 -> {out_path}')
print('\n完了')
