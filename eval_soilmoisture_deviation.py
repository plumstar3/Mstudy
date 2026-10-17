"""
eval_soilmoisture_deviation.py
================================================================
[目的]
  eval_yield_deviation.py と同一の評価フレームワーク（LightGBM LOYO + Optuna）で
  入力特徴量を「土壌水分量（VWC）由来の特徴量のみ」に差し替えて比較する。

  目的変数: 収量偏差 y_diff = yield - MAFF前年移動平均
  特徴量  : VWC（土壌水分量）から DAP% 基準で抽出した特徴量のみ（breedは除外）
    vwc_early  : DAP 20〜50% の平均VWC（生育前半）
    vwc_mid    : DAP 50〜75% の平均VWC（登熟前半）
    vwc_late   : DAP 75〜100%の平均VWC（登熟後半）
    vwc_mean   : 全期間平均VWC
    vwc_std    : 全期間標準偏差
    vwc_slope  : VWCの傾き（DAP%単位）
    vwc_cv     : 変動係数（std/mean）

  母集団: SolidMoistureテーブルに記録がある圃場のみ（2016〜2018年）

[eval_yield_deviation.py からの変更点]
  - 入力特徴量を VWC 7次元 + breed に変更
  - 対象年: 2016〜2018（SolidMoisture なし→2015除外）
  - 気象・病害・NDVI 特徴量は使用しない

[出力]
  outputs/yield_pred_vwc/
    eval_soilmoisture_deviation_loyo.png
    loyo_summary_vwc.csv
"""

import sqlite3, os, warnings, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import linregress
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── フォント設定 ──────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定（eval_yield_deviation.py と同一） ────────────────────
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
MAFF_CSV    = os.path.join('data', 'raw', 'summary-soy-2010-2018.csv')
MAFF_WINDOW = 1
USE_QUANTILE_MAPPING = False
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_vwc')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS = [2016, 2017, 2018]   # 2015はSolidMoistureなし
RANDOM_STATE = 42
N_TRIALS     = 100

# VWC特徴量: DAP% 5%刻みビン（30〜85% の 12ビン = カバレッジ主要帯）
VWC_BIN_EDGES = list(range(30, 90, 5))   # [30, 35, 40, ..., 85]
VWC_BIN_COLS  = [f'vwc_dap{lo:03d}' for lo in VWC_BIN_EDGES[:-1]]   # vwc_dap030 〜 vwc_dap085
# 統計量特徴量（時系列全体の要約）
VWC_STAT_COLS = ['vwc_mean', 'vwc_std', 'vwc_slope', 'vwc_cv']
# VWC_FEAT_COLS = VWC_BIN_COLS + VWC_STAT_COLS   # 計 12 + 4 = 16次元
VWC_FEAT_COLS = VWC_BIN_COLS
CAT_COLS      = []   # 品種(breed)は特徴量から除外

# ── 1. 収量・位置データ ───────────────────────────────────────
print('=' * 60)
print('  土壌水分量 VWC 特徴量版 偏差予測モデル（LightGBM LOYO）')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon, breed,
           seed_date, harvest_date
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2016 AND 2018''', conn)

# SolidMoisture に存在する field_id × year のみ抽出
sm_ids = pd.read_sql(
    'SELECT DISTINCT field_id, year FROM SolidMoisture WHERE field_id IS NOT NULL',
    conn)
conn.close()

quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)
quest_df['yield']    = quest_df['yield'].astype(float)
quest_df['lat']      = pd.to_numeric(quest_df['lat'], errors='coerce')
quest_df['lon']      = pd.to_numeric(quest_df['lon'], errors='coerce')
quest_df['breed']    = quest_df['breed'].fillna('Unknown').astype(str).str.strip()
quest_df['seed_date']    = pd.to_datetime(quest_df['seed_date'], errors='coerce')
quest_df['harvest_date'] = pd.to_datetime(quest_df['harvest_date'], errors='coerce')
quest_df['growing_days'] = (quest_df['harvest_date'] - quest_df['seed_date']).dt.days
quest_df = quest_df[quest_df['growing_days'] > 0].copy()

sm_ids['field_id'] = pd.to_numeric(sm_ids['field_id'], errors='coerce').dropna().astype(int)
sm_ids['year']     = sm_ids['year'].astype(int)
sm_key = set(zip(sm_ids['field_id'], sm_ids['year']))
quest_df = quest_df[
    quest_df.apply(lambda r: (r['field_id'], r['year']) in sm_key, axis=1)
].dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

print(f'質問票データ（SolidMoisture圃場のみ）: {len(quest_df)} 件')
print(f'年別: {quest_df.groupby("year").size().to_dict()}')

# ── 2. 逆ジオコーディング ────────────────────────────────────
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city', 'muniCd']].drop_duplicates('field_id')
geo_df['muniCd'] = pd.to_numeric(geo_df['muniCd'], errors='coerce').astype('Int64')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')
print(f'市町村紐付け: {quest_df["city"].notna().sum()} 件成功')

# ── 3. 農林水産省データ（前年移動平均ベースライン）───────────
maff_raw = pd.read_csv(MAFF_CSV, encoding='cp932')
maff_raw.columns = ['year', 'city', 'maff_yield', 'muniCd']
maff_raw['year']       = maff_raw['year'].astype(int)
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['muniCd']     = pd.to_numeric(maff_raw['muniCd'], errors='coerce').astype('Int64')
maff_valid = maff_raw.dropna(subset=['maff_yield'])
print(f'農水省データ: {len(maff_valid)} 件')

quest_df['muniCd'] = pd.to_numeric(quest_df['muniCd'], errors='coerce').astype('Int64')
maff_rows = []
for target_year in TARGET_YEARS:
    hist_years = list(range(target_year - MAFF_WINDOW, target_year))
    hist = maff_valid[maff_valid['year'].isin(hist_years)]
    muni_avg = hist.groupby('muniCd')['maff_yield'].mean().reset_index()
    muni_avg['join_year'] = target_year
    maff_rows.append(muni_avg)
maff_5yr = pd.concat(maff_rows, ignore_index=True)
maff_5yr['muniCd'] = maff_5yr['muniCd'].astype('Int64')

quest_df = quest_df.merge(
    maff_5yr[['join_year', 'muniCd', 'maff_yield']],
    left_on=['year', 'muniCd'], right_on=['join_year', 'muniCd'],
    how='left').drop(columns='join_year')

valid_df = quest_df.dropna(subset=['maff_yield']).reset_index(drop=True)
valid_df['y_diff'] = valid_df['yield'] - valid_df['maff_yield']
print(f'\n有効サンプル: {len(valid_df)} 件（MAFF未マッチ除外後）')
print(f'年別: {valid_df.groupby("year").size().to_dict()}')
print(f'y_diff 統計: mean={valid_df["y_diff"].mean():.1f}  '
      f'std={valid_df["y_diff"].std():.1f}  '
      f'min={valid_df["y_diff"].min():.1f}  '
      f'max={valid_df["y_diff"].max():.1f}')

# ── 4. VWC特徴量エンジニアリング ────────────────────────────
print('\nVWC特徴量生成中...')
conn = sqlite3.connect(FIELD_DB)
sm_all = pd.read_sql(
    'SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL',
    conn)
conn.close()
sm_all['field_id'] = pd.to_numeric(sm_all['field_id'], errors='coerce').dropna().astype(int)
sm_all['year']     = sm_all['year'].astype(int)
sm_all['date']     = pd.to_datetime(sm_all['date'], errors='coerce')
sm_all['VWC']      = pd.to_numeric(sm_all['VWC'], errors='coerce')
sm_all = sm_all.dropna(subset=['field_id', 'date', 'VWC'])

# valid_dfのseed_dateを結合してDAP%を計算
seed_info = valid_df[['field_id', 'year', 'seed_date', 'growing_days']].drop_duplicates()
sm_merged = sm_all.merge(seed_info, on=['field_id', 'year'], how='inner')
sm_merged['DAP_pct'] = (sm_merged['date'] - sm_merged['seed_date']).dt.days / sm_merged['growing_days'] * 100

feat_rows = []
for (fid, yr), grp in sm_merged.groupby(['field_id', 'year']):
    grp = grp.sort_values('DAP_pct')
    vwc = grp['VWC'].values
    dap = grp['DAP_pct'].values

    row = {'field_id': fid, 'year': yr}

    # 5%刻みビン平均（DAP 20〜100%）
    for lo, col in zip(VWC_BIN_EDGES[:-1], VWC_BIN_COLS):
        hi = lo + 5
        mask = (dap >= lo) & (dap < hi)
        row[col] = float(vwc[mask].mean()) if mask.sum() > 0 else np.nan

    # 統計量特徴量
    row['vwc_mean']  = float(vwc.mean())
    row['vwc_std']   = float(vwc.std())
    row['vwc_cv']    = row['vwc_std'] / row['vwc_mean'] if row['vwc_mean'] > 0 else np.nan
    if len(dap) >= 3:
        slope, *_ = linregress(dap, vwc)
        row['vwc_slope'] = float(slope)
    else:
        row['vwc_slope'] = np.nan
    feat_rows.append(row)

vwc_feat_df = pd.DataFrame(feat_rows)

# ビン欠損補完: 同一圃場の vwc_mean で埋め、残りは列中央値
for col in VWC_BIN_COLS:
    nan_mask = vwc_feat_df[col].isna()
    vwc_feat_df.loc[nan_mask, col] = vwc_feat_df.loc[nan_mask, 'vwc_mean']
for col in VWC_FEAT_COLS:
    vwc_feat_df[col] = vwc_feat_df[col].fillna(vwc_feat_df[col].median())

print(f'VWC特徴量行列: {vwc_feat_df.shape}')

# valid_dfに結合
all_data = valid_df.merge(vwc_feat_df[['field_id', 'year'] + VWC_FEAT_COLS],
                          on=['field_id', 'year'], how='inner')
print(f'VWC特徴量結合後: {len(all_data)} 件')
print(f'特徴量: {VWC_FEAT_COLS + CAT_COLS}')

# ── 5. モデル関数（eval_yield_deviation.py と同一） ──────────
def make_lgb_model(num_feat_cols: list, params: dict = None):
    if params is None:
        params = {}
    imputer = SimpleImputer(strategy='mean')
    model   = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)
    return imputer, model

def fit_predict(imputer, model, X_df_tr, y_tr, X_df_va, num_feat_cols):
    X_num_tr = imputer.fit_transform(X_df_tr[num_feat_cols])
    X_num_va = imputer.transform(X_df_va[num_feat_cols])
    X_tr_full = pd.DataFrame(X_num_tr, columns=num_feat_cols)
    X_va_full = pd.DataFrame(X_num_va, columns=num_feat_cols)
    if CAT_COLS:
        X_tr_full = pd.concat([X_tr_full, X_df_tr[CAT_COLS].reset_index(drop=True)], axis=1)
        X_va_full = pd.concat([X_va_full, X_df_va[CAT_COLS].reset_index(drop=True)], axis=1)
        model.fit(X_tr_full, y_tr, categorical_feature=CAT_COLS)
    else:
        model.fit(X_tr_full, y_tr)
    return model.predict(X_va_full)

def fit_quantile_map(y_maff_tr, y_true_tr, n_quantiles=100):
    quantiles = np.linspace(0, 100, n_quantiles)
    maff_q = np.percentile(y_maff_tr, quantiles)
    true_q = np.percentile(y_true_tr, quantiles)
    _, idx = np.unique(maff_q, return_index=True)
    return interp1d(maff_q[idx], true_q[idx], kind='linear', fill_value='extrapolate')

def _hpo_objective(trial, X_all, y_diff_all, y_true_all, y_maff_all, year_all, num_feat_cols):
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
        y_diff_tr     = y_diff_all[tr]
        maff_for_pred = y_maff_all[va]
        imputer, model = make_lgb_model(num_feat_cols, params)
        pred_diff = fit_predict(imputer, model,
                                X_all.iloc[tr], y_diff_tr,
                                X_all.iloc[va], num_feat_cols)
        pred_abs = pred_diff + maff_for_pred
        rmse_list.append(float(np.sqrt(((y_true_all[va] - pred_abs) ** 2).mean())))
    return float(np.mean(rmse_list))

# ── 6. LOYO 準備 ─────────────────────────────────────────────
num_feat_cols     = VWC_FEAT_COLS
all_feat_cols_cat = VWC_FEAT_COLS + CAT_COLS

X_num_all = all_data[VWC_FEAT_COLS].to_numpy(dtype=np.float64)
X_df      = all_data[all_feat_cols_cat].copy()
y_diff    = all_data['y_diff'].to_numpy(dtype=np.float32)
y_true    = all_data['yield'].to_numpy(dtype=np.float32)
y_maff    = all_data['maff_yield'].to_numpy(dtype=np.float32)
year_arr  = all_data['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))

print(f'\n特徴量次元数: {len(all_feat_cols_cat)} 次元')
print(f'  VWC: {len(VWC_FEAT_COLS)}（breedなし）')

# ── 7. LGBM ハイパーパラメータ（Optuna） ─────────────────────
bp_path = os.path.join(OUT_DIR, 'best_params_deviation.json')
if os.path.exists(bp_path):
    with open(bp_path) as f:
        saved = json.load(f)
    best_params_lgb = saved['best_params']
    print(f'\nLGBM パラメータ読み込み: {bp_path}  (HPO RMSE={saved.get("best_rmse", float("nan")):.4f})')
else:
    print(f'\nOptuna ハイパーパラメータ探索 ({N_TRIALS} trials)...')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _hpo_objective(
            trial, X_df, y_diff, y_true, y_maff, year_arr, num_feat_cols),
        n_trials=N_TRIALS, show_progress_bar=False)
    best_params_lgb = study.best_params
    best_rmse = study.best_value
    print(f'  最良 RMSE: {best_rmse:.4f}')
    with open(bp_path, 'w') as f:
        json.dump({'best_rmse': best_rmse, 'best_params': best_params_lgb}, f, indent=2)
    print(f'  パラメータ保存: {bp_path}')

# ── 8. 3モデル LOYO ループ ────────────────────────────────────
print(f'\n{"=" * 60}')
print(f'  LOYO 評価: LGBM / Ridge / Lasso')
print(f'  サンプル数: {len(all_data)} 件 / 特徴量: {len(VWC_FEAT_COLS)}')
print(f'  テスト年: {test_years}')
print(f'{"=" * 60}')

# alphaの候補（Ridge/Lasso共通）
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# モデルごとに予測を収集
model_preds = {m: {'true_abs': [], 'pred_abs': [], 'years': [],
                    'true_diff': [], 'pred_diff': []}
               for m in ['LGBM', 'Ridge', 'Lasso']}
per_year_rows = []

for test_year in test_years:
    tr = np.where(year_arr != test_year)[0]
    va = np.where(year_arr == test_year)[0]
    if len(tr) == 0 or len(va) == 0:
        continue

    y_diff_tr        = y_diff[tr]
    y_diff_va        = y_diff[va]
    maff_for_restore = y_maff[va]
    true_abs         = y_true[va]

    print(f'\n  [test={test_year}]  train={len(tr)}件 / test={len(va)}件')

    # ── LGBM ──────────────────────────────────────────────────
    imp_lgb, mdl_lgb = make_lgb_model(num_feat_cols, best_params_lgb)
    pred_diff_lgb    = fit_predict(imp_lgb, mdl_lgb,
                                   X_df.iloc[tr], y_diff_tr,
                                   X_df.iloc[va], num_feat_cols)
    pred_abs_lgb     = pred_diff_lgb + maff_for_restore
    rmse_lgb  = float(np.sqrt(((true_abs - pred_abs_lgb) ** 2).mean()))
    mape_lgb  = float(np.abs((pred_abs_lgb - true_abs) / true_abs).mean() * 100)
    r2_lgb    = float(1 - ((true_abs - pred_abs_lgb)**2).sum() /
                      max(((true_abs - true_abs.mean())**2).sum(), 1e-9))
    print(f'    LGBM  : AbsRMSE={rmse_lgb:7.3f}  MAPE={mape_lgb:5.2f}%  R2={r2_lgb:+.4f}')

    # ── Ridge ─────────────────────────────────────────────────
    imp_ridge = SimpleImputer(strategy='mean')
    sc_ridge  = StandardScaler()
    X_tr_r    = sc_ridge.fit_transform(imp_ridge.fit_transform(X_num_all[tr]))
    X_va_r    = sc_ridge.transform(imp_ridge.transform(X_num_all[va]))
    ridge     = RidgeCV(alphas=ALPHAS, cv=min(3, len(np.unique(year_arr[tr]))))
    ridge.fit(X_tr_r, y_diff_tr)
    pred_diff_ridge = ridge.predict(X_va_r)
    pred_abs_ridge  = pred_diff_ridge + maff_for_restore
    rmse_ridge = float(np.sqrt(((true_abs - pred_abs_ridge) ** 2).mean()))
    mape_ridge = float(np.abs((pred_abs_ridge - true_abs) / true_abs).mean() * 100)
    r2_ridge   = float(1 - ((true_abs - pred_abs_ridge)**2).sum() /
                       max(((true_abs - true_abs.mean())**2).sum(), 1e-9))
    print(f'    Ridge : AbsRMSE={rmse_ridge:7.3f}  MAPE={mape_ridge:5.2f}%  R2={r2_ridge:+.4f}  alpha={ridge.alpha_:.4f}')

    # ── Lasso ─────────────────────────────────────────────────
    imp_lasso = SimpleImputer(strategy='mean')
    sc_lasso  = StandardScaler()
    X_tr_l    = sc_lasso.fit_transform(imp_lasso.fit_transform(X_num_all[tr]))
    X_va_l    = sc_lasso.transform(imp_lasso.transform(X_num_all[va]))
    lasso     = LassoCV(alphas=ALPHAS, cv=min(3, len(np.unique(year_arr[tr]))),
                        max_iter=5000, random_state=RANDOM_STATE)
    lasso.fit(X_tr_l, y_diff_tr)
    pred_diff_lasso = lasso.predict(X_va_l)
    pred_abs_lasso  = pred_diff_lasso + maff_for_restore
    rmse_lasso = float(np.sqrt(((true_abs - pred_abs_lasso) ** 2).mean()))
    mape_lasso = float(np.abs((pred_abs_lasso - true_abs) / true_abs).mean() * 100)
    r2_lasso   = float(1 - ((true_abs - pred_abs_lasso)**2).sum() /
                       max(((true_abs - true_abs.mean())**2).sum(), 1e-9))
    n_nonzero  = int((lasso.coef_ != 0).sum())
    print(f'    Lasso : AbsRMSE={rmse_lasso:7.3f}  MAPE={mape_lasso:5.2f}%  R2={r2_lasso:+.4f}  alpha={lasso.alpha_:.4f}  '
          f'非ゼロ係数={n_nonzero}/{len(VWC_FEAT_COLS)}')

    per_year_rows.append({'year': test_year, 'n': len(va),
                          'RMSE_LGBM': rmse_lgb,   'MAPE_LGBM': mape_lgb,   'R2_LGBM': r2_lgb,
                          'RMSE_Ridge': rmse_ridge, 'MAPE_Ridge': mape_ridge, 'R2_Ridge': r2_ridge,
                          'RMSE_Lasso': rmse_lasso, 'MAPE_Lasso': mape_lasso, 'R2_Lasso': r2_lasso})

    for mname, pred_abs, pred_diff in [
        ('LGBM',  pred_abs_lgb,   pred_diff_lgb),
        ('Ridge', pred_abs_ridge, pred_diff_ridge),
        ('Lasso', pred_abs_lasso, pred_diff_lasso),
    ]:
        model_preds[mname]['true_abs'].extend(true_abs.tolist())
        model_preds[mname]['pred_abs'].extend(pred_abs.tolist())
        model_preds[mname]['years'].extend([test_year] * len(va))
        model_preds[mname]['true_diff'].extend(y_diff_va.tolist())
        model_preds[mname]['pred_diff'].extend(pred_diff.tolist())

# ── プール計算 & サマリ ───────────────────────────────────────
print(f'\n{"=" * 60}')
print('  プール計算 サマリ（全年合算）')
print(f'{"=" * 60}')

# ナイーブ（LGBM側の true_abs と maff を使う）
yt_all = np.array(model_preds['LGBM']['true_abs'])
ym_all = y_maff[np.concatenate([np.where(year_arr == yr)[0] for yr in test_years])]
ss_t   = ((yt_all - yt_all.mean()) ** 2).sum()
rmse_naive = float(np.sqrt(((yt_all - ym_all) ** 2).mean()))
mape_naive = float(np.abs((ym_all - yt_all) / yt_all).mean() * 100)
r2_naive   = float(1 - ((yt_all - ym_all)**2).sum() / max(ss_t, 1e-9))

print(f'  {"モデル":<22} {"RMSE":>8} {"MAPE":>8} {"R2(絶対)":>10}')
print(f'  {"-"*50}')
print(f'  {"ナイーブ(前年MAFF)":<22} {rmse_naive:>8.3f} {mape_naive:>7.2f}% {r2_naive:>10.4f}')

summary_rows = []
for mname in ['LGBM', 'Ridge', 'Lasso']:
    yp = np.array(model_preds[mname]['pred_abs'])
    yt = np.array(model_preds[mname]['true_abs'])
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    mape = float(np.abs((yp - yt) / yt).mean() * 100)
    r2   = float(1 - ((yt - yp)**2).sum() / max(ss_t, 1e-9))
    print(f'  {mname:<22} {rmse:>8.3f} {mape:>7.2f}% {r2:>10.4f}  (改善: {rmse_naive - rmse:+.2f})')
    summary_rows.append({'model': mname, 'RMSE': rmse, 'MAPE': mape, 'R2': r2})

results_df = pd.DataFrame(per_year_rows)
csv_path = os.path.join(OUT_DIR, 'loyo_summary_vwc.csv')
results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
pd.DataFrame(summary_rows + [{'model': 'Naive(前年MAFF)', 'RMSE': rmse_naive,
                              'MAPE': mape_naive, 'R2': r2_naive}]).to_csv(
    os.path.join(OUT_DIR, 'loyo_pooled_summary_vwc.csv'), index=False, encoding='utf-8-sig')
print(f'\n  CSVサマリ保存: {csv_path}')

# ── 9. Lasso係数プロット（どの時期が重要か） ─────────────────
# 最終フォールド（2016テスト時 = 最大trainセット）のLasso係数を可視化
tr_final = np.where(year_arr != 2016)[0]
imp_f = SimpleImputer(strategy='mean'); sc_f = StandardScaler()
X_f = sc_f.fit_transform(imp_f.fit_transform(X_num_all[tr_final]))
lasso_f = LassoCV(alphas=ALPHAS, cv=3, max_iter=5000, random_state=RANDOM_STATE)
lasso_f.fit(X_f, y_diff[tr_final])

coef = lasso_f.coef_
feat_names = VWC_FEAT_COLS

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f8f9fa')
fig.suptitle('VWC特徴量 LOYO 3モデル比較\n（土壌水分量のみで収量偏差を予測）',
             fontsize=13, fontweight='bold')

YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60'}

# 左・中：LGBM / Ridge の散布図
for ax, mname in zip(axes[:2], ['LGBM', 'Ridge']):
    yt = np.array(model_preds[mname]['true_abs'])
    yp = np.array(model_preds[mname]['pred_abs'])
    yrs_arr = np.array(model_preds[mname]['years'])
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    mape = float(np.abs((yp - yt) / yt).mean() * 100)
    r2   = float(1 - ((yt - yp)**2).sum() / max(ss_t, 1e-9))
    for yr, col in YEAR_COLORS.items():
        m = yrs_arr == yr
        if m.any():
            ax.scatter(yt[m], yp[m], alpha=0.7, s=50, c=col,
                       edgecolors='white', lw=0.5, label=f'{yr}(n={m.sum()})')
    mn = min(yt.min(), yp.min()) - 10; mx = max(yt.max(), yp.max()) + 10
    ax.plot([mn, mx], [mn, mx], '--', color='#555', lw=1.2)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_title(f'{mname}: RMSE={rmse:.2f}  MAPE={mape:.2f}%  R2={r2:.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Observed yield (kg/10a)', fontsize=9)
    ax.set_ylabel('Predicted yield (kg/10a)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# 右：Lasso係数バープロット（どの時期のVWCが重要か）
ax3 = axes[2]
colors_coef = ['#E74C3C' if c > 0 else '#3498DB' for c in coef]
ax3.barh(feat_names, coef, color=colors_coef, alpha=0.8)
ax3.axvline(0, color='black', lw=0.8)
ax3.set_title(f'Lasso係数（alpha={lasso_f.alpha_:.4f}）\n非ゼロ={int((coef!=0).sum())}個',
              fontsize=10, fontweight='bold')
ax3.set_xlabel('係数値（正=高VWCで収量増、負=高VWCで収量減）', fontsize=8)
ax3.grid(axis='x', ls='--', alpha=0.3)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'eval_soilmoisture_deviation_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  図 -> {out_path}')
print('\n完了')
