"""
eval_model_comparison.py
================================================================
[目的]
  eval_yield_deviation.py と同じデータ・評価設定で
  XGBoost と SVR を Optuna チューニングし LightGBM と比較する。

[モデル]
  - LightGBM : best_params_deviation.json から読み込み（なければデフォルト）
  - XGBoost  : Optuna N_TRIALS 試行でチューニング
  - SVR      : Optuna N_TRIALS 試行でチューニング

[評価設定]
  - LOYO（Leave-One-Year-Out）
  - 目的変数: y_diff = Y_survey - Y_muni_prev_year (前年市町村平均からの偏差)
  - バイアス補正: フォールド内平均補正（加法的）
  - 特徴量: 気象27次元 + 病害5次元 = 32次元
"""

import sqlite3, os, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── フォント設定 ──────────────────────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定 ─────────────────────────────────────────────────────────────────────
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB  = os.path.join('data', 'processed', 'weather_database_fieldid.db')
GDD_CSV     = os.path.join('outputs', 'gdd', 'gdd_daily.csv')
MAFF_CSV    = os.path.join('data', 'raw', 'soy-2014-2018-summary.csv')
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_v3')
LGB_PARAMS_JSON = os.path.join(OUT_DIR, 'best_params_deviation.json')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS   = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']
GDD_THRESHOLDS = [600, 1000]
RANDOM_STATE   = 42
N_TRIALS       = 100   # 各モデルの探索試行数

# ═══════════════════════════════════════════════════════════════
# 1. データ読み込み（eval_yield_deviation.py と同一ロジック）
# ═══════════════════════════════════════════════════════════════
print('=' * 60)
print('  モデル比較: LightGBM / XGBoost / SVR')
print('  （前年市町村平均偏差予測・LOYO評価）')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn)
conn.close()
quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)
quest_df['yield']    = quest_df['yield'].astype(float)
quest_df['lat']      = pd.to_numeric(quest_df['lat'], errors='coerce')
quest_df['lon']      = pd.to_numeric(quest_df['lon'], errors='coerce')
quest_df = quest_df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city']].drop_duplicates('field_id')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')

maff_raw = pd.read_csv(MAFF_CSV, encoding='utf-8-sig')
maff_raw.columns = ['year', 'city', 'maff_yield']
maff_raw['year']       = maff_raw['year'].astype(int)
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['city']       = maff_raw['city'].astype(str).str.strip()
maff_valid = maff_raw.dropna(subset=['maff_yield'])
maff_prev  = maff_valid.copy()
maff_prev['join_year'] = maff_prev['year'] + 1
quest_df['city'] = quest_df['city'].astype(str).str.strip()
quest_df = quest_df.merge(
    maff_prev[['join_year', 'city', 'maff_yield']],
    left_on=['year', 'city'], right_on=['join_year', 'city'],
    how='left').drop(columns='join_year')

valid_df = quest_df.dropna(subset=['maff_yield']).reset_index(drop=True)
valid_df['y_diff'] = valid_df['yield'] - valid_df['maff_yield']
print(f'有効サンプル: {len(valid_df)} 件  '
      f'y_diff mean={valid_df["y_diff"].mean():.1f}  std={valid_df["y_diff"].std():.1f}')

fids  = sorted(valid_df['field_id'].unique().tolist())
years = sorted(valid_df['year'].unique().tolist())

print('GDD読み込み...', end=' ')
gdd_df = pd.read_csv(GDD_CSV, encoding='utf-8-sig')
gdd_df['date'] = pd.to_datetime(gdd_df['date'])
cum_col = [c for c in gdd_df.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
th1, th2 = GDD_THRESHOLDS
gdd_df['period'] = 1
gdd_df.loc[gdd_df[cum_col] > th1, 'period'] = 2
gdd_df.loc[gdd_df[cum_col] > th2, 'period'] = 3
gdd_df = gdd_df[['field_id', 'year', 'date', 'period']]
print(f'{len(gdd_df):,} 行')

print(f'気象データ読み込み ({len(fids)} 圃場)...', end=' ')
conn_w = sqlite3.connect(WEATHER_DB)
fid_ph = ','.join(['?'] * len(fids))
yr_ph  = ','.join(f"'{y}'" for y in years)
weather_df = pd.read_sql(f'''
    SELECT field_id, date, {", ".join(WEATHER_COLS)} FROM weather_data
    WHERE field_id IN ({fid_ph})
      AND CAST(SUBSTR(date,1,4) AS INTEGER) IN ({yr_ph})
    ORDER BY field_id, date''', conn_w, params=fids)
conn_w.close()
weather_df['field_id'] = weather_df['field_id'].astype(int)
weather_df['date']     = pd.to_datetime(weather_df['date'])
print(f'{len(weather_df):,} 行')

merged_gdd = gdd_df.merge(weather_df[['field_id', 'date'] + WEATHER_COLS],
                          on=['field_id', 'date'], how='left')
grp = merged_gdd.groupby(['field_id', 'year', 'period'])[WEATHER_COLS].agg('mean')
grp_pivot = grp.unstack('period')
grp_pivot.columns = [f'{v}_p{int(p)}_mean' for v, p in grp_pivot.columns]
gdd_feat_cols = [f'{v}_p{p}_mean' for p in [1, 2, 3] for v in WEATHER_COLS]
for col in gdd_feat_cols:
    if col not in grp_pivot.columns:
        grp_pivot[col] = np.nan
feat_df = grp_pivot[gdd_feat_cols].reset_index()
all_data = valid_df.merge(feat_df, on=['field_id', 'year'], how='inner')

print('病害データ読み込み...', end=' ')
conn_h = sqlite3.connect(FIELD_DB)
harm_df = pd.read_sql(f'''
    SELECT field_id, year, {", ".join(HARM_COLS)} FROM harm
    WHERE field_id IS NOT NULL AND year BETWEEN 2015 AND 2018''', conn_h)
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

all_feat_cols = gdd_feat_cols + HARM_COLS
X        = all_data[all_feat_cols].to_numpy(dtype=np.float64)
y_diff   = all_data['y_diff'].to_numpy(dtype=np.float64)
y_true   = all_data['yield'].to_numpy(dtype=np.float64)
y_maff   = all_data['maff_yield'].to_numpy(dtype=np.float64)
year_arr = all_data['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))
print(f'特徴量: {X.shape[1]} 次元 / サンプル: {X.shape[0]} 件\n')

# ═══════════════════════════════════════════════════════════════
# 2. 共通パイプライン + LOYO 評価関数
# ═══════════════════════════════════════════════════════════════
def make_pipe(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])

def loyo_eval(pipe_factory, X, y_diff, y_true, y_maff, year_arr, label=''):
    """LOYO評価（フォールド内平均補正付き）。AbsRMSEリストとプールRMSEを返す"""
    all_true, all_pred = [], []
    per_year = []
    for test_year in sorted(np.unique(year_arr)):
        tr = np.where(year_arr != test_year)[0]
        va = np.where(year_arr == test_year)[0]
        if len(tr) == 0 or len(va) == 0:
            continue
        pipe = pipe_factory()
        mu   = float(y_diff[tr].mean())
        pipe.fit(X[tr], y_diff[tr] - mu)
        pred_abs = pipe.predict(X[va]) + mu + y_maff[va]
        true_abs = y_true[va]
        rmse = float(np.sqrt(((true_abs - pred_abs) ** 2).mean()))
        mape = float(np.abs((pred_abs - true_abs) / true_abs).mean() * 100)
        ss_r = ((true_abs - pred_abs) ** 2).sum()
        ss_t = ((true_abs - true_abs.mean()) ** 2).sum()
        r2   = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
        per_year.append({'year': test_year, 'n': len(va),
                         'RMSE': rmse, 'MAPE': mape, 'R2': r2})
        all_true.extend(true_abs.tolist())
        all_pred.extend(pred_abs.tolist())
    yt = np.array(all_true); yp = np.array(all_pred)
    pool_rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    pool_mape = float(np.abs((yp - yt) / yt).mean() * 100)
    ss_r = ((yt - yp) ** 2).sum(); ss_t = ((yt - yt.mean()) ** 2).sum()
    pool_r2 = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
    return per_year, pool_rmse, pool_mape, pool_r2

def hpo_objective(trial_params_fn, X, y_diff, y_true, y_maff, year_arr):
    """Optunaオブジェクティブ: LOYO RMSEを返す"""
    factory = trial_params_fn
    _, rmse, _, _ = loyo_eval(factory, X, y_diff, y_true, y_maff, year_arr)
    return rmse

# ═══════════════════════════════════════════════════════════════
# 3. LightGBM（best_params_deviation.json があれば読み込み）
# ═══════════════════════════════════════════════════════════════
print('=' * 60)
print('  [1/3] LightGBM')
print('=' * 60)
lgb_params = {
    'num_leaves': 20, 'min_child_samples': 15, 'learning_rate': 0.08,
    'n_estimators': 100, 'reg_lambda': 10.0, 'reg_alpha': 0.0,
    'subsample': 1.0, 'colsample_bytree': 0.6,
}
if os.path.exists(LGB_PARAMS_JSON):
    with open(LGB_PARAMS_JSON) as f:
        saved = json.load(f)
    lgb_params = saved.get('best_params', lgb_params)
    print(f'  saved params from: {LGB_PARAMS_JSON}')
    print(f'  HPO RMSE (saved): {saved.get("best_rmse", "N/A")}')
else:
    print('  best_params_deviation.json が見つかりません。デフォルトパラメータを使用。')

def lgb_factory():
    return make_pipe(lgb.LGBMRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **lgb_params))

per_lgb, rmse_lgb, mape_lgb, r2_lgb = loyo_eval(
    lgb_factory, X, y_diff, y_true, y_maff, year_arr)
for r in per_lgb:
    print(f"  test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'  [Pool] RMSE={rmse_lgb:.3f}  MAPE={mape_lgb:.2f}%  R2={r2_lgb:.4f}')

# ═══════════════════════════════════════════════════════════════
# 4. XGBoost + Optuna チューニング
# ═══════════════════════════════════════════════════════════════
print(f'\n{"=" * 60}')
print(f'  [2/3] XGBoost  (Optuna {N_TRIALS} trials)')
print('=' * 60)

def xgb_objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int(  'n_estimators',     50,  500),
        'max_depth'       : trial.suggest_int(  'max_depth',         2,   10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01,  0.3, log=True),
        'subsample'       : trial.suggest_float('subsample',     0.5,   1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha'       : trial.suggest_float('reg_alpha',     0.0,  10.0),
        'reg_lambda'      : trial.suggest_float('reg_lambda',    0.0,  30.0),
        'min_child_weight': trial.suggest_int(  'min_child_weight',  1,   20),
        'gamma'           : trial.suggest_float('gamma',         0.0,   5.0),
    }
    def factory():
        return make_pipe(xgb.XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **params))
    _, rmse, _, _ = loyo_eval(factory, X, y_diff, y_true, y_maff, year_arr)
    return rmse

study_xgb = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_xgb = study_xgb.best_params
print(f'  最良HPO RMSE: {study_xgb.best_value:.4f}')
print(f'  BEST_XGB_JSON: {json.dumps(best_xgb)}')
with open(os.path.join(OUT_DIR, 'best_params_xgb.json'), 'w') as f:
    json.dump({'best_rmse': study_xgb.best_value, 'best_params': best_xgb}, f, indent=2)

def xgb_best_factory():
    return make_pipe(xgb.XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **best_xgb))

per_xgb, rmse_xgb, mape_xgb, r2_xgb = loyo_eval(
    xgb_best_factory, X, y_diff, y_true, y_maff, year_arr)
for r in per_xgb:
    print(f"  test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'  [Pool] RMSE={rmse_xgb:.3f}  MAPE={mape_xgb:.2f}%  R2={r2_xgb:.4f}')

# ═══════════════════════════════════════════════════════════════
# 5. SVR + Optuna チューニング
# ═══════════════════════════════════════════════════════════════
print(f'\n{"=" * 60}')
print(f'  [3/3] SVR (RBF kernel)  (Optuna {N_TRIALS} trials)')
print('=' * 60)

def svr_objective(trial):
    params = {
        'C'      : trial.suggest_float('C',       0.1, 2000.0, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.01,  100.0, log=True),
        'gamma'  : trial.suggest_float('gamma',  1e-5,   10.0, log=True),
    }
    def factory():
        return make_pipe(SVR(kernel='rbf', **params))
    _, rmse, _, _ = loyo_eval(factory, X, y_diff, y_true, y_maff, year_arr)
    return rmse

study_svr = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_svr.optimize(svr_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_svr = study_svr.best_params
print(f'  最良HPO RMSE: {study_svr.best_value:.4f}')
print(f'  BEST_SVR_JSON: {json.dumps(best_svr)}')
with open(os.path.join(OUT_DIR, 'best_params_svr.json'), 'w') as f:
    json.dump({'best_rmse': study_svr.best_value, 'best_params': best_svr}, f, indent=2)

def svr_best_factory():
    return make_pipe(SVR(kernel='rbf', **best_svr))

per_svr, rmse_svr, mape_svr, r2_svr = loyo_eval(
    svr_best_factory, X, y_diff, y_true, y_maff, year_arr)
for r in per_svr:
    print(f"  test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'  [Pool] RMSE={rmse_svr:.3f}  MAPE={mape_svr:.2f}%  R2={r2_svr:.4f}')

# ═══════════════════════════════════════════════════════════════
# 6. 比較サマリ
# ═══════════════════════════════════════════════════════════════
# ナイーブベース（前年市町村平均をそのまま予測値とする）
yt_all = y_true; ym_all = y_maff
naive_rmse = float(np.sqrt(((yt_all - ym_all) ** 2).mean()))
naive_mape = float(np.abs((ym_all - yt_all) / yt_all).mean() * 100)
ss_t = ((yt_all - yt_all.mean()) ** 2).sum()
naive_r2 = float(1 - ((yt_all - ym_all) ** 2).sum() / ss_t)

print(f'\n{"=" * 60}')
print('  モデル比較サマリ（プール計算）')
print('=' * 60)
rows = [
    ('ナイーブ（前年市町村平均）', naive_rmse, naive_mape, naive_r2),
    ('LightGBM（Optuna済み）',  rmse_lgb,   mape_lgb,   r2_lgb),
    ('XGBoost （Optuna済み）',  rmse_xgb,   mape_xgb,   r2_xgb),
    ('SVR/RBF （Optuna済み）',  rmse_svr,   mape_svr,   r2_svr),
]
print(f'  {"モデル":<25} {"RMSE":>8} {"MAPE":>8} {"R2(絶対)":>10}')
print(f'  {"-" * 55}')
for name, rmse, mape, r2 in rows:
    best_mark = ' ★' if rmse == min(r[1] for r in rows[1:]) else ''
    print(f'  {name:<25} {rmse:>8.3f} {mape:>7.2f}% {r2:>10.4f}{best_mark}')

# ═══════════════════════════════════════════════════════════════
# 7. 散布図（3モデル並列）
# ═══════════════════════════════════════════════════════════════
YEAR_COLORS = {2015: '#8e44ad', 2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60'}

def collect_preds(pipe_factory, X, y_diff, y_true, y_maff, year_arr):
    all_true, all_pred, all_yrs = [], [], []
    for test_year in sorted(np.unique(year_arr)):
        tr = np.where(year_arr != test_year)[0]
        va = np.where(year_arr == test_year)[0]
        if len(tr) == 0 or len(va) == 0:
            continue
        pipe = pipe_factory()
        mu   = float(y_diff[tr].mean())
        pipe.fit(X[tr], y_diff[tr] - mu)
        pred = pipe.predict(X[va]) + mu + y_maff[va]
        all_true.extend(y_true[va].tolist())
        all_pred.extend(pred.tolist())
        all_yrs.extend([test_year] * len(va))
    return np.array(all_true), np.array(all_pred), np.array(all_yrs)

models_info = [
    ('LightGBM', lgb_factory,      rmse_lgb, mape_lgb, r2_lgb),
    ('XGBoost',  xgb_best_factory, rmse_xgb, mape_xgb, r2_xgb),
    ('SVR/RBF',  svr_best_factory, rmse_svr, mape_svr, r2_svr),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f8f9fa')
fig.suptitle('LOYO Comparison: LightGBM / XGBoost / SVR\n'
             '(Target = deviation from prev-year city average)',
             fontsize=13, fontweight='bold')

for ax, (name, factory, rmse, mape, r2) in zip(axes, models_info):
    yt_p, yp_p, yrs_p = collect_preds(factory, X, y_diff, y_true, y_maff, year_arr)
    for yr_k, col in YEAR_COLORS.items():
        m = yrs_p == yr_k
        if m.any():
            ax.scatter(yt_p[m], yp_p[m], alpha=0.75, s=60, c=col,
                       edgecolors='white', linewidths=0.6, zorder=3,
                       label=f'{yr_k} (n={m.sum()})')
    mn = min(yt_p.min(), yp_p.min()) - 20
    mx = max(yt_p.max(), yp_p.max()) + 20
    ax.plot([mn, mx], [mn, mx], '--', color='#555555', lw=1.5, zorder=2)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('Observed yield (kg/10a)', fontsize=11)
    ax.set_ylabel('Predicted yield (kg/10a)', fontsize=11)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.text(0.04, 0.96,
            f'RMSE={rmse:.2f}\nMAPE={mape:.2f}%\nR2={r2:.4f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.87, edgecolor='#cccccc'))
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)
    ax.grid(True, alpha=0.25); ax.set_facecolor('#fdfdfd'); ax.set_axisbelow(True)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'model_comparison_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  散布図 -> {out_path}')
print('\n完了')
