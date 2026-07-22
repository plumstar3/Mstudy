"""
eval_lgb_objective.py
================================================================
[目的]
  eval_yield_deviation.py の最良パラメータ（best_params_deviation.json）を
  ベースに、LightGBM の損失関数（objective）を変えて LOYO 精度を比較する。

[比較する損失関数]
  - regression_l2  : L2損失（MSE、デフォルト）
  - regression_l1  : L1損失（MAE、外れ値に頑健）
  - huber          : Huber損失（L2とL1の混合、Optuna で alpha をチューニング）
  - fair           : Fair損失（L1の滑らかな近似、外れ値に頑健）
  - quantile       : 分位点回帰（alpha=0.5 で中央値予測）

[評価設定]
  - LOYO + フォールド内平均補正（eval_yield_deviation.py と同一）
  - 特徴量: 気象27次元 + 病害5次元 = 32次元
"""

import sqlite3, os, json, warnings
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
N_TRIALS_HUBER = 50   # Huber の alpha チューニング試行数

# ═══════════════════════════════════════════════════════════════
# 1. データ読み込み（eval_yield_deviation.py と同一ロジック）
# ═══════════════════════════════════════════════════════════════
print('=' * 60)
print('  LightGBM 損失関数比較')
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
print(f'有効サンプル: {len(valid_df)} 件')

fids  = sorted(valid_df['field_id'].unique().tolist())
years = sorted(valid_df['year'].unique().tolist())

gdd_df = pd.read_csv(GDD_CSV, encoding='utf-8-sig')
gdd_df['date'] = pd.to_datetime(gdd_df['date'])
cum_col = [c for c in gdd_df.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
th1, th2 = GDD_THRESHOLDS
gdd_df['period'] = 1
gdd_df.loc[gdd_df[cum_col] > th1, 'period'] = 2
gdd_df.loc[gdd_df[cum_col] > th2, 'period'] = 3
gdd_df = gdd_df[['field_id', 'year', 'date', 'period']]

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
print(f'特徴量: {X.shape[1]} 次元\n')

# ── ベースパラメータの読み込み ─────────────────────────────────────────────
base_params = {
    'num_leaves': 20, 'min_child_samples': 15, 'learning_rate': 0.08,
    'n_estimators': 100, 'reg_lambda': 10.0, 'reg_alpha': 0.0,
    'subsample': 1.0, 'colsample_bytree': 0.6,
}
if os.path.exists(LGB_PARAMS_JSON):
    with open(LGB_PARAMS_JSON) as f:
        saved = json.load(f)
    base_params = saved.get('best_params', base_params)
    print(f'  ベースパラメータ: {LGB_PARAMS_JSON}')
print(f'  base_params: {json.dumps(base_params)}\n')

# ═══════════════════════════════════════════════════════════════
# 2. 共通 LOYO 評価関数
# ═══════════════════════════════════════════════════════════════
def make_lgb_pipe(objective, alpha=None, extra=None):
    params = dict(base_params)
    params['objective'] = objective
    if alpha is not None:
        params['alpha'] = alpha     # huber: huber_delta の分位数 / quantile: 分位点
    if extra:
        params.update(extra)
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   lgb.LGBMRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params)),
    ])

def loyo_eval(pipe_factory):
    all_true, all_pred = [], []
    per_year = []
    for test_year in test_years:
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

# ═══════════════════════════════════════════════════════════════
# 3. 損失関数ごとの評価
# ═══════════════════════════════════════════════════════════════
print('=' * 60)
print('  LOYO 評価（損失関数比較）')
print('=' * 60)

results = {}

# ── (A) regression_l2（デフォルト MSE）─────────────────────────────────────
print('\n  [A] regression_l2 (MSE, default)')
per, rmse, mape, r2 = loyo_eval(lambda: make_lgb_pipe('regression_l2'))
for r in per:
    print(f"    test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'    [Pool] RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
results['(A) regression_l2\n    (MSE, default)'] = (rmse, mape, r2)

# ── (B) regression_l1（MAE）────────────────────────────────────────────────
print('\n  [B] regression_l1 (MAE / L1)')
per, rmse, mape, r2 = loyo_eval(lambda: make_lgb_pipe('regression_l1'))
for r in per:
    print(f"    test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'    [Pool] RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
results['(B) regression_l1\n    (MAE / L1)'] = (rmse, mape, r2)

# ── (C) Huber（alpha を Optuna でチューニング）───────────────────────────────
print(f'\n  [C] huber  (Optuna alpha tuning, {N_TRIALS_HUBER} trials)')

def huber_objective(trial):
    a = trial.suggest_float('alpha', 0.5, 0.99)
    _, rmse_h, _, _ = loyo_eval(lambda: make_lgb_pipe('huber', alpha=a))
    return rmse_h

study_h = optuna.create_study(direction='minimize',
                              sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_h.optimize(huber_objective, n_trials=N_TRIALS_HUBER, show_progress_bar=False)
best_alpha_h = study_h.best_params['alpha']
print(f'    最良 alpha={best_alpha_h:.4f}  HPO RMSE={study_h.best_value:.4f}')

per, rmse, mape, r2 = loyo_eval(lambda: make_lgb_pipe('huber', alpha=best_alpha_h))
for r in per:
    print(f"    test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'    [Pool] RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
results[f'(C) huber\n    (alpha={best_alpha_h:.3f})'] = (rmse, mape, r2)

# ── (D) fair ────────────────────────────────────────────────────────────────
print('\n  [D] fair  (L1の滑らかな近似)')
per, rmse, mape, r2 = loyo_eval(lambda: make_lgb_pipe('fair'))
for r in per:
    print(f"    test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'    [Pool] RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
results['(D) fair\n    (smooth L1)'] = (rmse, mape, r2)

# ── (E) quantile alpha=0.5（中央値回帰、MAE最小化と等価）────────────────────
print('\n  [E] quantile alpha=0.5  (中央値回帰 / MAE等価)')
per, rmse, mape, r2 = loyo_eval(lambda: make_lgb_pipe('quantile', alpha=0.5))
for r in per:
    print(f"    test={r['year']} n={r['n']:3d}  RMSE={r['RMSE']:7.3f}  "
          f"MAPE={r['MAPE']:6.2f}%  R2={r['R2']:7.4f}")
print(f'    [Pool] RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}')
results['(E) quantile\n    (alpha=0.5)'] = (rmse, mape, r2)

# ═══════════════════════════════════════════════════════════════
# 4. 比較サマリ
# ═══════════════════════════════════════════════════════════════
best_rmse = min(v[0] for v in results.values())
best_mape = min(v[1] for v in results.values())
best_r2   = max(v[2] for v in results.values())

print(f'\n{"=" * 60}')
print('  損失関数比較サマリ（プール計算）')
print('=' * 60)
print(f'  {"損失関数":<30} {"RMSE":>8} {"MAPE":>8} {"R2":>9}')
print(f'  {"-" * 60}')
for name, (rmse, mape, r2) in results.items():
    label = name.split('\n')[0]  # 1行目だけ使う
    marks = []
    if rmse == best_rmse: marks.append('RMSE★')
    if mape == best_mape: marks.append('MAPE★')
    if r2   == best_r2:   marks.append('R2★')
    mark_str = ' ' + ' '.join(marks) if marks else ''
    print(f'  {label:<30} {rmse:>8.3f} {mape:>7.2f}% {r2:>9.4f}{mark_str}')

# JSON保存
summary = {k.split('\n')[0]: {'RMSE': v[0], 'MAPE': v[1], 'R2': v[2]}
           for k, v in results.items()}
summary['huber_best_alpha'] = best_alpha_h
out_json = os.path.join(OUT_DIR, 'objective_comparison.json')
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n  結果保存: {out_json}')

# ═══════════════════════════════════════════════════════════════
# 5. 棒グラフ比較
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#f8f9fa')
fig.suptitle('LightGBM Objective Function Comparison (LOYO, Pool)',
             fontsize=13, fontweight='bold')

labels    = [k.split('\n')[0].replace('(', '').strip() for k in results.keys()]
rmses     = [v[0] for v in results.values()]
mapes     = [v[1] for v in results.values()]
r2s       = [v[2] for v in results.values()]
colors    = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
best_color = '#e74c3c'

for ax, vals, ylabel, title, lower_better in zip(
        axes,
        [rmses, mapes, r2s],
        ['RMSE (kg/10a)', 'MAPE (%)', 'R²'],
        ['AbsRMSE', 'MAPE', 'R² (absolute yield)'],
        [True, True, False]):
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
    best_val = min(vals) if lower_better else max(vals)
    for bar, val in zip(bars, vals):
        if val == best_val:
            bar.set_edgecolor('#222222')
            bar.set_linewidth(2.5)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) - min(vals)) * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8, rotation=15)
    ax.grid(axis='y', alpha=0.3); ax.set_facecolor('#fdfdfd')

fig.tight_layout()
out_fig = os.path.join(OUT_DIR, 'objective_comparison.png')
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  棒グラフ  -> {out_fig}')
print('\n完了')
