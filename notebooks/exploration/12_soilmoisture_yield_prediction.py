"""
12_soilmoisture_yield_prediction.py
=====================================
土壌水分量（VWC）を特徴量として収量を予測する回帰モデル。

[特徴量エンジニアリング]
  VWC時系列からDAP%ベースで以下を抽出（1圃場×年 → 1行）:
    vwc_early : DAP 20〜50% の平均VWC（生育前半〜出穂期）
    vwc_mid   : DAP 50〜75% の平均VWC（登熟前半）
    vwc_late  : DAP 75〜100%の平均VWC（登熟後半〜収穫前）
    vwc_mean  : 全測定期間の平均VWC
    vwc_std   : 全測定期間の標準偏差
    vwc_slope : VWCの傾き（OLS推定、DAP%単位）
    vwc_cv    : 変動係数（vwc_std / vwc_mean）

  ※ 測定期間が短く各区間でデータがない場合は NaN → 圃場全体平均で補完

[モデル]
  - Ridge回帰      （線形ベースライン）
  - Random Forest  （非線形・頑健）
  - SVR            （少標本向き）
  - XGBoost        （高精度）

[データ分割]
  LOYO（Leave-One-Year-Out）: 2016 / 2017 / 2018 の3フォールド
  各フォールドでハイパーパラメータを GridSearchCV で調整（訓練側のみ）

[出力]
  outputs/soil_prediction/
    12_loyo_results.png      (LOYO各フォールドの実測vs予測散布図)
    12_loyo_summary.csv      (モデル×フォールドの精度サマリ)
    12_feature_importance.png (重要特徴量ランキング: RF, XGB)
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('XGBoost が見つかりません（pip install xgboost で導入可）')

warnings.filterwarnings('ignore')

for _fn in ['IPAexGothic', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# =============================================================================
# 定数
# =============================================================================
_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
OUT_DIR = os.path.join(_ROOT, 'outputs', 'soil_prediction')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS = [2016, 2017, 2018]

# DAP%区間定義
PERIODS = {
    'vwc_early': (20, 50),
    'vwc_mid':   (50, 75),
    'vwc_late':  (75, 100),
}
FEATURE_COLS = ['vwc_early', 'vwc_mid', 'vwc_late',
                'vwc_mean', 'vwc_std', 'vwc_slope', 'vwc_cv']

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'

# =============================================================================
# 1. データ読み込み
# =============================================================================
print('=' * 60)
print('  データ読み込み')
print('=' * 60)

conn = sqlite3.connect(DB_PATH)
sm = pd.read_sql(
    'SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL',
    conn)
quest = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2016 AND 2018 AND yield IS NOT NULL
""", conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce').dropna().astype(int)
sm['year']     = sm['year'].astype(int)
sm['date']     = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC']      = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['field_id', 'date', 'VWC'])

quest = quest.dropna(subset=['field_id', 'yield'])
quest['field_id']     = quest['field_id'].astype(int)
quest['year']         = quest['year'].astype(int)
quest['seed_date']    = pd.to_datetime(quest['seed_date'], errors='coerce')
quest['harvest_date'] = pd.to_datetime(quest['harvest_date'], errors='coerce')
quest['growing_days'] = (quest['harvest_date'] - quest['seed_date']).dt.days
quest = quest[quest['growing_days'] > 0].copy()

print(f'  SolidMoisture: {len(sm):,} レコード')
print(f'  収量データ: {len(quest)} 件')

# =============================================================================
# 2. DAP% 計算 & 結合
# =============================================================================
df = sm.merge(
    quest[['field_id', 'year', 'yield', 'seed_date', 'growing_days']],
    on=['field_id', 'year'], how='inner'
)
df['DAP_pct'] = (df['date'] - df['seed_date']).dt.days / df['growing_days'] * 100

print(f'  結合済み: {len(df):,} レコード / {df["field_id"].nunique()} 圃場')

# =============================================================================
# 3. 特徴量エンジニアリング（圃場×年 → 1行）
# =============================================================================
print('\n  特徴量エンジニアリング中...')

feature_rows = []
for (fid, yr), grp in df.groupby(['field_id', 'year']):
    grp = grp.sort_values('DAP_pct')
    vwc = grp['VWC'].values
    dap = grp['DAP_pct'].values
    yld = grp['yield'].iloc[0]

    # 区間平均
    row = {'field_id': fid, 'year': yr, 'yield': yld}
    for pname, (lo, hi) in PERIODS.items():
        mask = (dap >= lo) & (dap < hi)
        row[pname] = float(vwc[mask].mean()) if mask.sum() > 0 else np.nan

    # 全期間統計
    row['vwc_mean']  = float(vwc.mean())
    row['vwc_std']   = float(vwc.std())
    row['vwc_cv']    = row['vwc_std'] / row['vwc_mean'] if row['vwc_mean'] > 0 else np.nan

    # 傾き（VWC vs DAP%の線形回帰）
    if len(dap) >= 3:
        slope, *_ = linregress(dap, vwc)
        row['vwc_slope'] = float(slope)
    else:
        row['vwc_slope'] = np.nan

    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows)

# NaN補完: 区間データがない場合は vwc_mean で埋める
for col in ['vwc_early', 'vwc_mid', 'vwc_late']:
    nan_mask = feat_df[col].isna()
    feat_df.loc[nan_mask, col] = feat_df.loc[nan_mask, 'vwc_mean']

# それでも残るNaN（slope, cv など）を列平均で補完
for col in FEATURE_COLS:
    feat_df[col] = feat_df[col].fillna(feat_df[col].median())

print(f'  特徴量行列サイズ: {feat_df.shape}')
print(f'  年別件数:\n{feat_df.groupby("year").size().to_string()}')

# 特徴量統計
print('\n  特徴量の基本統計:')
print(feat_df[FEATURE_COLS].describe().round(4).to_string())

# =============================================================================
# 4. モデル定義
# =============================================================================
models = {
    'Ridge': {
        'model': Ridge(),
        'params': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'scale': True,
    },
    'SVR': {
        'model': SVR(),
        'params': {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'],
                   'epsilon': [0.1, 1, 5]},
        'scale': True,
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [100, 200],
                   'max_depth': [None, 5, 10],
                   'min_samples_leaf': [1, 3, 5]},
        'scale': False,
    },
}
if HAS_XGB:
    models['XGBoost'] = {
        'model': XGBRegressor(random_state=42, verbosity=0),
        'params': {'n_estimators': [100, 200],
                   'max_depth': [3, 5, 7],
                   'learning_rate': [0.05, 0.1, 0.2],
                   'subsample': [0.8, 1.0]},
        'scale': False,
    }

# =============================================================================
# 5. LOYO 交差検証
# =============================================================================
print('\n' + '=' * 60)
print('  LOYO 交差検証')
print('=' * 60)

loyo_results = []
fold_preds   = {m: {} for m in models}  # {model_name: {year: (y_true, y_pred)}}
feat_importances = {}

for test_year in TARGET_YEARS:
    train_years = [y for y in TARGET_YEARS if y != test_year]
    train_df = feat_df[feat_df['year'].isin(train_years)]
    test_df  = feat_df[feat_df['year'] == test_year]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['yield'].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df['yield'].values

    print(f'\n  [Fold: test={test_year}]  train={train_years}  '
          f'n_train={len(X_train)} / n_test={len(X_test)}')

    for mname, mdict in models.items():
        # スケーリング
        if mdict['scale']:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train)
            X_te = sc.transform(X_test)
        else:
            X_tr, X_te = X_train, X_test

        # GridSearchCV（訓練データ内でCV）
        inner_cv = KFold(n_splits=min(3, len(train_years)),
                         shuffle=True, random_state=0)
        gs = GridSearchCV(mdict['model'], mdict['params'],
                          cv=inner_cv, scoring='neg_root_mean_squared_error',
                          n_jobs=-1, refit=True)
        gs.fit(X_tr, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mae  = float(mean_absolute_error(y_test, y_pred))

        print(f'    {mname:<15}: RMSE={rmse:.2f}  R2={r2:.3f}  MAE={mae:.2f}'
              f'  best={gs.best_params_}')

        loyo_results.append({
            'model': mname, 'test_year': test_year,
            'n_train': len(X_train), 'n_test': len(X_test),
            'RMSE': rmse, 'R2': r2, 'MAE': mae,
        })
        fold_preds[mname][test_year] = (y_test.copy(), y_pred.copy())

        # 特徴量重要度の収集（RF, XGB）
        if mname in ['RandomForest', 'XGBoost']:
            imp = best.feature_importances_
            if mname not in feat_importances:
                feat_importances[mname] = np.zeros(len(FEATURE_COLS))
            feat_importances[mname] += imp / len(TARGET_YEARS)

# =============================================================================
# 6. 結果サマリ
# =============================================================================
results_df = pd.DataFrame(loyo_results)
mean_df = results_df.groupby('model')[['RMSE','R2','MAE']].mean().round(3)
print('\n' + '=' * 60)
print('  LOYO 平均精度サマリ')
print('=' * 60)
print(mean_df.sort_values('RMSE').to_string())

csv_path = os.path.join(OUT_DIR, '12_loyo_summary.csv')
results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n  CSV保存: {csv_path}')

# =============================================================================
# 7. 可視化: 実測 vs 予測 散布図（LOYO）
# =============================================================================
n_models = len(models)
n_years  = len(TARGET_YEARS)
fig, axes = plt.subplots(n_models, n_years,
                         figsize=(5 * n_years, 4.5 * n_models))
if n_models == 1:
    axes = axes.reshape(1, -1)

for mi, mname in enumerate(models):
    for yi, yr in enumerate(TARGET_YEARS):
        ax = axes[mi][yi]
        if yr not in fold_preds[mname]:
            ax.set_visible(False)
            continue
        yt, yp = fold_preds[mname][yr]
        all_vals = np.concatenate([yt, yp])
        lo_v, hi_v = all_vals.min(), all_vals.max()

        ax.scatter(yt, yp, s=30, alpha=0.7, color='#2C3E50', edgecolors='white', lw=0.3)
        ax.plot([lo_v, hi_v], [lo_v, hi_v], 'r--', lw=1.2, alpha=0.6)

        rmse = next(r['RMSE'] for r in loyo_results
                    if r['model'] == mname and r['test_year'] == yr)
        r2   = next(r['R2']   for r in loyo_results
                    if r['model'] == mname and r['test_year'] == yr)
        ax.set_title(f'{mname} — test {yr}\nRMSE={rmse:.1f}  R²={r2:.3f}',
                     fontsize=9)
        ax.set_xlabel('実測収量 (kg/10a)', fontsize=8)
        ax.set_ylabel('予測収量 (kg/10a)', fontsize=8)
        ax.grid(ls='--', alpha=0.3)

fig.suptitle('LOYO 交差検証: 実測 vs 予測収量（VWC特徴量のみ）', fontsize=13)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, '12_loyo_results.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'  散布図保存: {out_path}')

# =============================================================================
# 8. 特徴量重要度（RF, XGB）
# =============================================================================
if feat_importances:
    n_imp = len(feat_importances)
    fig, axes = plt.subplots(1, n_imp, figsize=(7 * n_imp, 5))
    if n_imp == 1:
        axes = [axes]
    for ax, (mname, imp) in zip(axes, feat_importances.items()):
        order = np.argsort(imp)[::-1]
        ax.barh([FEATURE_COLS[o] for o in order[::-1]],
                imp[order[::-1]], color='#3498DB', alpha=0.8)
        ax.set_title(f'{mname} — 特徴量重要度（LOYO平均）', fontsize=10)
        ax.set_xlabel('重要度', fontsize=9)
        ax.grid(axis='x', ls='--', alpha=0.4)
    plt.tight_layout()
    imp_path = os.path.join(OUT_DIR, '12_feature_importance.png')
    plt.savefig(imp_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  重要度グラフ保存: {imp_path}')

print()
print('=' * 60)
print('  完了！')
print(f'  出力先: {OUT_DIR}')
print('=' * 60)
