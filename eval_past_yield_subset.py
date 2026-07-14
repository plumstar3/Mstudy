"""
eval_past_yield_subset.py
================================================================
[目的]
  past_yield_features_v2.csv で過去記録が実際に紐づいた
  サンプルに限定して、

    (A) ベースライン: 気象135特徴量のみ
    (B) 過去情報あり: 気象135 + past_yield_mean + 過去気象135 + 過去病刷5

  の LOYO (Leave-One-Year-Out) 精度を比較し、
  過去情報特徴量の「真の有効性」を評価する。
"""

import os, sqlite3, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── 日本語フォント ─────────────────────────────────────────────────────────────
for _fn in ['Yu Gothic', 'Meiryo', 'MS Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定 ──────────────────────────────────────────────────────────────────────
FIELD_DB       = os.path.join('data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB     = os.path.join('data', 'processed', 'weather_database_fieldid.db')
GDD_CSV        = os.path.join('outputs', 'gdd', 'gdd_daily.csv')
PAST_YIELD_CSV = os.path.join('outputs', 'data_analysis', 'past_yield_features_v2.csv')
OUT_DIR        = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS     = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
# 変数ごとに取る統計量を限定（135次元 → 39次元に削減）
WEATHER_STAT_MAP = {
    'TMP_mea': ['mean'],
    'TMP_max': ['mean', 'max'],
    'TMP_min': ['mean', 'min'],
    'APCPRA':  ['mean', 'max'],
    'SSD':     ['mean'],
    'GSR':     ['mean'],
    'WIND':    ['mean', 'max'],
    'SWE':     ['mean'],
    'RH':      ['mean'],
}
GDD_THRESHOLDS = [600, 1000]
RANDOM_STATE   = 42
RIDGE_ALPHA    = 100
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']

# ── データ読み込み ─────────────────────────────────────────────────────────────
print('=' * 60)
print('  過去記録あり限定 LOYO 精度比較')
print('='*60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn)
conn.close()
quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)
quest_df['yield']    = quest_df['yield'].astype(float)
quest_df['lat']      = pd.to_numeric(quest_df['lat'], errors='coerce')
quest_df['lon']      = pd.to_numeric(quest_df['lon'], errors='coerce')
quest_df = quest_df.dropna(subset=['lat', 'lon', 'yield']).reset_index(drop=True)

# 過去収量CSV
past_df = pd.read_csv(PAST_YIELD_CSV, encoding='utf-8-sig')
past_df['field_id']        = past_df['field_id'].astype(int)
past_df['year']            = past_df['year'].astype(int)
past_df['has_past_record'] = (past_df['past_yield_n'] > 0).astype(int)

# GDD 期間ラベル
print('GDD 読み込み...', end=' ')
gdd_df = pd.read_csv(GDD_CSV, encoding='utf-8-sig')
gdd_df['date'] = pd.to_datetime(gdd_df['date'])
cum_col = 'GDD' if 'GDD' not in gdd_df.columns else 'GDD'
cum_col = [c for c in gdd_df.columns if 'GDD' in c or 'gdd' in c.lower()][-1]
th1, th2 = GDD_THRESHOLDS
gdd_df['period'] = 1
gdd_df.loc[gdd_df[cum_col] > th1, 'period'] = 2
gdd_df.loc[gdd_df[cum_col] > th2, 'period'] = 3
gdd_df = gdd_df[['field_id', 'year', 'date', 'period']]
print(f'{len(gdd_df):,} 行')

# 気象データ
fids  = sorted(quest_df['field_id'].unique().tolist())
years = sorted(quest_df['year'].unique().tolist())
print(f'気象データ読み込み ({len(fids)} 圃場)...', end=' ')
conn_w = sqlite3.connect(WEATHER_DB)
fid_ph = ','.join(['?'] * len(fids))
yr_ph  = ','.join(f"'{y}'" for y in years)
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

# GDD期間 × 気象特徴量
print('GDD期間別特徴量計算...')
merged_gdd = gdd_df.merge(weather_df[['field_id','date']+WEATHER_COLS],
                          on=['field_id','date'], how='left')
agg_dict = {v: stats for v, stats in WEATHER_STAT_MAP.items()}
grp = merged_gdd.groupby(['field_id','year','period']).agg(agg_dict)
grp_pivot = grp.unstack('period')
grp_pivot.columns = [f'{v}_p{int(p)}_{s}' for v,s,p in grp_pivot.columns]
gdd_feat_cols = [f'{v}_p{p}_{s}'
                 for p in [1,2,3]
                 for v, stats in WEATHER_STAT_MAP.items()
                 for s in stats]
for col in gdd_feat_cols:
    if col not in grp_pivot.columns:
        grp_pivot[col] = np.nan
feat_df = grp_pivot[gdd_feat_cols].reset_index()

# GDD期間別気象特徴量列名（WEATHER_STAT_MAPに基づく）
all_data = quest_df.merge(feat_df, on=['field_id','year'], how='inner')

# 過去特徴量列名を定義
PAST_WX_COLS   = [f'past_{c}' for c in gdd_feat_cols]   # 39列
PAST_HARM_COLS = [f'past_harm_{c}' for c in HARM_COLS]  # 5列
PAST_FEAT_COLS = ['past_yield_mean'] + PAST_WX_COLS + PAST_HARM_COLS  # 45列

# 過去特徴量 CSV をマージ
past_load_cols = ['field_id', 'year', 'has_past_record'] + PAST_FEAT_COLS
avail_cols = [c for c in past_load_cols if c in past_df.columns]
all_data = all_data.merge(past_df[avail_cols], on=['field_id','year'], how='left')
all_data['has_past_record'] = all_data['has_past_record'].fillna(0).astype(int)
print(f'全データ: {len(all_data)} 件')

# ── サブセットに絞る ───────────────────────────────────────────────────
subset = all_data[all_data['has_past_record'] == 1].reset_index(drop=True)
print(f'過去記録あり: {len(subset)} 件')
print(f'年別内訳: {subset.groupby("year").size().to_dict()}')
if 'past_yield_mean' in subset.columns:
    print(f'past_yield_mean: min={subset["past_yield_mean"].min():.1f}  '
          f'max={subset["past_yield_mean"].max():.1f}  '
          f'mean={subset["past_yield_mean"].mean():.1f}')

# 有効な過去特徴量列（CSVに実在するもののみ）
valid_past_cols = [c for c in PAST_FEAT_COLS if c in subset.columns]
print(f'過去特徴量有効列数: {len(valid_past_cols)} / {len(PAST_FEAT_COLS)}')

# ── モデル定義 ─────────────────────────────────────────────────────────────────
def make_pipeline(model_obj):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   model_obj),
    ])

# ── LOYO 実行 ─────────────────────────────────────────────────────────────────
def run_loyo(X, y, year_arr, label):
    """LOYO (Leave-One-Year-Out) CV を実行して指標を返す。
    指標はすべて全フォールドのプール予測値から計算（一般的な標準方式）。
    """
    test_years = sorted(np.unique(year_arr))
    models = {
        'Ridge':    make_pipeline(Ridge(alpha=RIDGE_ALPHA)),
        'LightGBM': make_pipeline(lgb.LGBMRegressor(
            # ── ランダムサーチ最良設定 (tune_lgbm.py / N_TRIALS=200) ──────
            num_leaves       = 20,     # 31 → 20（複雑さを抑制）
            min_child_samples= 15,     # 葉のサンプル数下限（過学習防止）
            learning_rate    = 0.08,   # 0.05 → 0.08
            n_estimators     = 100,    # 200 → 100
            reg_lambda       = 10.0,   # L2正則化（最重要：小サンプル向け）
            reg_alpha        = 0.0,    # L1正則化
            subsample        = 1.0,    # 行サブサンプリング
            colsample_bytree = 0.6,    # 特徴量サブサンプリング
            random_state     = RANDOM_STATE, n_jobs=-1, verbose=-1)),
    }

    print(f'\n  [{label}]  特徴量次元={X.shape[1]}')
    results   = {name: [] for name in models}
    all_preds = {name: {'true': [], 'pred': [], 'year': []} for name in models}

    for test_year in test_years:
        va_mask = year_arr == test_year
        tr_idx  = np.where(~va_mask)[0]
        va_idx  = np.where(va_mask)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        for name, pipe in models.items():
            pipe.fit(X[tr_idx], y[tr_idx])
            pred = pipe.predict(X[va_idx])
            rmse = float(np.sqrt(mean_squared_error(y[va_idx], pred)))
            mape = float(np.abs((pred - y[va_idx]) / y[va_idx]).mean() * 100)
            ss_r = ((y[va_idx]-pred)**2).sum()
            ss_t = ((y[va_idx]-y[va_idx].mean())**2).sum()
            r2   = float(1 - ss_r/ss_t) if ss_t > 0 else float('nan')
            results[name].append({'year': test_year, 'RMSE': rmse, 'MAPE': mape, 'R2': r2,
                                  'n_train': len(tr_idx), 'n_val': len(va_idx)})
            all_preds[name]['true'].extend(y[va_idx].tolist())
            all_preds[name]['pred'].extend(pred.tolist())
            all_preds[name]['year'].extend([test_year] * len(va_idx))
            print(f'    test={test_year}(n={len(va_idx):2d}) train={len(tr_idx):2d}  '
                  f'{name:<10} RMSE={rmse:7.3f} MAPE={mape:6.2f}%  R2={r2:6.4f}')

    # ── サマリ: 全フォールドのプール予測値から計算（標準方式）─────────────────
    summary = {}
    for name, year_data in results.items():
        yt_all = np.array(all_preds[name]['true'])
        yp_all = np.array(all_preds[name]['pred'])
        rmse_pool = float(np.sqrt(((yt_all - yp_all)**2).mean()))
        mape_pool = float(np.abs((yp_all - yt_all) / yt_all).mean() * 100)
        ss_r = ((yt_all - yp_all)**2).sum()
        ss_t = ((yt_all - yt_all.mean())**2).sum()
        r2_pool = float(1 - ss_r/ss_t) if ss_t > 0 else float('nan')
        summary[name] = {'RMSE': rmse_pool, 'MAPE': mape_pool, 'R2': r2_pool,
                         'per_year': year_data}
        print(f'  {name:<10} [プール] RMSE={rmse_pool:.3f}  MAPE={mape_pool:.2f}%  R2={r2_pool:.4f}')

    return summary, all_preds


X_base   = subset[gdd_feat_cols].to_numpy(dtype=np.float32)
X_with   = subset[gdd_feat_cols + valid_past_cols].to_numpy(dtype=np.float32)
y        = subset['yield'].to_numpy(dtype=np.float32)
year_arr = subset['year'].to_numpy(dtype=int)

print('\n' + '='*60)
print('  LOYO 比較（限定 / Leave-One-Year-Out）')
print('  テスト年: 2016, 2017, 2018')
print('='*60)
sum_base, preds_base = run_loyo(X_base, y, year_arr, 'ベースライン（気象のみ）')
with_label = f'+ past情報({len(valid_past_cols)}列: yield+気象135+病刷5)'
sum_with, preds_with = run_loyo(X_with, y, year_arr, with_label)

# ── 改善量まとめ ──────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  改善量サマリ（88件限定 LOYO / プール計算）')
print('='*60)
print(f'  {"モデル":<10} {"ベースRMSE":>12} {"追加後RMSE":>12} {"RMSE改善":>10} '
      f'{"ベースMAPE":>11} {"追加後MAPE":>11} {"MAPE改善":>10} '
      f'{"ベースR2":>10} {"追加後R2":>10}')
for name in ['Ridge', 'LightGBM']:
    rb   = sum_base[name]['RMSE']
    ra   = sum_with[name]['RMSE']
    mpb  = sum_base[name]['MAPE']
    mpa  = sum_with[name]['MAPE']
    r2b  = sum_base[name]['R2']
    r2a  = sum_with[name]['R2']
    print(f'  {name:<10} {rb:>12.3f} {ra:>12.3f} {rb-ra:>+10.3f} '
          f'{mpb:>10.2f}% {mpa:>10.2f}% {mpb-mpa:>+9.2f}% '
          f'{r2b:>10.4f} {r2a:>10.4f}')

print('\n  年別詳細（プール計算のため参考値）:')
for name in ['Ridge', 'LightGBM']:
    print(f'  [{name}]')
    base_by_yr = {d['year']: d for d in sum_base[name]['per_year']}
    with_by_yr = {d['year']: d for d in sum_with[name]['per_year']}
    for yr in sorted(base_by_yr.keys()):
        b = base_by_yr[yr]
        w = with_by_yr.get(yr, {})
        print(f'    {yr}年 (n={b["n_val"]:2d})  '
              f'BASE RMSE={b["RMSE"]:7.2f} MAPE={b["MAPE"]:6.2f}% R2={b["R2"]:6.4f}  '
              f'WITH RMSE={w.get("RMSE", float("nan")):7.2f} '
              f'MAPE={w.get("MAPE", float("nan")):6.2f}% '
              f'R2={w.get("R2", float("nan")):6.4f}  '
              f'RMSE改善={b["RMSE"]-w.get("RMSE", b["RMSE"]):+.2f}')

# ── 散布図（LightGBM のみ・年別色分け）─────────────────────────────────────────
YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60'}

fig, (ax_base, ax_with) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f8f9fa')
fig.suptitle('過去記録あり88件限定: LightGBM LOYO 予測値 vs 実測値\nベースライン vs 過去収量追加（色=テスト年）',
             fontsize=13, fontweight='bold', y=1.01)

configs = [
    (ax_base, preds_base, 'LightGBM', 'ベースライン（気象135のみ）'),
    (ax_with, preds_with, 'LightGBM', f'+ past情報({len(valid_past_cols)}列)'),
]

for ax, preds, model, label in configs:
    yt  = np.array(preds[model]['true'])
    yp  = np.array(preds[model]['pred'])
    yrs = np.array(preds[model]['year'])
    rmse = float(np.sqrt(((yt-yp)**2).mean()))
    mape = float((np.abs((yt-yp)/yt)).mean() * 100)
    ss_r = ((yt-yp)**2).sum(); ss_t = ((yt-yt.mean())**2).sum()
    r2   = float(1-ss_r/ss_t) if ss_t > 0 else float('nan')

    for yr, col in YEAR_COLORS.items():
        m = yrs == yr
        if m.any():
            ax.scatter(yt[m], yp[m], alpha=0.8, s=80, c=col,
                       edgecolors='white', linewidths=0.8, zorder=3,
                       label=f'{yr}年 (n={m.sum()})')

    mn = min(yt.min(), yp.min()) - 20
    mx = max(yt.max(), yp.max()) + 20
    ax.plot([mn,mx],[mn,mx],'--',color='#555555',lw=1.5,zorder=2)
    ax.set_xlim(mn,mx); ax.set_ylim(mn,mx)
    ax.set_xlabel('実測収量 (kg/10a)', fontsize=12)
    ax.set_ylabel('予測収量 (kg/10a)', fontsize=12)
    ax.set_title(f'LightGBM: {label}', fontsize=12, fontweight='bold')
    ax.text(0.04, 0.96,
            f'RMSE={rmse:.2f}\nMAPE={mape:.2f}%\nR2={r2:.4f}',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.87, edgecolor='#cccccc'))
    ax.legend(fontsize=10, loc='lower right', framealpha=0.85)
    ax.grid(True, alpha=0.25); ax.set_facecolor('#fdfdfd')
    ax.set_axisbelow(True)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'eval_past_yield_subset88_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  散布図 → {out_path}')
print('\n完了')
