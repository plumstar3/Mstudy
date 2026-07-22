"""
eval_yield_ratio.py
================================================================
[目的]
  農林水産省の市町村別大豆収量データ（soy-2014-2018-summary.csv）の
  前年（Y-1）の市町村平均収量をベースラインとして、
  目的変数を「比率」( Anomaly = Y_survey / Y_muni ) として
  LightGBM LOYO を実施する。

[入力特徴量]
  気象変数 9種 × 3 GDD期間 × mean = 27次元
  病害変数 5種 = 5次元
  合計 32次元

[バイアス補正（乗法的）]
  各LOYOフォールドで訓練データの y_ratio 平均（mu_ratio）で割って
  比率を1付近に中心化し、予測時に掛け戻すことで
  データセットと市町村平均の系統的ズレを補正する。

  学習: y_ratio_adj = y_ratio / mu_ratio  (平均が 1.0 に)
  予測: pred_ratio  = model(X) * mu_ratio
  復元: pred_abs    = pred_ratio * Y_muni

[出力]
  - コンソール: LOYO RMSE, MAPE, R2（絶対収量ベース・比率ベース両方）
  - 散布図: outputs/yield_pred_v3/eval_ratio_loyo.png
  - 除外圃場リスト: outputs/yield_pred_v3/excluded_fields_deviation.csv（共用）
"""

import sqlite3, os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
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
MAFF_CSV    = os.path.join('data', 'raw', 'soy-2014-2018-summary.csv')
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS   = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']
GDD_THRESHOLDS = [600, 1000]
RANDOM_STATE   = 42

# ── 1. 収量・位置データ ──────────────────────────────────────────────────────
print('=' * 60)
print('  大豆収量比率予測モデル（農水省前年平均ベースライン）')
print('=' * 60)

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
print(f'質問票データ: {len(quest_df)} 件')

# ── 2. 逆ジオコーディング結果（field_id → city）──────────────────────────────
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city']].drop_duplicates('field_id')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')
n_no_city = quest_df['city'].isna().sum()
print(f'市町村紐付け: {quest_df["city"].notna().sum()} 件成功 / {n_no_city} 件失敗')

# ── 3. 農林水産省データ読み込みとクリーニング ─────────────────────────────────
maff_raw = pd.read_csv(MAFF_CSV, encoding='utf-8-sig')
maff_raw.columns = ['year', 'city', 'maff_yield']
maff_raw['year']       = maff_raw['year'].astype(int)
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['city']       = maff_raw['city'].astype(str).str.strip()
maff_valid = maff_raw.dropna(subset=['maff_yield'])
print(f'\n農水省データ: {len(maff_raw)} 行 / 有効（数値）: {len(maff_valid)} 行')
print(f'  対象年度: {sorted(maff_valid["year"].unique())}')

# ── 4. 前年（Y-1）の市町村平均収量をマッチング ──────────────────────────────
maff_prev = maff_valid[['year', 'city', 'maff_yield']].copy()
maff_prev['join_year'] = maff_prev['year'] + 1
quest_df['city'] = quest_df['city'].astype(str).str.strip()
quest_df = quest_df.merge(
    maff_prev[['join_year', 'city', 'maff_yield']],
    left_on=['year', 'city'],
    right_on=['join_year', 'city'],
    how='left'
).drop(columns='join_year')

n_matched  = quest_df['maff_yield'].notna().sum()
n_unmatch  = quest_df['maff_yield'].isna().sum()
print(f'\n前年MAFF収量マッチング結果:')
print(f'  マッチ成功: {n_matched} 件')
print(f'  マッチ失敗（除外予定）: {n_unmatch} 件')

# ── 5. 除外圃場を記録して保存 ──────────────────────────────────────────────
excluded_df = quest_df[quest_df['maff_yield'].isna()].copy()
geo_cities  = set(geo_df['city'].dropna().astype(str).str.strip().tolist())
maff_cities = set(maff_valid['city'].astype(str).str.strip().tolist())
def get_reason(row):
    if pd.isna(row.get('city')) or row['city'] in ('nan', ''):
        return '市町村名なし（逆ジオコーディング失敗）'
    c = str(row['city']).strip()
    if c not in maff_cities:
        return f'MAFF未掲載市町村: {c}'
    return f'MAFF該当年度データ欠損（"-"）: city={c}, year={row["year"]}'
excluded_df['reason'] = excluded_df.apply(get_reason, axis=1)
excl_path = os.path.join(OUT_DIR, 'excluded_fields_deviation.csv')
excluded_df.to_csv(excl_path, index=False, encoding='utf-8-sig')
print(f'  除外リスト保存: {excl_path}')
for reason, cnt in excluded_df['reason'].value_counts().items():
    print(f'    [{cnt:3d}件] {reason}')

# ── 6. 有効サンプルで比率を計算 ───────────────────────────────────────────
valid_df = quest_df.dropna(subset=['maff_yield']).reset_index(drop=True)
valid_df['y_ratio'] = valid_df['yield'] / valid_df['maff_yield']   # ← 比率（÷）
print(f'\n有効サンプル: {len(valid_df)} 件')
print(f'年別内訳: {valid_df.groupby("year").size().to_dict()}')
print(f'y_ratio (比率 = Y_survey / Y_muni) 統計:')
print(f'  mean={valid_df["y_ratio"].mean():.3f}  '
      f'std={valid_df["y_ratio"].std():.3f}  '
      f'min={valid_df["y_ratio"].min():.3f}  '
      f'max={valid_df["y_ratio"].max():.3f}')
print(f'maff_yield (前年市町村平均) 統計:')
print(f'  mean={valid_df["maff_yield"].mean():.1f}  '
      f'std={valid_df["maff_yield"].std():.1f}')

# ── 7a. GDD期間別気象特徴量の生成（9変数×3期間×mean = 27次元）───────────────
fids  = sorted(valid_df['field_id'].unique().tolist())
years = sorted(valid_df['year'].unique().tolist())

print(f'\nGDD読み込み...', end=' ')
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
print('当年病害データ読み込み...', end=' ')
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
def make_lgb_pipe():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   lgb.LGBMRegressor(
            num_leaves       = 20,
            min_child_samples= 15,
            learning_rate    = 0.08,
            n_estimators     = 100,
            reg_lambda       = 10.0,
            reg_alpha        = 0.0,
            subsample        = 1.0,
            colsample_bytree = 0.6,
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            verbose          = -1)),
    ])

# ── 9. LOYO 実行 ──────────────────────────────────────────────────────────────
all_feat_cols = gdd_feat_cols + HARM_COLS  # 27次元気象 + 5次元病害 = 32次元
X        = all_data[all_feat_cols].to_numpy(dtype=np.float32)
y_ratio  = all_data['y_ratio'].to_numpy(dtype=np.float64)    # ← 比率
y_true   = all_data['yield'].to_numpy(dtype=np.float32)
y_maff   = all_data['maff_yield'].to_numpy(dtype=np.float32)
year_arr = all_data['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))

print(f'\n{"=" * 60}')
print(f'  LOYO 評価（比率予測モデル＋乗法的バイアス補正）')
print(f'  サンプル数: {len(all_data)} 件 / 特徴量: {X.shape[1]} 次元')
print(f'  テスト年: {test_years}')
print(f'  目的変数: y_ratio = Y_survey / Y_muni')
print(f'  バイアス補正: 各フォールドの y_ratio 平均で除算してから学習')
print(f'{"=" * 60}')

all_true_abs, all_pred_abs, all_years   = [], [], []
all_true_ratio, all_pred_ratio, all_maff = [], [], []
per_year_results = []

for test_year in test_years:
    tr = np.where(year_arr != test_year)[0]
    va = np.where(year_arr == test_year)[0]
    if len(tr) == 0 or len(va) == 0:
        continue

    pipe = make_lgb_pipe()
    # ── 乗法的バイアス補正 ────────────────────────────────────────────────────
    mu_ratio = float(y_ratio[tr].mean())             # 訓練データの系統バイアス比率
    y_ratio_adj = y_ratio[tr] / mu_ratio             # 平均が 1.0 になるよう正規化
    pipe.fit(X[tr], y_ratio_adj)
    pred_ratio_va = pipe.predict(X[va]) * mu_ratio   # 予測時にバイアスを掛け戻す
    # ─────────────────────────────────────────────────────────────────────────

    # 絶対収量に復元（予測比率 × 前年MAFF市町村平均）
    pred_abs   = pred_ratio_va * y_maff[va]          # ← 積で復元
    true_abs   = y_true[va]
    true_ratio = y_ratio[va]

    rmse_abs = float(np.sqrt(((true_abs - pred_abs) ** 2).mean()))
    mape_abs = float(np.abs((pred_abs - true_abs) / true_abs).mean() * 100)
    ss_r  = ((true_abs - pred_abs) ** 2).sum()
    ss_t  = ((true_abs - true_abs.mean()) ** 2).sum()
    r2    = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')
    rmse_ratio = float(np.sqrt(((true_ratio - pred_ratio_va) ** 2).mean()))

    print(f'  test={test_year}(n={len(va):2d}) train={len(tr):2d}  '
          f'AbsRMSE={rmse_abs:7.3f}  MAPE={mape_abs:6.2f}%  '
          f'R2={r2:6.4f}  RatioRMSE={rmse_ratio:.4f}')
    per_year_results.append({'year': test_year, 'n': len(va),
                             'AbsRMSE': rmse_abs, 'MAPE': mape_abs, 'R2': r2,
                             'RatioRMSE': rmse_ratio})

    all_true_abs.extend(true_abs.tolist())
    all_pred_abs.extend(pred_abs.tolist())
    all_years.extend([test_year] * len(va))
    all_true_ratio.extend(true_ratio.tolist())
    all_pred_ratio.extend(pred_ratio_va.tolist())
    all_maff.extend(y_maff[va].tolist())

# ── プール計算 ────────────────────────────────────────────────────────────────
yt = np.array(all_true_abs)
yp = np.array(all_pred_abs)
rmse_pool = float(np.sqrt(((yt - yp) ** 2).mean()))
mape_pool = float(np.abs((yp - yt) / yt).mean() * 100)
ss_r = ((yt - yp) ** 2).sum()
ss_t = ((yt - yt.mean()) ** 2).sum()
r2_pool = float(1 - ss_r / ss_t) if ss_t > 0 else float('nan')

yr = np.array(all_true_ratio)
ypr = np.array(all_pred_ratio)
rmse_ratio_pool = float(np.sqrt(((yr - ypr) ** 2).mean()))
ss_r2 = ((yr - ypr) ** 2).sum()
ss_t2 = ((yr - yr.mean()) ** 2).sum()
r2_ratio_pool = float(1 - ss_r2 / ss_t2) if ss_t2 > 0 else float('nan')

# ── ナイーブベース（比率=1 → pred_abs = Y_muni）─────────────────────────────
ym = np.array(all_maff)   # naive: pred_abs = Y_muni (ratio=1)
rmse_naive  = float(np.sqrt(((yt - ym) ** 2).mean()))
mape_naive  = float(np.abs((ym - yt) / yt).mean() * 100)
ss_r_n = ((yt - ym) ** 2).sum()
r2_naive = float(1 - ss_r_n / ss_t) if ss_t > 0 else float('nan')
# ratio R2 for naive (pred_ratio=1.0)
ss_r2_n = ((yr - 1.0) ** 2).sum()
r2_naive_ratio = float(1 - ss_r2_n / ss_t2) if ss_t2 > 0 else float('nan')

print(f'\n{"=" * 60}')
print('  プール計算 サマリ')
print(f'{"=" * 60}')
print(f'  {"条件":<27} {"RMSE":>8} {"MAPE":>8} {"R2(絶対)":>10} {"R2(比率)":>10}')
print(f'  {"-"*65}')
print(f'  {"ナイーブ（前年市町村平均）":<27} {rmse_naive:>8.3f} {mape_naive:>7.2f}%'
      f' {r2_naive:>10.4f} {r2_naive_ratio:>10.4f}')
print(f'  {"LightGBM（比率予測→復元）":<27} {rmse_pool:>8.3f} {mape_pool:>7.2f}%'
      f' {r2_pool:>10.4f} {r2_ratio_pool:>10.4f}')
print(f'  {"-"*65}')
rmse_imp = rmse_naive - rmse_pool
mape_imp = mape_naive - mape_pool
print(f'  {"モデルの改善量":<27} {rmse_imp:>+8.3f} {mape_imp:>+7.2f}%')

# ── 10. 散布図 ────────────────────────────────────────────────────────────────
YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60', 2015: '#8e44ad'}
yrs = np.array(all_years)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f8f9fa')
fig.suptitle('LightGBM LOYO: Ratio Prediction Model\n'
             '(Target = Y_survey / Y_muni_prev_year)',
             fontsize=13, fontweight='bold')

# 左: 絶対収量での評価
for yr_k, col in YEAR_COLORS.items():
    m = yrs == yr_k
    if m.any():
        ax1.scatter(yt[m], yp[m], alpha=0.8, s=80, c=col,
                    edgecolors='white', linewidths=0.8, zorder=3,
                    label=f'{yr_k} (n={m.sum()})')
mn = min(yt.min(), yp.min()) - 20
mx = max(yt.max(), yp.max()) + 20
ax1.plot([mn, mx], [mn, mx], '--', color='#555555', lw=1.5, zorder=2)
ax1.set_xlim(mn, mx); ax1.set_ylim(mn, mx)
ax1.set_xlabel('Observed yield (kg/10a)', fontsize=12)
ax1.set_ylabel('Predicted yield (kg/10a)', fontsize=12)
ax1.set_title('Absolute yield (ratio -> restored)', fontsize=11, fontweight='bold')
ax1.text(0.04, 0.96,
         f'RMSE={rmse_pool:.2f}\nMAPE={mape_pool:.2f}%\nR2={r2_pool:.4f}',
         transform=ax1.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.87, edgecolor='#cccccc'))
ax1.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax1.grid(True, alpha=0.25); ax1.set_facecolor('#fdfdfd'); ax1.set_axisbelow(True)

# 右: 比率での評価
for yr_c, col in YEAR_COLORS.items():
    m = yrs == yr_c
    if m.any():
        ax2.scatter(yr[m], ypr[m], alpha=0.8, s=80, c=col,
                    edgecolors='white', linewidths=0.8, zorder=3,
                    label=f'{yr_c} (n={m.sum()})')
mn2 = min(yr.min(), ypr.min()) - 0.1
mx2 = max(yr.max(), ypr.max()) + 0.1
ax2.plot([mn2, mx2], [mn2, mx2], '--', color='#555555', lw=1.5, zorder=2)
ax2.axhline(1.0, color='#888888', lw=0.8, ls=':', alpha=0.7)
ax2.axvline(1.0, color='#888888', lw=0.8, ls=':', alpha=0.7)
ax2.set_xlim(mn2, mx2); ax2.set_ylim(mn2, mx2)
ax2.set_xlabel('Observed ratio (Y_survey / Y_muni)', fontsize=12)
ax2.set_ylabel('Predicted ratio', fontsize=12)
ax2.set_title('Ratio space evaluation', fontsize=11, fontweight='bold')
ax2.text(0.04, 0.96,
         f'RMSE={rmse_ratio_pool:.4f}\nR2={r2_ratio_pool:.4f}',
         transform=ax2.transAxes, fontsize=11, va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.87, edgecolor='#cccccc'))
ax2.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax2.grid(True, alpha=0.25); ax2.set_facecolor('#fdfdfd'); ax2.set_axisbelow(True)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'eval_ratio_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n  散布図 -> {out_path}')
print('\n完了')
