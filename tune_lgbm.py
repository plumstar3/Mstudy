"""
tune_lgbm.py
================================================================
[目的]
  LOYO (Leave-One-Year-Out) のプール MAPE を目的関数として
  LightGBM のハイパーパラメータをランダムサーチで最適化する。

  小サンプル（~70-90件学習）に適した正則化範囲を探索し、
  最良設定を eval_past_yield_subset.py に反映できるよう出力する。

[対象データ]
  eval_past_yield_subset.py と同じ特徴量（気象39 + 過去情報45 = 84次元）
  サブセット（past_yield_n > 0 の 124件）のみ使用。
"""

import sqlite3, os, warnings, time, random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ── 設定（eval_past_yield_subset.py と同一）─────────────────────────────────
FIELD_DB       = os.path.join('data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB     = os.path.join('data', 'processed', 'weather_database_fieldid.db')
GDD_CSV        = os.path.join('outputs', 'gdd', 'gdd_daily.csv')
PAST_YIELD_CSV = os.path.join('outputs', 'data_analysis', 'past_yield_features_v2.csv')
OUT_DIR        = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_COLS     = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCPRA', 'SSD', 'GSR', 'WIND', 'SWE', 'RH']
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
HARM_COLS      = ['sick', 'wet', 'typhoon', 'unripen', 'weed']
RANDOM_STATE   = 42

# ランダムサーチの試行回数
N_TRIALS = 200

# ── 探索パラメータ空間（小サンプル向け）────────────────────────────────────
PARAM_SPACE = {
    'num_leaves':        [4, 7, 11, 15, 20, 31],
    'min_child_samples': [3, 5, 8, 10, 15, 20],
    'learning_rate':     [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    'n_estimators':      [100, 150, 200, 300, 500],
    'reg_lambda':        [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
    'reg_alpha':         [0.0, 0.1, 0.5, 1.0, 2.0],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.5, 0.6, 0.7, 0.8, 1.0],
}

# ── データロード（eval と同じ流れ）──────────────────────────────────────────
print('='*60)
print('  LightGBM ハイパーパラメータ チューニング (LOYO-MAPE)')
print('='*60)

# 収量・位置データ
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

# 過去特徴量の列名定義
PAST_WX_COLS   = [f'past_{c}' for c in gdd_feat_cols]
PAST_HARM_COLS = [f'past_harm_{c}' for c in HARM_COLS]
PAST_FEAT_COLS = ['past_yield_mean'] + PAST_WX_COLS + PAST_HARM_COLS

# データ結合
all_data = quest_df.merge(feat_df, on=['field_id','year'], how='inner')
past_load_cols = ['field_id', 'year', 'has_past_record'] + PAST_FEAT_COLS
avail_cols = [c for c in past_load_cols if c in past_df.columns]
all_data = all_data.merge(past_df[avail_cols], on=['field_id','year'], how='left')
all_data['has_past_record'] = all_data['has_past_record'].fillna(0).astype(int)

subset = all_data[all_data['has_past_record'] == 1].reset_index(drop=True)
valid_past_cols = [c for c in PAST_FEAT_COLS if c in subset.columns]
print(f'サブセット: {len(subset)} 件 / 特徴量: 気象{len(gdd_feat_cols)} + 過去{len(valid_past_cols)} = {len(gdd_feat_cols)+len(valid_past_cols)}次元')

# 特徴量行列
X = subset[gdd_feat_cols + valid_past_cols].to_numpy(dtype=np.float32)
y = subset['yield'].to_numpy(dtype=np.float32)
year_arr = subset['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))

# ── LOYO 評価関数 ─────────────────────────────────────────────────────────────
def make_lgb_pipe(params):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler',  StandardScaler()),
        ('model',   lgb.LGBMRegressor(**params, random_state=RANDOM_STATE,
                                       n_jobs=1, verbose=-1)),
    ])

def loyo_score(params):
    """LOYO プール MAPE を返す（小さいほど良い）。"""
    all_true, all_pred = [], []
    for test_year in test_years:
        tr = np.where(year_arr != test_year)[0]
        va = np.where(year_arr == test_year)[0]
        if len(tr) == 0 or len(va) == 0:
            continue
        pipe = make_lgb_pipe(params)
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[va])
        all_true.extend(y[va].tolist())
        all_pred.extend(pred.tolist())
    yt = np.array(all_true)
    yp = np.array(all_pred)
    mape = float(np.abs((yp - yt) / yt).mean() * 100)
    rmse = float(np.sqrt(((yt - yp)**2).mean()))
    return mape, rmse

# ── ランダムサーチ ────────────────────────────────────────────────────────────
rng = random.Random(RANDOM_STATE)
print(f'\nランダムサーチ開始 (N_TRIALS={N_TRIALS})...')
t0 = time.time()

results = []
for trial in range(N_TRIALS):
    params = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
    try:
        mape, rmse = loyo_score(params)
        results.append({'mape': mape, 'rmse': rmse, 'params': params})
    except Exception as e:
        pass  # 一部の組み合わせで失敗する場合はスキップ

    if (trial + 1) % 50 == 0:
        best_so_far = min(results, key=lambda x: x['mape'])
        elapsed = time.time() - t0
        print(f'  [{trial+1}/{N_TRIALS}]  経過: {elapsed:.1f}s  '
              f'最良MAPE: {best_so_far["mape"]:.2f}%  '
              f'最良RMSE: {best_so_far["rmse"]:.2f}')

elapsed = time.time() - t0
print(f'\n完了 ({elapsed:.1f}s)')

# ── Top-5 表示 ────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['mape'])
print('\n' + '='*60)
print('  Top-5 ハイパーパラメータ（LOYO プール MAPE 昇順）')
print('='*60)
for rank, res in enumerate(results[:5], 1):
    p = res['params']
    print(f'\n  [{rank}位] MAPE={res["mape"]:.2f}%  RMSE={res["rmse"]:.2f}')
    print(f'       num_leaves={p["num_leaves"]}  min_child_samples={p["min_child_samples"]}')
    print(f'       lr={p["learning_rate"]}  n_est={p["n_estimators"]}')
    print(f'       reg_λ={p["reg_lambda"]}  reg_α={p["reg_alpha"]}')
    print(f'       subsample={p["subsample"]}  colsample={p["colsample_bytree"]}')

# ── ベスト設定をそのまま貼れる形で出力 ────────────────────────────────────────
best = results[0]
bp = best['params']
print('\n' + '='*60)
print('  ▼ ベスト設定（eval_past_yield_subset.py への貼り付け用）')
print('='*60)
print(f"""
lgb.LGBMRegressor(
    num_leaves       = {bp['num_leaves']},
    min_child_samples= {bp['min_child_samples']},
    learning_rate    = {bp['learning_rate']},
    n_estimators     = {bp['n_estimators']},
    reg_lambda       = {bp['reg_lambda']},
    reg_alpha        = {bp['reg_alpha']},
    subsample        = {bp['subsample']},
    colsample_bytree = {bp['colsample_bytree']},
    random_state     = {RANDOM_STATE},
    n_jobs           = -1,
    verbose          = -1,
)""")

# ── CSV 保存 ──────────────────────────────────────────────────────────────────
res_df = pd.DataFrame([
    {'rank': i+1, 'mape': r['mape'], 'rmse': r['rmse'], **r['params']}
    for i, r in enumerate(results[:20])
])
res_path = os.path.join(OUT_DIR, 'lgbm_tune_results.csv')
res_df.to_csv(res_path, index=False, encoding='utf-8-sig')
print(f'\n  上位20件を保存: {res_path}')
