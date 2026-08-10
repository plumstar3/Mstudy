"""
害データの収量偏差への寄与分析スクリプト
- スピアマン順位相関 / t検定（p値）
- LightGBM Feature Importance (Gain)
- Permutation Importance
"""
import os
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

os.chdir(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy')

# ─── 設定 ───────────────────────────────────────────────────────────────────
FIELD_DB   = os.path.join('data', 'processed', 'FieldData_fieldid.db')
MAFF_CSV   = os.path.join('data', 'raw', 'summary-soy-2010-2018.csv')
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR    = os.path.join('outputs', 'yield_pred_v5', 'harm_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

HARM_SCORE_COLS = ['sick', 'wet', 'unripen', 'weed', 'bug', 'lay', 'loss']
HARM_FLAG_COLS  = ['typhoon', 'long_rain', 'heavy_rain', 'drought', 'gale', 'few_solar']
ALL_HARM_COLS   = HARM_SCORE_COLS + HARM_FLAG_COLS

# ─── 1. 収量データ ────────────────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year,
           CAST(yield AS REAL) AS yield,
           breed
    FROM Questionaire
    WHERE year BETWEEN 2015 AND 2018
      AND yield IS NOT NULL
      AND yield > 0''', conn)
conn.close()

# ─── 2. 位置・市町村コード ─────────────────────────────────────────────────────
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')[['field_id', 'muniCd']].drop_duplicates('field_id')
geo_df['muniCd'] = pd.to_numeric(geo_df['muniCd'], errors='coerce')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')

# ─── 3. MAFF前年データ（1年移動平均） ─────────────────────────────────────────
maff_raw = pd.read_csv(MAFF_CSV, encoding='cp932')
maff_raw.columns = ['year', 'city', 'maff_yield', 'muniCd']
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['muniCd']     = pd.to_numeric(maff_raw['muniCd'],     errors='coerce')
maff_raw = maff_raw.dropna(subset=['maff_yield', 'muniCd'])

rows = []
for ty in [2015, 2016, 2017, 2018]:
    hist = maff_raw[maff_raw['year'].between(ty - 1, ty - 1)]
    hist = hist.groupby('muniCd')['maff_yield'].mean().reset_index()
    hist['year'] = ty
    rows.append(hist)
maff_prev = pd.concat(rows).rename(columns={'maff_yield': 'maff_prev'})

valid_df = quest_df.merge(maff_prev, on=['year', 'muniCd'], how='inner')
valid_df['y_diff'] = valid_df['yield'] - valid_df['maff_prev']
print(f'有効サンプル: {len(valid_df)} 件')

# ─── 4. 害データ読み込み（不明→NaN補正済み） ────────────────────────────────
conn_h = sqlite3.connect(FIELD_DB)
harm_col_str = ', '.join(ALL_HARM_COLS)
harm_df = pd.read_sql(f'''
    SELECT field_id, year, {harm_col_str} FROM harm
    WHERE field_id IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn_h)
conn_h.close()
harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)

# スコア列：数値変換
for c in HARM_SCORE_COLS:
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce')

# 不明→NaN補正
harm_df['sick'] = harm_df['sick'].where(harm_df['sick'] != 3, other=np.nan)
harm_df.loc[harm_df['year'] == 2015, 'bug'] = harm_df.loc[harm_df['year'] == 2015, 'bug'].where(
    harm_df.loc[harm_df['year'] == 2015, 'bug'] != 3, other=np.nan)
harm_df.loc[harm_df['year'] >= 2016, 'bug'] = harm_df.loc[harm_df['year'] >= 2016, 'bug'].where(
    harm_df.loc[harm_df['year'] >= 2016, 'bug'] != 4, other=np.nan)

# フラグ列：TRUE/FALSE → 1/0
for c in HARM_FLAG_COLS:
    harm_df[c] = harm_df[c].replace({'TRUE': 1, 'FALSE': 0, 'true': 1, 'false': 0,
                                     True: 1, False: 0})
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce').fillna(0).astype(int)

agg_dict = {c: 'max' for c in HARM_FLAG_COLS}
agg_dict.update({c: 'mean' for c in HARM_SCORE_COLS})
harm_df = harm_df.groupby(['field_id', 'year'])[ALL_HARM_COLS].agg(agg_dict).reset_index()

# 結合
df = valid_df.merge(harm_df, on=['field_id', 'year'], how='left')
for c in HARM_FLAG_COLS:
    df[c] = df[c].fillna(0)
print(f'害データ結合後: {len(df)} 件')

# ─── 5. 統計的寄与分析 ────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  収量偏差（y_diff）への害データ寄与分析')
print('='*65)

results = []

# スコア変数：スピアマン順位相関
print('\n--- スコア変数（スピアマン順位相関） ---')
print(f'{"変数":<12} {"相関係数":>8} {"p値":>10} {"有効n":>6} {"判定":>8}')
print('-'*50)
for c in HARM_SCORE_COLS:
    sub = df[['y_diff', c]].dropna()
    if len(sub) < 10:
        continue
    rho, pval = stats.spearmanr(sub['y_diff'], sub[c])
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
    print(f'{c:<12} {rho:>+8.4f} {pval:>10.4f} {len(sub):>6} {sig:>8}')
    results.append({'col': c, 'type': 'score', 'stat': rho, 'pval': pval,
                    'abs_stat': abs(rho), 'n': len(sub)})

# フラグ変数：t検定（Welch）
print('\n--- フラグ変数（Welch t検定：被害あり vs なし） ---')
print(f'{"変数":<14} {"被害なし平均":>10} {"被害あり平均":>10} {"差":>8} {"p値":>10} {"判定":>8}')
print('-'*65)
for c in HARM_FLAG_COLS:
    g0 = df.loc[df[c] == 0, 'y_diff'].dropna()
    g1 = df.loc[df[c] == 1, 'y_diff'].dropna()
    if len(g0) < 5 or len(g1) < 5:
        print(f'{c:<14} サンプル不足')
        continue
    t_stat, pval = stats.ttest_ind(g0, g1, equal_var=False)
    diff = g1.mean() - g0.mean()
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
    print(f'{c:<14} {g0.mean():>+10.1f} {g1.mean():>+10.1f} {diff:>+8.1f} {pval:>10.4f} {sig:>8}')
    results.append({'col': c, 'type': 'flag', 'stat': abs(diff), 'pval': pval,
                    'abs_stat': abs(diff), 'n0': len(g0), 'n1': len(g1)})

# ─── 6. LightGBM Feature Importance ─────────────────────────────────────────
print('\n' + '='*65)
print('  LightGBM Feature Importance (全害変数のみ)')
print('='*65)

X_harm = df[ALL_HARM_COLS].copy()
# スコアのNaNをmedianで補完
for c in HARM_SCORE_COLS:
    X_harm[c] = X_harm[c].fillna(X_harm[c].median())
y = df['y_diff'].values

model = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                           random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_harm, y)

fi_gain  = pd.Series(model.booster_.feature_importance(importance_type='gain'),
                     index=ALL_HARM_COLS).sort_values(ascending=False)
fi_split = pd.Series(model.booster_.feature_importance(importance_type='split'),
                     index=ALL_HARM_COLS).sort_values(ascending=False)

print('\n--- Gain（情報利得）ランキング ---')
for rank, (feat, val) in enumerate(fi_gain.items(), 1):
    print(f'  {rank:2d}. {feat:<14} {val:>10.2f}')

# ─── 7. Permutation Importance ────────────────────────────────────────────────
print('\n' + '='*65)
print('  Permutation Importance（LOYO全データ訓練モデルで計算）')
print('='*65)
perm = permutation_importance(model, X_harm, y, n_repeats=30,
                               random_state=42, scoring='neg_root_mean_squared_error')
perm_df = pd.DataFrame({
    'feature': ALL_HARM_COLS,
    'mean_decrease': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('mean_decrease', ascending=False)

print('\n--- RMSE増加量（シャッフルで精度がどれだけ落ちるか） ---')
for _, row in perm_df.iterrows():
    bar = '#' * max(0, int(row['mean_decrease'] * 10))
    print(f'  {row["feature"]:<14} {row["mean_decrease"]:>+8.4f} +/- {row["std"]:.4f}  {bar}')

# ─── 8. 総合ランキング ────────────────────────────────────────────────────────
print('\n' + '='*65)
print('  総合推奨ランキング')
print('='*65)

# 各スコアをmin-max正規化して統合
fi_norm   = (fi_gain - fi_gain.min()) / (fi_gain.max() - fi_gain.min() + 1e-9)
perm_norm = (perm_df.set_index('feature')['mean_decrease'] -
             perm_df['mean_decrease'].min()) / (perm_df['mean_decrease'].max() -
             perm_df['mean_decrease'].min() + 1e-9)
stat_scores = {r['col']: (1 - r['pval']) * r['abs_stat'] for r in results}
s_max = max(stat_scores.values()) if stat_scores else 1
stat_norm = pd.Series({k: v / s_max for k, v in stat_scores.items()})

all_feats = ALL_HARM_COLS
total_score = {}
for f in all_feats:
    s = (fi_norm.get(f, 0) * 0.4 +
         perm_norm.get(f, 0) * 0.4 +
         stat_norm.get(f, 0) * 0.2)
    total_score[f] = s

total_series = pd.Series(total_score).sort_values(ascending=False)
print(f'\n{"ランク":<5} {"変数":<14} {"総合スコア":>10} {"推奨"}')
print('-'*45)
for rank, (feat, score) in enumerate(total_series.items(), 1):
    recommend = '[採用推奨]' if score >= 0.2 else ('[要検討]' if score >= 0.1 else '[削除推奨]')
    print(f'  {rank:2d}.  {feat:<14} {score:>10.4f}  {recommend}')

# ─── 9. 可視化 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('害データの収量偏差への寄与分析', fontsize=14, fontweight='bold')

# 左: Feature Importance (Gain)
ax = axes[0]
colors = ['#e74c3c' if v > fi_gain.median() else '#95a5a6' for v in fi_gain.values]
ax.barh(fi_gain.index[::-1], fi_gain.values[::-1], color=colors[::-1])
ax.set_xlabel('Feature Importance (Gain)')
ax.set_title('LightGBM Gain')
ax.axvline(fi_gain.median(), color='gray', linestyle='--', alpha=0.7)

# 中央: Permutation Importance
ax = axes[1]
perm_plot = perm_df.sort_values('mean_decrease', ascending=True)
colors2 = ['#e74c3c' if v > 0 else '#3498db' for v in perm_plot['mean_decrease']]
ax.barh(perm_plot['feature'], perm_plot['mean_decrease'],
        xerr=perm_plot['std'], color=colors2, capsize=3)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('RMSE増加量（大きいほど重要）')
ax.set_title('Permutation Importance')

# 右: 総合スコア
ax = axes[2]
total_sorted = total_series.sort_values(ascending=True)
colors3 = ['#e74c3c' if v >= 0.2 else ('#f39c12' if v >= 0.1 else '#95a5a6')
           for v in total_sorted.values]
ax.barh(total_sorted.index, total_sorted.values, color=colors3)
ax.axvline(0.2, color='red', linestyle='--', alpha=0.7, label='採用推奨閾値')
ax.axvline(0.1, color='orange', linestyle='--', alpha=0.7, label='要検討閾値')
ax.set_xlabel('総合スコア')
ax.set_title('総合ランキング')
ax.legend(fontsize=8)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'harm_feature_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n可視化を保存: {save_path}')
print('\n分析完了！')
