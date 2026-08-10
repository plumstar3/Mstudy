"""
analyze_quantile_mapping.py
================================================================
[目的]
  分位点マッピング（Quantile Mapping: QM）の効果を可視化・分析する。
  ※ このスクリプトは「全データを使ってQM変換後の分布を見る」ための解析専用。
  ※ モデル評価（eval_yield_deviation.py）では LOYO ループ内で
     データリーク防止のため fold ごとに QM を構築している点に注意。

[出力]
  outputs/yield_pred_v5/qm_analysis/
    qm_cdf_comparison.png  : 変換前後のCDF比較
    qm_scatter.png         : 変換前後の散布図比較
    qm_stats.csv           : 変換前後の統計量
"""

import sqlite3, os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── フォント設定 ───────────────────────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定 ──────────────────────────────────────────────────────────────────────
FIELD_DB    = os.path.join('data', 'processed', 'FieldData_fieldid.db')
MAFF_CSV    = os.path.join('data', 'raw', 'summary-soy-2010-2018.csv')
GEOCODE_CSV = os.path.join('outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join('outputs', 'yield_pred_v5', 'qm_analysis')
os.makedirs(OUT_DIR, exist_ok=True)
MAFF_WINDOW = 1
N_QUANTILES = 100

print('=' * 60)
print('  分位点マッピング（QM）解析スクリプト')
print('  ※ 全データ一括でのQM変換（解析・可視化専用）')
print('=' * 60)

# ── 1. データ読み込み ─────────────────────────────────────────────────────────
conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, breed
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018''', conn)
conn.close()
quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)
quest_df['yield']    = pd.to_numeric(quest_df['yield'], errors='coerce')
quest_df['breed']    = quest_df['breed'].fillna('Unknown').astype(str).str.strip()
quest_df = quest_df.dropna(subset=['yield'])

geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df = geo_df[['field_id', 'city', 'muniCd']].drop_duplicates('field_id')
geo_df['muniCd'] = pd.to_numeric(geo_df['muniCd'], errors='coerce').astype('Int64')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')

maff_raw = pd.read_csv(MAFF_CSV, encoding='cp932')
maff_raw.columns = ['year', 'city', 'maff_yield', 'muniCd']
maff_raw['year']       = maff_raw['year'].astype(int)
maff_raw['maff_yield'] = pd.to_numeric(maff_raw['maff_yield'], errors='coerce')
maff_raw['muniCd']     = pd.to_numeric(maff_raw['muniCd'], errors='coerce').astype('Int64')
maff_valid = maff_raw.dropna(subset=['maff_yield'])

# ── 2. 前年MAFFをマッチング（MAFF_WINDOW=1）─────────────────────────────────
TARGET_YEARS = [2015, 2016, 2017, 2018]
quest_df['muniCd'] = pd.to_numeric(quest_df['muniCd'], errors='coerce').astype('Int64')

maff_rows = []
for target_year in TARGET_YEARS:
    hist_years = list(range(target_year - MAFF_WINDOW, target_year))
    hist = maff_valid[maff_valid['year'].isin(hist_years)]
    muni_avg = hist.groupby('muniCd')['maff_yield'].mean().reset_index()
    muni_avg['join_year'] = target_year
    maff_rows.append(muni_avg)
maff_prev_df = pd.concat(maff_rows, ignore_index=True)
maff_prev_df['muniCd'] = maff_prev_df['muniCd'].astype('Int64')

merged = quest_df.merge(
    maff_prev_df[['join_year', 'muniCd', 'maff_yield']],
    left_on=['year', 'muniCd'],
    right_on=['join_year', 'muniCd'],
    how='left'
).drop(columns='join_year')
merged = merged.dropna(subset=['maff_yield']).reset_index(drop=True)
print(f'有効サンプル: {len(merged)} 件')

y_maff = merged['maff_yield'].to_numpy(dtype=np.float32)
y_true = merged['yield'].to_numpy(dtype=np.float32)

# ── 3. QM変換関数の構築（全データ使用 ※解析専用） ───────────────────────────
print('\n■ QM変換関数の構築（全データ一括 / 解析専用）...')
quantiles = np.linspace(0, 100, N_QUANTILES)
maff_q    = np.percentile(y_maff, quantiles)
true_q    = np.percentile(y_true, quantiles)
_, idx    = np.unique(maff_q, return_index=True)
qm_func   = interp1d(maff_q[idx], true_q[idx], kind='linear', fill_value='extrapolate')

y_maff_mapped = qm_func(y_maff).astype(np.float32)

# ── 4. 変換前後の統計量 ───────────────────────────────────────────────────────
stats = pd.DataFrame({
    '指標': ['サンプル数', '平均 (mean)', '標準偏差 (std)', '最小値 (min)',
             '25%タイル', '50%タイル (中央値)', '75%タイル', '最大値 (max)'],
    '現場収量 (y_true)': [
        len(y_true), np.mean(y_true), np.std(y_true),
        np.min(y_true), np.percentile(y_true, 25), np.median(y_true),
        np.percentile(y_true, 75), np.max(y_true)
    ],
    'MAFF（変換前）': [
        len(y_maff), np.mean(y_maff), np.std(y_maff),
        np.min(y_maff), np.percentile(y_maff, 25), np.median(y_maff),
        np.percentile(y_maff, 75), np.max(y_maff)
    ],
    'MAFF（QM変換後）': [
        len(y_maff_mapped), np.mean(y_maff_mapped), np.std(y_maff_mapped),
        np.min(y_maff_mapped), np.percentile(y_maff_mapped, 25), np.median(y_maff_mapped),
        np.percentile(y_maff_mapped, 75), np.max(y_maff_mapped)
    ],
})
print('\n■ 変換前後の統計量比較:')
pd.set_option('display.float_format', '{:.1f}'.format)
print(stats.to_string(index=False))
stats.to_csv(os.path.join(OUT_DIR, 'qm_stats.csv'), index=False, encoding='utf-8-sig')

# RMSE 変化も確認
rmse_before = float(np.sqrt(((y_true - y_maff) ** 2).mean()))
rmse_after  = float(np.sqrt(((y_true - y_maff_mapped) ** 2).mean()))
print(f'\n■ ナイーブベースライン RMSE:')
print(f'  QM変換前: {rmse_before:.3f} kg/10a')
print(f'  QM変換後: {rmse_after:.3f} kg/10a  (改善: {rmse_before - rmse_after:+.3f})')

# ── 5. 可視化 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f8f9fa')
fig.suptitle('分位点マッピング（QM）解析\n（全データを使った一括変換 / 解析専用）',
             fontsize=13, fontweight='bold')

# 左: PDF比較
ax = axes[0]
sns.kdeplot(y_true,        ax=ax, label='現場収量 (Questionnaire)', fill=True, alpha=0.4, color='#3498db')
sns.kdeplot(y_maff,        ax=ax, label='MAFF（変換前）',           fill=True, alpha=0.4, color='#e74c3c')
sns.kdeplot(y_maff_mapped, ax=ax, label='MAFF（QM変換後）',         fill=True, alpha=0.4, color='#2ecc71')
ax.set_title('PDF（確率密度関数）比較', fontweight='bold')
ax.set_xlabel('収量 (kg/10a)')
ax.legend(fontsize=9)
ax.set_facecolor('#fdfdfd')
ax.grid(True, alpha=0.25)

# 中: CDF比較
ax = axes[1]
sns.ecdfplot(y_true,        ax=ax, label='現場収量 (Questionnaire)', color='#3498db', lw=2)
sns.ecdfplot(y_maff,        ax=ax, label='MAFF（変換前）',           color='#e74c3c', lw=2, linestyle='--')
sns.ecdfplot(y_maff_mapped, ax=ax, label='MAFF（QM変換後）',         color='#2ecc71', lw=2, linestyle='-.')
ax.set_title('CDF（累積分布関数）比較', fontweight='bold')
ax.set_xlabel('収量 (kg/10a)')
ax.set_ylabel('累積割合')
ax.legend(fontsize=9)
ax.set_facecolor('#fdfdfd')
ax.grid(True, alpha=0.25)

# 右: 変換前後の散布図（y_maff vs y_maff_mapped）
ax = axes[2]
ax.scatter(y_maff, y_maff_mapped, alpha=0.4, s=30, color='#9b59b6', edgecolors='none')
mn = min(y_maff.min(), y_maff_mapped.min()) - 10
mx = max(y_maff.max(), y_maff_mapped.max()) + 10
ax.plot([mn, mx], [mn, mx], '--', color='#555555', lw=1.5, label='y=x（変換なし）')
ax.set_title('QM変換マッピング\n（変換前MAFF → 変換後MAFF）', fontweight='bold')
ax.set_xlabel('MAFF収量（変換前）')
ax.set_ylabel('MAFF収量（QM変換後）')
ax.legend(fontsize=9)
ax.set_facecolor('#fdfdfd')
ax.grid(True, alpha=0.25)
ax.text(0.04, 0.96,
        f'RMSE改善\n{rmse_before:.1f} → {rmse_after:.1f}\n（Δ{rmse_before - rmse_after:+.1f}）',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.87, edgecolor='#cccccc'))

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'qm_cdf_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\n■ CDF比較グラフ保存: {out_path}')

# ── 6. 品種ごとのQM補正効果 ───────────────────────────────────────────────────
merged['y_diff_before'] = y_true - y_maff
merged['y_diff_after']  = y_true - y_maff_mapped
top_breeds = merged['breed'].value_counts().head(10).index
breed_stats = (merged[merged['breed'].isin(top_breeds)]
               .groupby('breed')[['y_diff_before', 'y_diff_after']]
               .agg(['mean', 'std'])
               .round(1))
print('\n■ 主要品種ごとのQM補正効果（偏差の平均値）:')
print(f'  {"品種":<15} {"補正前の偏差平均":>16} {"補正後の偏差平均":>16}')
print(f'  {"-"*50}')
for breed in top_breeds:
    before_mean = merged[merged['breed'] == breed]['y_diff_before'].mean()
    after_mean  = merged[merged['breed'] == breed]['y_diff_after'].mean()
    print(f'  {breed:<15} {before_mean:>16.1f} {after_mean:>16.1f}')

print('\n完了')
