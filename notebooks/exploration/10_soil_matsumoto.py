"""
10_soil_matsumoto.py
=====================
松本市内の圃場に絞った土壌特徴量 High / Low 収量群間比較。

[グループ定義 - 08_yield_intra_analysis.py と同じ方式]
  各年の松本市圃場の中で収量の上位33% → High、下位33% → Low
  全年のHigh/Lowをプールして統計検定を行う。

[採取タイミング]
  松本市には収穫前後（10〜12月）のみ土壌データが存在する。
  播種前後（5〜7月）のデータなし → 収穫前後のみ分析。

[注意: 検出力について]
  High=8件 / Low=9件 という非常に少ない標本数のため、
  Mann-Whitney U 検定の検出力が低い（効果量d≈1.3以上でないと検出できない）。
  「有意差なし」は「差がない」の証明ではなく「差を検出できなかった」と解釈すること。
  全データ点を可視化して、傾向の方向性を目視で確認することを推奨。

[出力]
  outputs/soil_analysis/
    10_matsumoto_soil_postharv.png   (ドットプロット + 中央値線)
    10_matsumoto_soil_stats.csv      (統計サマリ)
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

for _fn in ['IPAexGothic', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# =============================================================================
# 定数・パス
# =============================================================================
_ROOT         = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB      = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEOCODE_CSV   = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR       = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

MATSUMOTO_MUNI = 20202
TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33

POST_HARV_MONTHS = [10, 11, 12]
EXCLUDE_MONTHS   = [8, 9]

SOIL_COLS = ['pH', 'CaO', 'MgO', 'K2O', 'P2O2',
             'N_able', 'N_per', 'C_per', 'CEC',
             'H3PO4', 'B_able', 'Mo_able', 'Mn_c']

SOIL_LABELS = {
    'pH':     'pH',
    'CaO':    '石灰 CaO\n(mg/100g)',
    'MgO':    '苦土 MgO\n(mg/100g)',
    'K2O':    '加里 K2O\n(mg/100g)',
    'P2O2':   'リン酸 P2O2\n(mg/100g)',
    'N_able': '有効態窒素\n(mg/100g)',
    'N_per':  '全窒素 N\n(%)',
    'C_per':  '全炭素 C\n(%)',
    'CEC':    'CEC\n(meq/100g)',
    'H3PO4':  'リン酸吸収\n(mg/100g)',
    'B_able': '有効態B\n(mg/kg)',
    'Mo_able':'有効態Mo\n(μg/kg)',
    'Mn_c':   '交換性Mn\n(mg/kg)',
}

MIN_PER_GROUP = 3   # 最低サンプル数（これ未満はn.a.）
COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'

# =============================================================================
# 1. 松本市 field_id 取得
# =============================================================================
geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce')
mat_ids = set(geo[geo['muniCd'] == MATSUMOTO_MUNI]['field_id'].unique())
print(f'松本市 field_id 数: {len(mat_ids)}')

# =============================================================================
# 2. 収量データ読み込み & High/Low 定義（年ごと松本市内）
# =============================================================================
conn = sqlite3.connect(FIELD_DB)
quest = pd.read_sql("""
    SELECT field_id, year, yield
    FROM Questionaire
    WHERE yield IS NOT NULL AND year BETWEEN 2015 AND 2018
""", conn)
quest = quest.dropna(subset=['field_id'])
quest['field_id'] = quest['field_id'].astype(int)
quest['year']     = quest['year'].astype(int)
quest = quest[quest['field_id'].isin(mat_ids)].copy()

quest['group'] = 'Mid'
for yr in TARGET_YEARS:
    mask = quest['year'] == yr
    sub  = quest[mask]
    if len(sub) < 6:
        continue
    lo = sub['yield'].quantile(GROUP_QUANTILE)
    hi = sub['yield'].quantile(1 - GROUP_QUANTILE)
    quest.loc[mask & (quest['yield'] <= lo), 'group'] = 'Low'
    quest.loc[mask & (quest['yield'] >= hi), 'group'] = 'High'

print('\n収量グループ（年別）:')
print(quest.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())

# =============================================================================
# 3. 土壌データ読み込み
# =============================================================================
soil_cols_str = ', '.join(SOIL_COLS)
solid = pd.read_sql(f"""
    SELECT field_id, year, place, date, {soil_cols_str}
    FROM Solid
    WHERE field_id IS NOT NULL AND date IS NOT NULL
""", conn)
conn.close()

solid = solid.dropna(subset=['field_id'])
solid['field_id'] = solid['field_id'].astype(int)
solid['year']     = solid['year'].astype(int)
solid = solid[solid['field_id'].isin(mat_ids)].copy()
solid['date']  = pd.to_datetime(solid['date'])
solid['month'] = solid['date'].dt.month

for c in SOIL_COLS:
    solid[c] = pd.to_numeric(solid[c], errors='coerce')

# 採取タイミングでフィルタ（収穫前後のみ）
solid_ph = solid[solid['month'].isin(POST_HARV_MONTHS)].copy()
print(f'\n松本市 収穫前後土壌レコード数: {len(solid_ph)} 件')

# =============================================================================
# 4. 収量グループと結合
# =============================================================================
df = solid_ph.merge(
    quest[['field_id', 'year', 'yield', 'group']],
    on=['field_id', 'year'], how='inner'
)

df_h = df[df['group'] == 'High'].reset_index(drop=True)
df_l = df[df['group'] == 'Low'].reset_index(drop=True)
n_h  = len(df_h)
n_l  = len(df_l)

print(f'\n統計対象: High={n_h} 件 / Low={n_l} 件')
print('（Mid群は分析から除外）')
print()
print('年別内訳:')
print(df[df['group'].isin(['High','Low'])].groupby(['year','group']).size().unstack(fill_value=0).to_string())

# =============================================================================
# 5. 統計検定 + ドットプロット作成
# =============================================================================
print()
print('=' * 55)
print('  土壌特徴量 高収量 vs 低収量 比較（松本市）')
print(f'  ※ n={n_h}/{n_l} と少標本 → 低検出力に注意')
print('=' * 55)

stats_rows = []
n_cols = 7
n_rows = int(np.ceil(len(SOIL_COLS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.5 * n_rows))
axes = np.array(axes).flatten()

for i, col in enumerate(SOIL_COLS):
    ax = axes[i]
    h_v = df_h[col].dropna()
    l_v = df_l[col].dropna()
    nh, nl = len(h_v), len(l_v)

    # --- MWU検定 ---
    if nh >= MIN_PER_GROUP and nl >= MIN_PER_GROUP:
        try:
            _, p = mannwhitneyu(h_v, l_v, alternative='two-sided')
            sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        except Exception:
            p, sig = np.nan, 'n.a.'
    else:
        p, sig = np.nan, 'n.a.'

    # --- ドットプロット（全点を表示）---
    x_h = np.random.uniform(0.75, 1.25, size=nh)
    x_l = np.random.uniform(1.75, 2.25, size=nl)
    ax.scatter(x_h, h_v.values, color=COL_HIGH, s=55, alpha=0.80,
               edgecolors='white', lw=0.5, zorder=3)
    ax.scatter(x_l, l_v.values, color=COL_LOW,  s=55, alpha=0.80,
               edgecolors='white', lw=0.5, zorder=3)

    # 中央値の横線
    if nh > 0:
        ax.hlines(h_v.median(), 0.7, 1.3, colors=COL_HIGH, lw=2.0, zorder=4)
    if nl > 0:
        ax.hlines(l_v.median(), 1.7, 2.3, colors=COL_LOW,  lw=2.0, zorder=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'High\n(n={nh})', f'Low\n(n={nl})'], fontsize=8)
    ax.set_xlim(0.4, 2.6)
    ax.set_title(SOIL_LABELS.get(col, col), fontsize=8)
    ax.grid(axis='y', ls='--', alpha=0.4)

    # 注釈
    sig_col = 'red' if sig == '**' else ('orange' if sig == '*' else 'gray')
    p_str   = f'{p:.3f}' if not np.isnan(p) else 'n.a.'
    h_med   = f'{h_v.median():.2f}' if nh > 0 else '-'
    l_med   = f'{l_v.median():.2f}' if nl > 0 else '-'
    ax.text(0.97, 0.97,
            f'{sig}  p={p_str}\nH中央値={h_med}\nL中央値={l_med}',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=7, color=sig_col,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=sig_col, lw=0.8))

    print(f'  {col:<10}: High中央値={h_med:>8} / Low中央値={l_med:>8} | {sig}  p={p_str}')

    stats_rows.append({
        'variable':     col,
        'n_high':       nh,
        'n_low':        nl,
        'high_median':  round(float(h_v.median()), 4) if nh > 0 else np.nan,
        'low_median':   round(float(l_v.median()), 4) if nl > 0 else np.nan,
        'mwu_p':        p,
        'sig':          sig,
    })

# 余ったパネルを非表示
for ax_off in axes[len(SOIL_COLS):]:
    ax_off.set_visible(False)

fig.suptitle(
    f'松本市 — 土壌特徴量 High vs Low 比較（収穫前後採取：10〜12月）\n'
    f'High n={n_h} / Low n={n_l}  ※MWU検定 / 横線=中央値 / 低検出力（n小）に注意',
    fontsize=11)
plt.tight_layout()
out_fig = os.path.join(OUT_DIR, '10_matsumoto_soil_postharv.png')
plt.savefig(out_fig, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n  保存: {out_fig}')

# =============================================================================
# 6. 統計サマリ CSV
# =============================================================================
stats_df = pd.DataFrame(stats_rows)
out_csv  = os.path.join(OUT_DIR, '10_matsumoto_soil_stats.csv')
stats_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'  統計CSV保存: {out_csv}')

print()
print('=' * 55)
print('  有意差あり（p<0.05）の変数:')
sig_df = stats_df[stats_df['sig'].isin(['*', '**'])]
if len(sig_df) == 0:
    print('    （有意差なし — 検出力不足の可能性あり）')
else:
    for _, r in sig_df.iterrows():
        print(f"    {r['variable']:<10}: {r['sig']}  p={r['mwu_p']:.4f}"
              f"  High中央値={r['high_median']}  Low中央値={r['low_median']}")
print('=' * 55)
