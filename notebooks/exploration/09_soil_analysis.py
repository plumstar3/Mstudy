"""
09_soil_analysis.py
====================
土壌特徴量（Solidテーブル）の High / Low 収量群間比較。

[採取タイミングによるグループ分け]
  播種前後群 (pre_sow)  : 採取月 5〜7月
  収穫前後群 (post_harv): 採取月 10〜12月
  除外               : 採取月 8〜9月

[収量グループ定義]
  07_yield_group_analysis.py と同じ全体比較方式:
  NDVI保有 & 有効圃場×年の中で上位33% → High、下位33% → Low

[解析特徴量（Solidテーブル）]
  pH, CaO, MgO, K2O, P2O2, N_able, N_per, C_per, CEC,
  H3PO4, B_able, Mo_able, Mn_c

[検定]
  Mann-Whitney U 検定（連続量 × 小〜中標本）
  有意水準: ** p<0.01, * p<0.05, ns

[出力]
  outputs/soil_analysis/
    01_soil_presow.png     (播種前後群の箱ひげ図)
    02_soil_postharv.png   (収穫前後群の箱ひげ図)
    03_soil_stats.csv      (統計サマリ)
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

# フォント設定
for _fn in ['IPAexGothic', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# =============================================================================
# パス・定数
# =============================================================================
_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
NDVI_CSV = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
OUT_DIR  = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33   # 上位/下位 33%

# 採取月による群分け
PRE_SOW_MONTHS  = [5, 6, 7]       # 播種前後
POST_HARV_MONTHS = [10, 11, 12]   # 収穫前後
EXCLUDE_MONTHS   = [8, 9]         # 除外

# 解析する土壌特徴量
SOIL_COLS = ['pH', 'CaO', 'MgO', 'K2O', 'P2O2',
             'N_able', 'N_per', 'C_per', 'CEC',
             'H3PO4', 'B_able', 'Mo_able', 'Mn_c']

# 日本語ラベル
SOIL_LABELS = {
    'pH':     'pH',
    'CaO':    '石灰 CaO\n(mg/100g)',
    'MgO':    '苦土 MgO\n(mg/100g)',
    'K2O':    '加里 K2O\n(mg/100g)',
    'P2O2':   'リン酸 P2O2\n(mg/100g)',
    'N_able': '有効態窒素\n(mg/100g)',
    'N_per':  '全窒素 N\n(%)',
    'C_per':  '全炭素 C\n(%)',
    'CEC':    '陽イオン交換容量\nCEC (meq/100g)',
    'H3PO4':  'リン酸吸収係数\n(mg/100g)',
    'B_able': '有効態ホウ素\n(mg/kg)',
    'Mo_able':'有効態モリブデン\n(μg/kg)',
    'Mn_c':   '交換性マンガン\n(mg/kg)',
}

MIN_PER_GROUP = 5   # 検定に必要な最低サンプル数

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'

# =============================================================================
# データ読み込み
# =============================================================================
print('=' * 60)
print('  データ読み込み')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)

# 収量データ（NDVI保有圃場に絞る）
quest_df = pd.read_sql("""
    SELECT field_id, year, yield
    FROM Questionaire
    WHERE yield IS NOT NULL
      AND seed_date IS NOT NULL
      AND harvest_date IS NOT NULL
      AND year BETWEEN 2015 AND 2018
""", conn)
quest_df = quest_df.dropna(subset=['field_id'])
quest_df['field_id'] = quest_df['field_id'].astype(int)
quest_df['year']     = quest_df['year'].astype(int)

# NDVI保有フィルタ
ndvi_df = pd.read_csv(NDVI_CSV)
ndvi_df['field_id'] = ndvi_df['field_id'].astype(int)
ndvi_df['year']     = ndvi_df['year'].astype(int)
ndvi_fields = set(zip(ndvi_df['field_id'], ndvi_df['year']))
quest_df = quest_df[
    quest_df.apply(lambda r: (int(r['field_id']), int(r['year'])) in ndvi_fields, axis=1)
].copy().reset_index(drop=True)
print(f'  有効収量レコード: {len(quest_df)} 件')

# 土壌データ（date列あり）
soil_cols_str = ', '.join(SOIL_COLS)
solid_df = pd.read_sql(f"""
    SELECT field_id, year, place, date, {soil_cols_str}
    FROM Solid
    WHERE date IS NOT NULL AND field_id IS NOT NULL
""", conn)
conn.close()

solid_df = solid_df.dropna(subset=['field_id'])
solid_df['field_id'] = solid_df['field_id'].astype(int)
solid_df['year']     = solid_df['year'].astype(int)
solid_df['date']     = pd.to_datetime(solid_df['date'])
solid_df['month']    = solid_df['date'].dt.month

for c in SOIL_COLS:
    solid_df[c] = pd.to_numeric(solid_df[c], errors='coerce')

print(f'  土壌レコード（date付き）: {len(solid_df)} 件')

# 採取タイミングで群分け
solid_df['timing'] = None
solid_df.loc[solid_df['month'].isin(PRE_SOW_MONTHS),   'timing'] = 'pre_sow'
solid_df.loc[solid_df['month'].isin(POST_HARV_MONTHS),  'timing'] = 'post_harv'
solid_df.loc[solid_df['month'].isin(EXCLUDE_MONTHS),    'timing'] = 'exclude'
solid_df = solid_df[solid_df['timing'] != 'exclude'].copy()

print(f'  採取タイミング別件数:')
print(f'    播種前後（5〜7月） : {(solid_df["timing"]=="pre_sow").sum()} 件')
print(f'    収穫前後（10〜12月）: {(solid_df["timing"]=="post_harv").sum()} 件')

# =============================================================================
# High / Low 収量グループ定義（全体比較：07と同じ）
# =============================================================================
quest_df['group'] = 'Mid'
for yr in TARGET_YEARS:
    mask = quest_df['year'] == yr
    sub  = quest_df[mask]
    if len(sub) < 6:
        continue
    lo_thr = sub['yield'].quantile(GROUP_QUANTILE)
    hi_thr = sub['yield'].quantile(1 - GROUP_QUANTILE)
    quest_df.loc[mask & (quest_df['yield'] <= lo_thr), 'group'] = 'Low'
    quest_df.loc[mask & (quest_df['yield'] >= hi_thr), 'group'] = 'High'

n_high = (quest_df['group'] == 'High').sum()
n_low  = (quest_df['group'] == 'Low').sum()
print(f'  High群: {n_high} 件 / Low群: {n_low} 件 / Mid: {(quest_df["group"]=="Mid").sum()} 件')

# 土壌データに収量グループを結合
# （同一field_id×yearで土壌と収量を突合）
solid_df = solid_df.merge(
    quest_df[['field_id', 'year', 'yield', 'group']],
    on=['field_id', 'year'], how='inner'
)
print(f'\n  収量グループ付き土壌レコード: {len(solid_df)} 件')
print(f'  タイミング × グループ:')
print(solid_df.groupby(['timing', 'group']).size().unstack(fill_value=0).to_string())

# =============================================================================
# 統計関数
# =============================================================================
def run_mwu(h_vals, l_vals):
    h = h_vals.dropna()
    l = l_vals.dropna()
    if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP:
        return np.nan, 'n.a.'
    try:
        _, p = mannwhitneyu(h, l, alternative='two-sided')
        sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        return p, sig
    except Exception:
        return np.nan, 'n.a.'


def plot_soil_group(df_timing, timing_label, out_path, stats_rows, timing_key):
    """土壌特徴量の箱ひげ図（High vs Low）を描画。"""
    df_h = df_timing[df_timing['group'] == 'High']
    df_l = df_timing[df_timing['group'] == 'Low']
    n_h  = len(df_h)
    n_l  = len(df_l)

    n_cols = 7
    n_rows = int(np.ceil(len(SOIL_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(SOIL_COLS):
        ax = axes[i]
        h_v = df_h[col].dropna()
        l_v = df_l[col].dropna()

        if len(h_v) >= 2 and len(l_v) >= 2:
            bp = ax.boxplot([h_v, l_v],
                            labels=[f'High\n(n={len(h_v)})', f'Low\n(n={len(l_v)})'],
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', lw=2.0))
            bp['boxes'][0].set_facecolor(COL_HIGH); bp['boxes'][0].set_alpha(0.65)
            bp['boxes'][1].set_facecolor(COL_LOW);  bp['boxes'][1].set_alpha(0.65)
        else:
            ax.text(0.5, 0.5, 'データ不足', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')

        p, sig = run_mwu(h_v, l_v)
        sig_col = 'red' if sig == '**' else ('orange' if sig == '*' else 'gray')
        p_str   = f'{p:.3f}' if not np.isnan(p) else 'n.a.'
        ax.set_title(SOIL_LABELS.get(col, col), fontsize=8)
        ax.grid(axis='y', ls='--', alpha=0.4)
        ax.text(0.97, 0.97, f'{sig}\np={p_str}',
                ha='right', va='top', transform=ax.transAxes,
                fontsize=8, color=sig_col,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=sig_col, lw=0.8))

        stats_rows.append({
            'timing': timing_key,
            'variable': col,
            'n_high': len(h_v),
            'n_low':  len(l_v),
            'high_median': round(float(h_v.median()), 4) if len(h_v) > 0 else np.nan,
            'low_median':  round(float(l_v.median()), 4) if len(l_v) > 0 else np.nan,
            'mwu_p': p,
            'sig': sig,
        })

    # 余ったパネルを非表示
    for ax_off in axes[len(SOIL_COLS):]:
        ax_off.set_visible(False)

    fig.suptitle(f'土壌特徴量 — {timing_label}\n（High n={n_h} / Low n={n_l}、MWU検定）',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out_path}')


# =============================================================================
# 分析実行
# =============================================================================
stats_rows = []

timing_configs = [
    ('pre_sow',   '播種前後（5〜7月採取）',   '01_soil_presow.png'),
    ('post_harv', '収穫前後（10〜12月採取）', '02_soil_postharv.png'),
]

print()
print('=' * 60)
print('  土壌特徴量 High/Low 比較')
print('=' * 60)

for timing_key, timing_label, fname in timing_configs:
    df_t = solid_df[solid_df['timing'] == timing_key]
    n_h  = (df_t['group'] == 'High').sum()
    n_l  = (df_t['group'] == 'Low').sum()
    print(f'\n  [{timing_label}]  High={n_h} / Low={n_l}')

    if n_h < MIN_PER_GROUP or n_l < MIN_PER_GROUP:
        print(f'    → 標本不足のためスキップ')
        continue

    plot_soil_group(
        df_timing=df_t,
        timing_label=timing_label,
        out_path=os.path.join(OUT_DIR, fname),
        stats_rows=stats_rows,
        timing_key=timing_key,
    )

# =============================================================================
# 統計サマリ CSV
# =============================================================================
stats_df = pd.DataFrame(stats_rows)
csv_path = os.path.join(OUT_DIR, '03_soil_stats.csv')
stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n  統計サマリ保存: {csv_path}')

# 有意変数の表示
print('\n  ★ 有意差あり（p<0.05）の変数一覧:')
sig_df = stats_df[stats_df['sig'].isin(['*', '**'])]
if len(sig_df) == 0:
    print('    （有意差なし）')
else:
    for _, r in sig_df.iterrows():
        timing_jp = '播種前後' if r['timing'] == 'pre_sow' else '収穫前後'
        print(f'    {timing_jp:6s} | {r["variable"]:10s} | {r["sig"]:3s} '
              f'| p={r["mwu_p"]:.4f} '
              f'| High中央値={r["high_median"]} / Low中央値={r["low_median"]}')

print()
print('=' * 60)
print('  土壌分析完了！')
print(f'  出力先: {OUT_DIR}')
print('=' * 60)
