"""
11_soilmoisture_analysis.py
============================
土壌水分量（VWC）の時系列を多収圃場群と低収圃場群で比較する。

[グループ定義]
  土壌水分データがある306圃場×年の中で、
  年ごとに収量の上位33% → High、下位33% → Low と定義。

[時間軸]
  DAP% = (測定日 - 播種日) / 生育日数 × 100
  播種〜収穫を0〜100%に正規化。

[分析方法]
  DAP% を 5%刻みのビンに分け、各ビンで:
    - High群・Low群それぞれの平均VWC ± SE を算出
    - Mann-Whitney U 検定（n_High ≥ 5 かつ n_Low ≥ 5 の場合のみ）
  主要分析範囲: カバレッジ ≥ 80% (≈245圃場) の DAP 30〜85%

[出力]
  outputs/soil_analysis/
    11_soilmoisture_ts.png      (時系列比較グラフ)
    11_soilmoisture_stats.csv   (DAP%ビン別統計)
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
# 定数
# =============================================================================
_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
OUT_DIR  = os.path.join(_ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2016, 2017, 2018]
GROUP_QUANTILE = 0.33

DAP_BIN_WIDTH  = 5        # DAP%ビン幅
DAP_BINS       = np.arange(-5, 125, DAP_BIN_WIDTH)   # ビンエッジ
DAP_CENTERS    = DAP_BINS[:-1] + DAP_BIN_WIDTH / 2   # ビン中心

MIN_PER_GROUP  = 5        # 検定に必要な最低n
COVERAGE_THR   = 245      # カバレッジ閾値（圃場数）→ 主要分析範囲フラグ用

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'
COL_MAIN = '#2ECC71'   # カバレッジ主要範囲の背景色

# =============================================================================
# 1. データ読み込み
# =============================================================================
print('=' * 60)
print('  データ読み込み')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)

sm = pd.read_sql(
    'SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL',
    conn)
quest = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2016 AND 2018
""", conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce').dropna().astype(int)
sm['year']     = sm['year'].astype(int)
sm['date']     = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC']      = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['field_id', 'date', 'VWC'])

quest = quest.dropna(subset=['field_id'])
quest['field_id']     = quest['field_id'].astype(int)
quest['year']         = quest['year'].astype(int)
quest['seed_date']    = pd.to_datetime(quest['seed_date'], errors='coerce')
quest['harvest_date'] = pd.to_datetime(quest['harvest_date'], errors='coerce')
quest['growing_days'] = (quest['harvest_date'] - quest['seed_date']).dt.days
quest = quest[quest['growing_days'] > 0].copy()

print(f'  SolidMoisture レコード数: {len(sm):,} 件')
print(f'  収量・播種日データ: {len(quest)} 件')

# =============================================================================
# 2. DAP% の計算
# =============================================================================
df = sm.merge(
    quest[['field_id', 'year', 'yield', 'seed_date', 'harvest_date', 'growing_days']],
    on=['field_id', 'year'], how='inner'
)
df['DAP']     = (df['date'] - df['seed_date']).dt.days
df['DAP_pct'] = df['DAP'] / df['growing_days'] * 100

print(f'  マッチ済みレコード: {len(df):,} 件 / {df["field_id"].nunique()} 圃場')

# =============================================================================
# 3. H/L グループ定義（土壌水分圃場内・年ごと）
# =============================================================================
# 各field_id×yearの代表収量を取得（1圃場1値）
field_yield = df[['field_id', 'year', 'yield']].drop_duplicates(['field_id', 'year'])

field_yield['group'] = 'Mid'
for yr in TARGET_YEARS:
    mask = field_yield['year'] == yr
    sub  = field_yield[mask]
    if len(sub) < 6:
        continue
    lo = sub['yield'].quantile(GROUP_QUANTILE)
    hi = sub['yield'].quantile(1 - GROUP_QUANTILE)
    field_yield.loc[mask & (field_yield['yield'] <= lo), 'group'] = 'Low'
    field_yield.loc[mask & (field_yield['yield'] >= hi), 'group'] = 'High'

print('\n  収量グループ（年別）:')
print(field_yield.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())

n_high_total = (field_yield['group'] == 'High').sum()
n_low_total  = (field_yield['group'] == 'Low').sum()

# グループ情報を時系列データに結合
df = df.merge(field_yield[['field_id', 'year', 'group']], on=['field_id', 'year'], how='left')

# =============================================================================
# 4. DAP%ビンごとの統計計算
# =============================================================================
print('\n  DAP%ビン別集計中...')

def assign_bin(dap_pct):
    """DAP%をビンインデックスに割り当て。"""
    idx = int((dap_pct - DAP_BINS[0]) / DAP_BIN_WIDTH)
    return idx if 0 <= idx < len(DAP_CENTERS) else None

df['bin_idx'] = df['DAP_pct'].apply(assign_bin)

stats_rows = []
for i, center in enumerate(DAP_CENTERS):
    sub = df[df['bin_idx'] == i]
    h   = sub[sub['group'] == 'High']['VWC']
    l   = sub[sub['group'] == 'Low']['VWC']
    mid = sub[sub['group'] == 'Mid']['VWC']

    nh, nl, nm = len(h), len(l), len(mid)
    total_fields = sub['field_id'].nunique()

    if nh >= MIN_PER_GROUP and nl >= MIN_PER_GROUP:
        _, p = mannwhitneyu(h, l, alternative='two-sided')
        sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
    else:
        p, sig = np.nan, 'n.a.'

    stats_rows.append({
        'DAP_pct_center': center,
        'total_fields':   total_fields,
        'n_high': nh,
        'n_low':  nl,
        'high_mean': float(h.mean())  if nh > 0 else np.nan,
        'high_se':   float(h.sem())   if nh > 1 else np.nan,
        'low_mean':  float(l.mean())  if nl > 0 else np.nan,
        'low_se':    float(l.sem())   if nl > 1 else np.nan,
        'mwu_p':  p,
        'sig':    sig,
        'main_range': total_fields >= COVERAGE_THR,
    })

stats_df = pd.DataFrame(stats_rows)

# CSV保存
csv_path = os.path.join(OUT_DIR, '11_soilmoisture_stats.csv')
stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'  統計CSV保存: {csv_path}')

# 有意な時期を表示
print('\n  *** 有意差あり (p<0.05) の DAP% 区間 ***')
sig_rows = stats_df[stats_df['sig'].isin(['*', '**'])]
if len(sig_rows) == 0:
    print('  （なし）')
else:
    for _, r in sig_rows.iterrows():
        print(f"  DAP {r['DAP_pct_center']:5.1f}%: {r['sig']:3s}  p={r['mwu_p']:.4f}"
              f"  High={r['high_mean']:.4f}  Low={r['low_mean']:.4f}"
              f"  n={r['n_high']}/{r['n_low']}")

# =============================================================================
# 5. 可視化
# =============================================================================
fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(3, 1, height_ratios=[5, 1.5, 1], hspace=0.08)
ax_main = fig.add_subplot(gs[0])
ax_sig  = fig.add_subplot(gs[1], sharex=ax_main)
ax_n    = fig.add_subplot(gs[2], sharex=ax_main)

x  = stats_df['DAP_pct_center'].values
hm = stats_df['high_mean'].values
hs = stats_df['high_se'].values
lm = stats_df['low_mean'].values
ls = stats_df['low_se'].values
ns_h = stats_df['n_high'].values
ns_l = stats_df['n_low'].values
sigs = stats_df['sig'].values
main = stats_df['main_range'].values

# --- メインプロット: VWC時系列 ---
# カバレッジ主要範囲を背景でハイライト
main_x = x[main]
if len(main_x) > 0:
    ax_main.axvspan(main_x[0] - DAP_BIN_WIDTH/2, main_x[-1] + DAP_BIN_WIDTH/2,
                    color=COL_MAIN, alpha=0.07, label='主要カバレッジ範囲 (>=245圃場)')

# High群
valid_h = ~np.isnan(hm) & (ns_h >= MIN_PER_GROUP)
ax_main.plot(x[valid_h], hm[valid_h], color=COL_HIGH, lw=2.5, label=f'High群 (n={n_high_total})')
ax_main.fill_between(x[valid_h],
                     hm[valid_h] - hs[valid_h],
                     hm[valid_h] + hs[valid_h],
                     color=COL_HIGH, alpha=0.18)

# Low群
valid_l = ~np.isnan(lm) & (ns_l >= MIN_PER_GROUP)
ax_main.plot(x[valid_l], lm[valid_l], color=COL_LOW, lw=2.5, label=f'Low群 (n={n_low_total})')
ax_main.fill_between(x[valid_l],
                     lm[valid_l] - ls[valid_l],
                     lm[valid_l] + ls[valid_l],
                     color=COL_LOW, alpha=0.18)

ax_main.set_ylabel('土壌体積含水率 VWC', fontsize=11)
ax_main.set_ylim(0.05, 0.65)
ax_main.axvline(0,   color='gray', lw=1.0, ls='--', alpha=0.6)
ax_main.axvline(100, color='gray', lw=1.0, ls='--', alpha=0.6)
ax_main.legend(fontsize=10, loc='upper right')
ax_main.grid(axis='y', ls='--', alpha=0.35)
ax_main.set_title(
    '土壌水分量（VWC）の生育ステージ別推移 — High vs Low 収量群\n'
    '（土壌水分測定圃場内でのH/L定義・DAP%正規化）',
    fontsize=12)
plt.setp(ax_main.get_xticklabels(), visible=False)

# --- 有意性パネル ---
sig_colors = {'**': '#C0392B', '*': '#E67E22', 'ns': '#BDC3C7', 'n.a.': '#ECF0F1'}
for xi, si in zip(x, sigs):
    ax_sig.bar(xi, 1, width=DAP_BIN_WIDTH * 0.9, color=sig_colors.get(si, '#ECF0F1'),
               align='center')
# 凡例代わりのラベル
for label, color in [('** p<0.01', '#C0392B'), ('* p<0.05', '#E67E22'),
                     ('ns', '#BDC3C7'), ('n.a. (n<5)', '#ECF0F1')]:
    ax_sig.bar([], [], color=color, label=label)
ax_sig.set_yticks([])
ax_sig.set_ylabel('有意性', fontsize=9)
ax_sig.legend(fontsize=8, loc='upper right', ncol=4)
ax_sig.set_ylim(0, 1.5)
plt.setp(ax_sig.get_xticklabels(), visible=False)

# --- nパネル ---
ax_n.plot(x, ns_h, color=COL_HIGH, lw=1.5, label='High n')
ax_n.plot(x, ns_l, color=COL_LOW,  lw=1.5, label='Low n')
ax_n.axhline(MIN_PER_GROUP, color='gray', lw=1, ls=':', alpha=0.7)
ax_n.fill_between(x, 0, [n if main[i] else 0 for i, n in enumerate(ns_h)],
                  alpha=0.07, color=COL_MAIN)
ax_n.set_xlabel('生育ステージ（播種を0%、収穫を100%とした DAP%）', fontsize=11)
ax_n.set_ylabel('圃場数 n', fontsize=9)
ax_n.legend(fontsize=8, loc='upper right')
ax_n.grid(axis='y', ls='--', alpha=0.35)
ax_n.set_xlim(-10, 115)

out_path = os.path.join(OUT_DIR, '11_soilmoisture_ts.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'\n  図保存: {out_path}')

print()
print('=' * 60)
print('  完了！')
print(f'  出力先: {OUT_DIR}')
print('=' * 60)
