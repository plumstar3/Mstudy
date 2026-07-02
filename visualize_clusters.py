"""
visualize_clusters.py
===========================================================
各クラスタの年別収量を1つずつ丁寧に可視化する。
また、収量偏差閾値の動作を説明する補足図も生成する。
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 日本語フォント
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR  = 'outputs/data_analysis'
FIELD_DB = 'data/processed/FieldData_fieldid.db'
YEARS    = [2015, 2016, 2017, 2018]
COLORS_YR = {2015: '#4C72B0', 2016: '#55A868', 2017: '#C44E52', 2018: '#8172B2'}
DIST_THR = 300
DEV_THR  = 70

# ── データ読み込み ─────────────────────────────────────────────────────────────
cl_df = pd.read_csv(f'{OUT_DIR}/continuous_clusters.csv', encoding='utf-8-sig')

conn = sqlite3.connect(FIELD_DB)
df = pd.read_sql('''
    SELECT field_id, year, yield, lat, lon
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND year BETWEEN 2015 AND 2018
''', conn)
conn.close()
df['field_id'] = df['field_id'].astype(int)
df['year']     = df['year'].astype(int)
df['yield']    = pd.to_numeric(df['yield'], errors='coerce')

# 年別平均収量（偏差計算用）
year_means = df.groupby('year')['yield'].mean()

# ── 図1: クラスタごと年別収量（4年連続・3年連続・2年連続 別ページ） ──────────

def plot_cluster_grid(subset_df, title, fname, cols=5):
    """クラスタ一覧をグリッドで表示"""
    n = len(subset_df)
    if n == 0:
        return
    rows = -(-n // cols)   # ceil division
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0),
                              facecolor='#f5f5f5')
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax_i, (_, cl_row) in enumerate(subset_df.iterrows()):
        ax = axes[ax_i]
        cid    = cl_row['cluster_id']
        n_yr   = int(cl_row['n_years'])
        n_fld  = int(cl_row['n_fields'])

        # 年別収量と偏差
        yr_data = []
        for yr in YEARS:
            y_val = cl_row.get(f'yield_{yr}')
            d_val = cl_row.get(f'dev_{yr}')
            if pd.notna(y_val):
                yr_data.append({'year': yr, 'yield': y_val, 'dev': d_val})
        if not yr_data:
            ax.set_visible(False)
            continue

        yr_plot = [d['year'] for d in yr_data]
        yl_plot = [d['yield'] for d in yr_data]
        dv_plot = [d['dev'] for d in yr_data]

        # 折れ線
        ax.plot(yr_plot, yl_plot, '-', color='#555555', linewidth=1.5, zorder=2)
        # 年ごとに色の点
        for d in yr_data:
            ax.scatter(d['year'], d['yield'],
                       s=80, color=COLORS_YR[d['year']],
                       zorder=3, edgecolors='white', linewidths=0.7)
            # 収量値ラベル
            ax.text(d['year'], d['yield'] + 6,
                    f"{d['yield']:.0f}",
                    ha='center', va='bottom', fontsize=7.5, color='#222222',
                    fontweight='bold')
            # 偏差ラベル（グレー小文字）
            sign = '+' if d['dev'] >= 0 else ''
            ax.text(d['year'], d['yield'] - 14,
                    f"{sign}{d['dev']:.0f}",
                    ha='center', va='top', fontsize=6.5,
                    color='#e74c3c' if d['dev'] >= 0 else '#3498db')

        # 全体平均の横線
        all_mean = cl_row['yield_mean']
        ax.axhline(all_mean, color='#888888', linestyle=':', linewidth=1.0, alpha=0.8)

        ax.set_title(f'{cid}  ({n_yr}年 / {n_fld}圃場)',
                     fontsize=8.5, fontweight='bold', pad=3)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y) for y in YEARS], fontsize=7)
        ax.set_ylim(max(0, min(yl_plot) - 60), max(yl_plot) + 60)
        ax.set_ylabel('kg/10a', fontsize=7)
        ax.set_facecolor('#fafafa')
        ax.grid(True, alpha=0.3, axis='y', linewidth=0.7)
        ax.tick_params(axis='y', labelsize=7)

    # 余ったサブプロットを非表示
    for i in range(len(subset_df), len(axes)):
        axes[i].set_visible(False)

    # 凡例
    patches = [mpatches.Patch(color=COLORS_YR[yr], label=f'{yr}年') for yr in YEARS]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = f'{OUT_DIR}/{fname}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  保存: {path}')

# 4年連続クラスタ
four_yr  = cl_df[cl_df['n_years'] == 4].reset_index(drop=True)
three_yr = cl_df[cl_df['n_years'] == 3].reset_index(drop=True)
two_yr   = cl_df[cl_df['n_years'] == 2].reset_index(drop=True)

print('クラスタグリッド図を生成中...')
plot_cluster_grid(four_yr,  f'4年連続クラスタ（距離≤{DIST_THR}m / 偏差差≤{DEV_THR}kg）',
                  'clusters_4yr.png', cols=3)
plot_cluster_grid(three_yr, f'3年連続クラスタ（距離≤{DIST_THR}m / 偏差差≤{DEV_THR}kg）',
                  'clusters_3yr.png', cols=5)
plot_cluster_grid(two_yr,   f'2年連続クラスタ（距離≤{DIST_THR}m / 偏差差≤{DEV_THR}kg）',
                  'clusters_2yr.png', cols=6)

# ── 図2: 収量偏差閾値の動作解説図 ─────────────────────────────────────────────
print('\n偏差閾値解説図を生成中...')

fig2 = plt.figure(figsize=(16, 7), facecolor='#f9f9f9')
gs   = GridSpec(1, 2, figure=fig2, wspace=0.35)

# ---- 左パネル: 偏差の概念 ----
ax_l = fig2.add_subplot(gs[0])

# 年平均ライン
ax_l.axhline(0, color='black', linewidth=2.0, label='各年の全圃場平均収量 (=偏差0)', zorder=2)

# 架空の例で説明
ex = [
    # 同クラスタ候補A（多収グループ）
    {'yr': 2015, 'yield': 360, 'dev': +103, 'label': '圃場A (2015)', 'color': '#C44E52', 'group': '多収グループ'},
    {'yr': 2016, 'yield': 330, 'dev': +71,  'label': '圃場B (2016)', 'color': '#C44E52', 'group': '多収グループ'},
    # 同クラスタ候補B（低収グループ）
    {'yr': 2015, 'yield': 140, 'dev': -117, 'label': '圃場C (2015)', 'color': '#4C72B0', 'group': '低収グループ'},
    {'yr': 2016, 'yield': 170, 'dev': -89,  'label': '圃場D (2016)', 'color': '#4C72B0', 'group': '低収グループ'},
]

# 各年の実際の平均収量を0基準にして偏差で表示
for e in ex:
    ax_l.scatter(e['yr'], e['dev'], s=120, color=e['color'],
                 zorder=4, edgecolors='white', linewidths=1.0)
    offset = 8 if e['dev'] > 0 else -16
    ax_l.text(e['yr'] + 0.05, e['dev'] + offset,
              f"{e['label']}\n偏差={e['dev']:+d}kg",
              ha='left', va='center', fontsize=8.5, color=e['color'],
              fontweight='bold')

# A↔B の偏差差（繋がるペア）
ax_l.annotate('', xy=(2016, 71), xytext=(2015, 103),
              arrowprops=dict(arrowstyle='<->', color='#C44E52', lw=1.5))
ax_l.text(2015.5, 90, '差=32kg\n≤70 → 同クラスタ✓',
          ha='center', va='center', fontsize=8, color='#C44E52',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0', alpha=0.9))

# A↔C の偏差差（繋がらないペア）
ax_l.annotate('', xy=(2015, -117), xytext=(2015, 103),
              arrowprops=dict(arrowstyle='<->', color='#888888', lw=1.5, linestyle='dashed'))
ax_l.text(2014.7, -7, '差=220kg\n>70 → 別クラスタ✗',
          ha='center', va='center', fontsize=8, color='#555555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#eeeeee', alpha=0.9))

# C↔D の偏差差（繋がるペア）
ax_l.annotate('', xy=(2016, -89), xytext=(2015, -117),
              arrowprops=dict(arrowstyle='<->', color='#4C72B0', lw=1.5))
ax_l.text(2015.5, -107, '差=28kg\n≤70 → 同クラスタ✓',
          ha='center', va='center', fontsize=8, color='#4C72B0',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#ddeeff', alpha=0.9))

ax_l.axhspan(-DEV_THR/2, DEV_THR/2, alpha=0.08, color='green',
             label=f'偏差差 ≤ {DEV_THR}kg の帯（参考）')
ax_l.set_xlim(2014.3, 2016.8)
ax_l.set_ylim(-160, 150)
ax_l.set_xticks([2015, 2016])
ax_l.set_xticklabels(['2015年', '2016年'], fontsize=10)
ax_l.set_ylabel('収量偏差 (実際の収量 − その年の全圃場平均)', fontsize=9)
ax_l.set_title('収量偏差閾値の動作イメージ\n（0より上＝多収、0より下＝低収）',
               fontsize=11, fontweight='bold')
ax_l.grid(True, alpha=0.3, axis='y')
ax_l.set_facecolor('#fdfdfd')

# ---- 右パネル: 実データの偏差分布と閾値 ----
ax_r = fig2.add_subplot(gs[1])

for yr, col in COLORS_YR.items():
    mask = df['year'] == yr
    devs = df.loc[mask, 'yield'] - year_means[yr]
    ax_r.hist(devs, bins=25, alpha=0.45, color=col, label=f'{yr}年',
              edgecolor='white', linewidth=0.5)

ax_r.axvline(-DEV_THR, color='red', linestyle='--', lw=2,
             label=f'偏差差 閾値 ±{DEV_THR}kg')
ax_r.axvline(+DEV_THR, color='red', linestyle='--', lw=2)
ax_r.axvline(0, color='black', linestyle='-', lw=1.5, alpha=0.7)

# 多収ゾーン・低収ゾーンの塗りつぶし
ax_r.axvspan(DEV_THR, ax_r.get_xlim()[1] if ax_r.get_xlim()[1] > 200 else 300,
             alpha=0.07, color='#C44E52', label='多収ゾーン（偏差>+70）')
ax_r.axvspan(-300, -DEV_THR,
             alpha=0.07, color='#4C72B0', label='低収ゾーン（偏差<−70）')

ax_r.set_xlabel('収量偏差 (kg/10a) ＝ 収量 − その年の全圃場平均', fontsize=9)
ax_r.set_ylabel('圃場数', fontsize=9)
ax_r.set_title(f'実データの収量偏差分布\n（偏差差が >{DEV_THR}kg のペアは別クラスタ扱い）',
               fontsize=11, fontweight='bold')
ax_r.legend(fontsize=8, loc='upper left')
ax_r.set_facecolor('#fdfdfd')
ax_r.grid(True, alpha=0.3)

fig2.suptitle('収量偏差閾値（DEV_THR）の仕組み',
              fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2_path = f'{OUT_DIR}/dev_threshold_explanation.png'
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  偏差閾値解説図: {fig2_path}')

print('\n完了')
