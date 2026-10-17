"""
16_disease_intra_analysis.py
=============================
病害特徴量の「同一地域内」多収 vs 低収圃場比較。

[対象地域の選定]
  病害データ（sick/weed/wet/bug/unripen/lay/loss）が揃っている圃場×年の件数が
  最も多い市町村を自動選定する（TARGET_MUNI で手動指定も可能）。
  ※ NDVIの有無は条件にしない（病害データ単独で揃い具合を評価）。
  ※ 土壌・土壌水分は別途それぞれ最適な市町村を選ぶ想定。

[グループ定義]
  対象市町村内で、年ごとに収量の上位33%→High、下位33%→Low。
  全年のHigh/Lowをプールして検定する。

[検定]
  - 二値病害（sick/weed/wet/bug）: 1=なし / 2以上=あり。不明値はNaN除外
  - 順序病害（unripen/lay/loss）: Mann-Whitney U + (1=なし / 2,3,4=あり) のχ²検定・Fisher
  - 気象フラグ（heavy_rain 等）: 発生あり率の比較
  ※ 標本が小さいためFisher exact test（正確確率検定）を主、χ²検定も併記する。

[出力]
  outputs/disease_intra_analysis/
    16_muni_candidates.csv         (市町村別の病害データ件数ランキング)
    16_{muniCd}_{city}_binary.png  (二値病害・フラグの被害あり率)
    16_{muniCd}_{city}_ordinal.png (順序病害の箱ひげ図)
    16_{muniCd}_{city}_stats.csv   (統計サマリ)
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 定数・パス
# =============================================================================
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB    = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEOCODE_CSV = os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR     = os.path.join(_ROOT, 'outputs', 'disease_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_MUNI    = None      # 例: 20202 で手動指定。None なら自動選定
TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33
MIN_PER_GROUP  = 3

HARM_BINARY_COLS    = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN = {'sick': [3], 'bug': [4]}   # 不明センチネル → NaN
HARM_ORDINAL_COLS   = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS      = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_CORE_COLS      = HARM_BINARY_COLS + HARM_ORDINAL_COLS   # 「揃っている」の判定対象
HARM_ALL_DB_COLS    = HARM_CORE_COLS + HARM_FLAG_COLS

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'

ORDINAL_LABELS = {
    'unripen': '青立ち\n(1=無〜4=多)',
    'lay':     '倒伏\n(1=無〜4=多)',
    'loss':    '収穫ロス\n(1=無〜4=多)',
}

# =============================================================================
# データ読み込み
# =============================================================================
print('=' * 60)
print('  データ読み込み')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)
quest = pd.read_sql("""
    SELECT field_id, year, yield
    FROM Questionaire
    WHERE yield IS NOT NULL AND field_id IS NOT NULL
      AND year BETWEEN 2015 AND 2018
""", conn)
harm = pd.read_sql(f"""
    SELECT field_id, year, {', '.join(HARM_ALL_DB_COLS)}
    FROM harm
    WHERE field_id IS NOT NULL
""", conn)
conn.close()

quest['field_id'] = quest['field_id'].astype(int)
quest['year']     = quest['year'].astype(int)
harm['field_id']  = harm['field_id'].astype(int)
harm['year']      = harm['year'].astype(int)

for c in HARM_CORE_COLS:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')
for c, unk in HARM_BINARY_UNKNOWN.items():
    harm[c] = harm[c].where(~harm[c].isin(unk), other=np.nan)
for c in HARM_FLAG_COLS:
    harm[c] = harm[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm[c] = pd.to_numeric(harm[c], errors='coerce')

agg = {c: 'mean' for c in HARM_CORE_COLS}
agg.update({c: 'max' for c in HARM_FLAG_COLS})
harm = harm.groupby(['field_id', 'year'])[HARM_ALL_DB_COLS].agg(agg).reset_index()

geo = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo['muniCd']   = pd.to_numeric(geo['muniCd'], errors='coerce').astype('Int64')
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

df = quest.merge(harm, on=['field_id', 'year'], how='left').merge(geo, on='field_id', how='left')
df = df.dropna(subset=['muniCd']).copy()
df['muniCd'] = df['muniCd'].astype(int)

# 病害データの充足度
df['n_harm_core'] = df[HARM_CORE_COLS].notna().sum(axis=1)
df['harm_any']    = df['n_harm_core'] > 0
df['harm_full']   = df['n_harm_core'] == len(HARM_CORE_COLS)

# =============================================================================
# 対象市町村の選定
# =============================================================================
cand = (df.groupby(['muniCd', 'city'])
          .agg(n_yield=('yield', 'size'),
               n_harm_any=('harm_any', 'sum'),
               n_harm_full=('harm_full', 'sum'))
          .reset_index()
          .sort_values(['n_harm_full', 'n_harm_any'], ascending=False)
          .reset_index(drop=True))
cand.to_csv(os.path.join(OUT_DIR, '16_muni_candidates.csv'), index=False, encoding='utf-8-sig')
print('\n  市町村別 病害データ件数（上位10）:')
print(cand.head(10).to_string(index=False))

if TARGET_MUNI is None:
    muni_cd = int(cand.loc[0, 'muniCd'])
    city    = str(cand.loc[0, 'city'])
else:
    muni_cd = TARGET_MUNI
    city    = str(cand.loc[cand['muniCd'] == muni_cd, 'city'].iloc[0])
print(f'\n  ★ 対象: {city} ({muni_cd})')

mdf = df[(df['muniCd'] == muni_cd) & df['harm_any']].copy().reset_index(drop=True)
print(f'  病害データあり: {len(mdf)} 件')

# =============================================================================
# High / Low 定義（年ごと、市町村内）
# =============================================================================
mdf['group'] = 'Mid'
for yr in TARGET_YEARS:
    m = mdf['year'] == yr
    sub = mdf[m]
    if len(sub) < 6:
        continue
    lo = sub['yield'].quantile(GROUP_QUANTILE)
    hi = sub['yield'].quantile(1 - GROUP_QUANTILE)
    mdf.loc[m & (mdf['yield'] <= lo), 'group'] = 'Low'
    mdf.loc[m & (mdf['yield'] >= hi), 'group'] = 'High'

print('\n  年別グループ件数:')
print(mdf.groupby(['year', 'group']).size().unstack(fill_value=0).to_string())

df_h = mdf[mdf['group'] == 'High']
df_l = mdf[mdf['group'] == 'Low']
n_h, n_l = len(df_h), len(df_l)
print(f'\n  High={n_h} / Low={n_l}')


# =============================================================================
# 統計ユーティリティ
# =============================================================================
def sig_mark(p):
    if p is None or np.isnan(p):
        return 'n.a.'
    return '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')


def test_binary(h_bin, l_bin):
    """h_bin/l_bin: bool Series（欠損除外済み）。Fisher + χ²。"""
    table = np.array([[int(h_bin.sum()), int((~h_bin).sum())],
                      [int(l_bin.sum()), int((~l_bin).sum())]])
    if len(h_bin) < MIN_PER_GROUP or len(l_bin) < MIN_PER_GROUP:
        return table, np.nan, np.nan
    try:
        _, p_f = fisher_exact(table)
    except Exception:
        p_f = np.nan
    try:
        _, p_c, _, _ = chi2_contingency(table)
    except Exception:
        p_c = np.nan
    return table, p_f, p_c


def annotate(ax, lines, sig):
    color = 'red' if sig == '**' else ('orange' if sig == '*' else 'gray')
    ax.text(0.97, 0.97, '\n'.join(lines), ha='right', va='top', transform=ax.transAxes,
            fontsize=7, color=color,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, lw=0.8))


rows = []

# =============================================================================
# 図1: 二値病害 + 気象フラグ
# =============================================================================
bin_cols = HARM_BINARY_COLS + HARM_FLAG_COLS
fig1, axes1 = plt.subplots(2, 5, figsize=(18, 7.5))
axes1 = axes1.flatten()
for i, c in enumerate(bin_cols):
    ax = axes1[i]
    h_v = df_h[c].dropna()
    l_v = df_l[c].dropna()
    if c in HARM_BINARY_COLS:
        h_bin, l_bin = (h_v >= 2), (l_v >= 2)
        ctype = 'binary_harm'
    else:
        h_bin, l_bin = (h_v == 1), (l_v == 1)
        ctype = 'flag_harm'
    table, p_f, p_c = test_binary(h_bin, l_bin)
    h_rate = h_bin.mean() * 100 if len(h_bin) else np.nan
    l_rate = l_bin.mean() * 100 if len(l_bin) else np.nan
    sig = sig_mark(p_f)

    ax.bar(['High', 'Low'], [h_rate, l_rate], color=[COL_HIGH, COL_LOW], alpha=0.8, width=0.5)
    ax.set_title(c, fontsize=9)
    ax.set_ylabel('被害あり率 (%)', fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', ls='--', alpha=0.5)
    annotate(ax, [f'Fisher: {sig} p={p_f:.3f}' if not np.isnan(p_f) else 'n.a.',
                  f'χ²: p={p_c:.3f}' if not np.isnan(p_c) else 'χ²: n.a.',
                  f'High {table[0][0]}/{len(h_v)} Low {table[1][0]}/{len(l_v)}'], sig)
    rows.append({'variable': c, 'type': ctype,
                 'high_n_valid': len(h_v), 'low_n_valid': len(l_v),
                 'high_n_damage': int(table[0][0]), 'low_n_damage': int(table[1][0]),
                 'high_damage_rate_pct': round(h_rate, 1) if not np.isnan(h_rate) else np.nan,
                 'low_damage_rate_pct':  round(l_rate, 1) if not np.isnan(l_rate) else np.nan,
                 'fisher_p': p_f, 'sig_fisher': sig, 'chi2_p': p_c, 'sig_chi2': sig_mark(p_c)})
for ax_off in axes1[len(bin_cols):]:
    ax_off.set_visible(False)
fig1.suptitle(f'{city}（{muni_cd}） 病害・気象フラグ  High n={n_h} / Low n={n_l}', fontsize=12)
plt.tight_layout()
tag = f'16_{muni_cd}_{city}'
fig1.savefig(os.path.join(OUT_DIR, f'{tag}_binary.png'), dpi=180, bbox_inches='tight')
plt.close()

# =============================================================================
# 図2: 順序病害（箱ひげ + MWU + 被害あり/なしFisher・χ²）
# =============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
for i, c in enumerate(HARM_ORDINAL_COLS):
    ax = axes2[i]
    h_v = df_h[c].dropna()
    l_v = df_l[c].dropna()
    try:
        _, p_mwu = mannwhitneyu(h_v, l_v, alternative='two-sided')
    except Exception:
        p_mwu = np.nan
    table, p_f, p_c = test_binary(h_v >= 2, l_v >= 2)
    h_rate = (h_v >= 2).mean() * 100 if len(h_v) else np.nan
    l_rate = (l_v >= 2).mean() * 100 if len(l_v) else np.nan

    if len(h_v) and len(l_v):
        bp = ax.boxplot([h_v, l_v], labels=['High', 'Low'], patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        bp['boxes'][0].set_facecolor(COL_HIGH); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(COL_LOW);  bp['boxes'][1].set_alpha(0.6)
        for k, v in enumerate([h_v, l_v], start=1):
            ax.scatter(np.random.default_rng(0).normal(k, 0.04, len(v)), v,
                       color='k', s=8, alpha=0.4, zorder=3)
    ax.set_title(ORDINAL_LABELS[c], fontsize=10)
    ax.set_ylabel('スコア（1〜4）', fontsize=8)
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(axis='y', ls='--', alpha=0.5)
    annotate(ax, [f'MWU: {sig_mark(p_mwu)} p={p_mwu:.3f}' if not np.isnan(p_mwu) else 'MWU: n.a.',
                  f'Fisher(≥2): p={p_f:.3f}' if not np.isnan(p_f) else 'Fisher: n.a.',
                  f'χ²(≥2): p={p_c:.3f}' if not np.isnan(p_c) else 'χ²: n.a.',
                  f'被害あり High {h_rate:.0f}% / Low {l_rate:.0f}%',
                  f'n: High {len(h_v)} / Low {len(l_v)}'], sig_mark(p_mwu))
    rows.append({'variable': c, 'type': 'ordinal_harm',
                 'high_n_valid': len(h_v), 'low_n_valid': len(l_v),
                 'high_median': float(h_v.median()) if len(h_v) else np.nan,
                 'low_median':  float(l_v.median()) if len(l_v) else np.nan,
                 'mwu_p': p_mwu, 'sig_mwu': sig_mark(p_mwu),
                 'high_n_damage': int(table[0][0]), 'low_n_damage': int(table[1][0]),
                 'high_damage_rate_pct': round(h_rate, 1) if not np.isnan(h_rate) else np.nan,
                 'low_damage_rate_pct':  round(l_rate, 1) if not np.isnan(l_rate) else np.nan,
                 'fisher_p': p_f, 'sig_fisher': sig_mark(p_f),
                 'chi2_p': p_c, 'sig_chi2': sig_mark(p_c)})
fig2.suptitle(f'{city}（{muni_cd}） 順序病害  High n={n_h} / Low n={n_l}', fontsize=12)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, f'{tag}_ordinal.png'), dpi=180, bbox_inches='tight')
plt.close()

# =============================================================================
# 統計サマリ保存
# =============================================================================
stats = pd.DataFrame(rows)
stats.insert(0, 'city', city)
stats.insert(0, 'muniCd', muni_cd)
stats['n_high'] = n_h
stats['n_low']  = n_l
csv_path = os.path.join(OUT_DIR, f'{tag}_stats.csv')
stats.to_csv(csv_path, index=False, encoding='utf-8-sig')

print('\n  ── 結果 ──')
show = ['variable', 'high_n_valid', 'low_n_valid', 'high_damage_rate_pct',
        'low_damage_rate_pct', 'fisher_p', 'chi2_p', 'mwu_p']
print(stats[[c for c in show if c in stats.columns]].round(4).to_string(index=False))
print(f'\n  保存: {csv_path}')
print('=' * 60)
print('  病害 地域内比較 完了！')
print(f'  出力先: {OUT_DIR}')
print('=' * 60)
