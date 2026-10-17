"""
08_yield_intra_analysis.py
===========================
同市町村内の多収 vs 低収圃場比較（地域内偏差分析）。
全体比較は 07_yield_group_analysis.py で実施する。

[グループ定義]
  各市町村 × 年 の中で収量の上位33%をHigh、下位33%をLow群とする。
  市町村内の全年のHigh/Lowをプールして統計検定を行う（年効果はグループ定義で制御）。

[対象市町村]
  NDVI保有圃場の総レコード数が MIN_RECORDS 件以上の市町村。

[解析特徴量]（気象時系列は除外）
  - NDVI 3期間平均（生育初期/中期/後期）: Mann-Whitney U検定
  - 病害二値変数（sick/weed/wet/bug）: Fisher's exact test
  - 病害順序変数（unripen/lay/loss）: MWU + Fisher's exact（≥2を被害あり）
  - 気象フラグ（heavy_rain/drought/typhoon/gale/long_rain/few_solar）: Fisher's exact test
  - 栽培変数（between_lines/between_stocks）: Mann-Whitney U検定
  - 品種（breed）: 記述統計のみ（棒グラフ、検定なし）

[出力]
  outputs/yield_intra_analysis/
    {muniCd}_{city}_ndvi.png
    {muniCd}_{city}_disease_flag.png
    {muniCd}_{city}_ordinal_cultivar.png
    all_stats_intra.csv
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

warnings.filterwarnings('ignore')

# フォント設定
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# =============================================================================
# パス・定数設定
# =============================================================================
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB   = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
NDVI_CSV   = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
GEOCODE_CSV= os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR    = os.path.join(_ROOT, 'outputs', 'yield_intra_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2015, 2016, 2017, 2018]
GROUP_QUANTILE = 0.33   # 上位/下位33%をHigh/Lowとする
MIN_RECORDS    = 12     # この件数以上の市町村を解析対象とする
MIN_PER_GROUP  = 3      # 各群の最低サンプル数（これ未満はテストをskip）

# NDVI生育期の分割（相対生育ステージ: 0〜100%）
NDVI_PERIODS = {
    'early（初期）': (0,  33),
    'mid（中期）':  (33, 67),
    'late（後期）': (67, 100),
}

# 病害変数
HARM_BINARY_COLS   = ['sick', 'weed', 'wet', 'bug']
HARM_BINARY_UNKNOWN= {'sick': [3], 'bug': [4]}   # NaN化するセンチネル値
HARM_ORDINAL_COLS  = ['unripen', 'lay', 'loss']
HARM_FLAG_COLS     = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']
HARM_ALL_DB_COLS   = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS

CULTIVAR_COLS   = ['between_lines', 'between_stocks']
BREED_MIN_COUNT = 3

COL_HIGH = '#E74C3C'
COL_LOW  = '#3498DB'

# =============================================================================
# データ読み込み
# =============================================================================
print('='*60)
print('  データ読み込み')
print('='*60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql("""
    SELECT field_id, year, yield, seed_date, harvest_date, breed,
           between_lines, between_stocks
    FROM Questionaire
    WHERE yield IS NOT NULL
      AND seed_date IS NOT NULL
      AND harvest_date IS NOT NULL
      AND year BETWEEN 2015 AND 2018
""", conn)

harm_col_str = ', '.join(HARM_ALL_DB_COLS)
harm_df = pd.read_sql(f"""
    SELECT field_id, year, {harm_col_str}
    FROM harm
    WHERE field_id IS NOT NULL
""", conn)
conn.close()

# 型変換
quest_df['field_id']     = quest_df['field_id'].astype(int)
quest_df['year']         = quest_df['year'].astype(int)
quest_df['seed_date']    = pd.to_datetime(quest_df['seed_date'])
quest_df['harvest_date'] = pd.to_datetime(quest_df['harvest_date'])
quest_df['growing_days'] = (quest_df['harvest_date'] - quest_df['seed_date']).dt.days
for c in CULTIVAR_COLS:
    quest_df[c] = pd.to_numeric(quest_df[c], errors='coerce')

harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)
for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce')
for c in HARM_FLAG_COLS:
    harm_df[c] = harm_df[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce').fillna(0).astype(int)
agg_d = {c: 'mean' for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS}
agg_d.update({c: 'max' for c in HARM_FLAG_COLS})
harm_df = harm_df.groupby(['field_id', 'year'])[HARM_ALL_DB_COLS].agg(agg_d).reset_index()

# 逆ジオコーディング
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df['muniCd']   = pd.to_numeric(geo_df['muniCd'], errors='coerce').astype('Int64')
geo_df = geo_df[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')

# NDVI
print('  NDVI読み込み...')
ndvi_df = pd.read_csv(NDVI_CSV)
ndvi_df['field_id'] = ndvi_df['field_id'].astype(int)
ndvi_df['year']     = ndvi_df['year'].astype(int)
ndvi_fields = set(zip(ndvi_df['field_id'], ndvi_df['year']))
quest_df = quest_df[
    quest_df.apply(lambda r: (int(r['field_id']), int(r['year'])) in ndvi_fields, axis=1)
].copy()

# 病害データ結合
quest_df = quest_df.merge(harm_df, on=['field_id', 'year'], how='left')
quest_df = quest_df.reset_index(drop=True)
print(f'  最終サンプル数: {len(quest_df)} 件')

# =============================================================================
# NDVI 3期間平均の計算
# =============================================================================
print('  NDVI 3期間平均を計算...')

ndvi_period_rows = []
for _, frow in tqdm(quest_df.iterrows(), total=len(quest_df), desc='NDVI期間平均'):
    fid   = int(frow['field_id'])
    year  = int(frow['year'])
    gdays = frow['growing_days']
    if gdays <= 0:
        continue
    nf = ndvi_df[(ndvi_df['field_id'] == fid) & (ndvi_df['year'] == year)].copy()
    if len(nf) < 2:
        continue
    rel = nf['DAP'].values / gdays * 100
    vals = nf['ndvi_smooth'].values.astype(float)
    row = {'field_id': fid, 'year': year}
    for label, (lo, hi) in NDVI_PERIODS.items():
        mask = (rel >= lo) & (rel < hi)
        row[f'ndvi_{label}'] = float(np.nanmean(vals[mask])) if mask.sum() >= 1 else np.nan
    ndvi_period_rows.append(row)

ndvi_period_df = pd.DataFrame(ndvi_period_rows)
quest_df = quest_df.merge(ndvi_period_df, on=['field_id', 'year'], how='left')
NDVI_PERIOD_COLS = [f'ndvi_{lbl}' for lbl in NDVI_PERIODS.keys()]

# =============================================================================
# 対象市町村の選定（NDVI保有レコード合計 ≥ MIN_RECORDS）
# =============================================================================
muni_counts = quest_df.groupby(['muniCd', 'city']).size().reset_index(name='n_records')
target_munis = muni_counts[muni_counts['n_records'] >= MIN_RECORDS].sort_values(
    'n_records', ascending=False).reset_index(drop=True)
print(f'\n  解析対象市町村: {len(target_munis)} 市町村（計 >={MIN_RECORDS}件）')
print(target_munis.to_string(index=False))

# =============================================================================
# 統計ユーティリティ
# =============================================================================
def safe_mwu(h_vals, l_vals):
    """Mann-Whitney U検定。最低MIN_PER_GROUP件なければNaN返却。"""
    h = h_vals.dropna() if hasattr(h_vals, 'dropna') else pd.Series(h_vals).dropna()
    l = l_vals.dropna() if hasattr(l_vals, 'dropna') else pd.Series(l_vals).dropna()
    if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP:
        return np.nan, 'n.a.'
    try:
        _, p = mannwhitneyu(h, l, alternative='two-sided')
        sig  = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        return p, sig
    except Exception:
        return np.nan, 'n.a.'


def safe_fisher(h_series, l_series, threshold=2):
    """Fisher's exact test（≥threshold を被害ありとして二値化）。"""
    h = h_series.dropna() if hasattr(h_series, 'dropna') else pd.Series(h_series).dropna()
    l = l_series.dropna() if hasattr(l_series, 'dropna') else pd.Series(l_series).dropna()
    if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP:
        return np.nan, 'n.a.', np.nan, np.nan
    h_bin = (h >= threshold).astype(int)
    l_bin = (l >= threshold).astype(int)
    h_rate = h_bin.mean() * 100
    l_rate = l_bin.mean() * 100
    table = np.array([
        [int(h_bin.sum()), int((~h_bin.astype(bool)).sum())],
        [int(l_bin.sum()), int((~l_bin.astype(bool)).sum())],
    ])
    try:
        _, p = fisher_exact(table)
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
    except Exception:
        p, sig = np.nan, 'n.a.'
    return p, sig, h_rate, l_rate


def safe_fisher_flag(h_flag, l_flag):
    """フラグ変数（0/1）に対するFisher's exact test。"""
    h = pd.Series(h_flag).dropna()
    l = pd.Series(l_flag).dropna()
    if len(h) < MIN_PER_GROUP or len(l) < MIN_PER_GROUP:
        return np.nan, 'n.a.', np.nan, np.nan
    h_rate = (h == 1).mean() * 100
    l_rate = (l == 1).mean() * 100
    table = np.array([
        [int((h == 1).sum()), int((h != 1).sum())],
        [int((l == 1).sum()), int((l != 1).sum())],
    ])
    try:
        _, p = fisher_exact(table)
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
    except Exception:
        p, sig = np.nan, 'n.a.'
    return p, sig, h_rate, l_rate


def annotate(ax, p, sig, extra=''):
    """axの右上にp値と有意性を注記。"""
    color = 'red' if sig == '**' else ('orange' if sig == '*' else 'gray')
    txt = f'{sig}  p={p:.3f}' if not np.isnan(p) else 'n.a.'
    if extra:
        txt += f'\n{extra}'
    ax.text(0.97, 0.97, txt, ha='right', va='top', transform=ax.transAxes,
            fontsize=7.5, color=color,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, lw=0.8))


# =============================================================================
# 市町村ごとの分析
# =============================================================================
all_stats_rows = []  # 全市町村の統計結果を蓄積

for _, mrow in tqdm(target_munis.iterrows(), total=len(target_munis), desc='市町村分析'):
    muniCd   = int(mrow['muniCd'])
    city     = str(mrow['city'])
    city_safe = city.replace('/', '_').replace('\\', '_')  # ファイル名用

    # ── この市町村のデータ ────────────────────────────────────────
    mdf = quest_df[quest_df['muniCd'] == muniCd].copy()

    # ── 年ごとにHigh/Lowを定義し、全年をプール ───────────────────
    mdf['group'] = 'Mid'
    for yr in TARGET_YEARS:
        yr_mask = mdf['year'] == yr
        yr_data = mdf[yr_mask]
        if len(yr_data) < 6:    # 各群最低2件確保できないなら年をスキップ
            continue
        lo_thr = yr_data['yield'].quantile(GROUP_QUANTILE)
        hi_thr = yr_data['yield'].quantile(1 - GROUP_QUANTILE)
        mdf.loc[yr_mask & (mdf['yield'] <= lo_thr), 'group'] = 'Low'
        mdf.loc[yr_mask & (mdf['yield'] >= hi_thr), 'group'] = 'High'

    df_h = mdf[mdf['group'] == 'High']
    df_l = mdf[mdf['group'] == 'Low']
    n_h  = len(df_h)
    n_l  = len(df_l)

    if n_h < MIN_PER_GROUP or n_l < MIN_PER_GROUP:
        print(f'  [{muniCd}] {city}: High={n_h} / Low={n_l} → スキップ（標本不足）')
        continue

    print(f'\n  [{muniCd}] {city}: High={n_h} / Low={n_l}')
    tag = f'{muniCd}_{city_safe}'

    # ==============================================================
    # 図1: NDVI 3期間（棒グラフ + MWU）
    # ==============================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(11, 4))
    for i, col in enumerate(NDVI_PERIOD_COLS):
        ax = axes1[i]
        lbl = list(NDVI_PERIODS.keys())[i]
        h_v = df_h[col].dropna()
        l_v = df_l[col].dropna()
        h_mean = float(h_v.mean()) if len(h_v) > 0 else np.nan
        l_mean = float(l_v.mean()) if len(l_v) > 0 else np.nan
        h_se   = float(h_v.sem())  if len(h_v) > 1 else 0
        l_se   = float(l_v.sem())  if len(l_v) > 1 else 0

        bars = ax.bar(['High', 'Low'], [h_mean, l_mean],
                      color=[COL_HIGH, COL_LOW], alpha=0.8, width=0.5,
                      yerr=[h_se, l_se], capsize=5, error_kw={'lw': 1.5})
        ax.set_title(f'NDVI {lbl}', fontsize=9)
        ax.set_ylabel('NDVI 平均', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', ls='--', alpha=0.5)

        p_mwu, sig_mwu = safe_mwu(h_v, l_v)
        annotate(ax, p_mwu, sig_mwu,
                 extra=f'High n={len(h_v)} / Low n={len(l_v)}')

        all_stats_rows.append({
            'muniCd': muniCd, 'city': city,
            'variable': col, 'type': 'ndvi_period',
            'n_high': n_h, 'n_low': n_l,
            'high_mean': round(h_mean, 4) if not np.isnan(h_mean) else np.nan,
            'low_mean':  round(l_mean, 4) if not np.isnan(l_mean) else np.nan,
            'mwu_p': p_mwu, 'sig': sig_mwu,
        })

    fig1.suptitle(f'{city}（{muniCd}）  NDVI 3期間比較\nHigh n={n_h} / Low n={n_l}', fontsize=11)
    plt.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, f'{tag}_ndvi.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ==============================================================
    # 図2: 病害二値変数 + フラグ + 品種（棒グラフ + Fisher's exact）
    # ==============================================================
    all_binary = HARM_BINARY_COLS + HARM_FLAG_COLS  # 4 + 6 = 10
    n_ax2 = len(all_binary) + 1   # +1 for breed
    fig2, axes2 = plt.subplots(2, 6, figsize=(22, 8))
    axes2 = axes2.flatten()

    for i, c in enumerate(all_binary):
        ax = axes2[i]
        if c in HARM_BINARY_COLS:
            # 不明センチネルをNaN化してから二値化
            h_raw = df_h[c].copy()
            l_raw = df_l[c].copy()
            for uval in HARM_BINARY_UNKNOWN.get(c, []):
                h_raw = h_raw.where(h_raw != uval, other=np.nan)
                l_raw = l_raw.where(l_raw != uval, other=np.nan)
            p, sig, h_rate, l_rate = safe_fisher(h_raw, l_raw, threshold=2)
        else:  # フラグ
            p, sig, h_rate, l_rate = safe_fisher_flag(df_h[c], df_l[c])

        ax.bar(['High', 'Low'], [h_rate, l_rate],
               color=[COL_HIGH, COL_LOW], alpha=0.8, width=0.5)
        ax.set_title(c, fontsize=8)
        ax.set_ylabel('被害あり率 (%)', fontsize=7)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', ls='--', alpha=0.5)
        annotate(ax, p, sig)

        all_stats_rows.append({
            'muniCd': muniCd, 'city': city,
            'variable': c,
            'type': 'binary_harm' if c in HARM_BINARY_COLS else 'flag_harm',
            'n_high': n_h, 'n_low': n_l,
            'high_damage_rate_pct': round(h_rate, 1) if not np.isnan(h_rate) else np.nan,
            'low_damage_rate_pct':  round(l_rate, 1) if not np.isnan(l_rate) else np.nan,
            'fisher_p': p, 'sig': sig,
        })

    # 品種（記述統計のみ）
    ax = axes2[len(all_binary)]
    all_b  = pd.concat([df_h[['breed']], df_l[['breed']]])
    all_b['breed'] = all_b['breed'].fillna('不明')
    rare   = all_b['breed'].value_counts()
    rare   = rare[rare < BREED_MIN_COUNT].index
    h_b    = df_h['breed'].fillna('不明').replace(rare, 'その他')
    l_b    = df_l['breed'].fillna('不明').replace(rare, 'その他')
    breeds = sorted(set(h_b.unique()) | set(l_b.unique()), key=str)
    h_cnt  = pd.Series(h_b.value_counts()).reindex(breeds, fill_value=0)
    l_cnt  = pd.Series(l_b.value_counts()).reindex(breeds, fill_value=0)
    x = np.arange(len(breeds)); w = 0.35
    ax.bar(x - w/2, h_cnt / h_cnt.sum() * 100, w, color=COL_HIGH, alpha=0.8, label='High')
    ax.bar(x + w/2, l_cnt / l_cnt.sum() * 100, w, color=COL_LOW,  alpha=0.8, label='Low')
    ax.set_xticks(x); ax.set_xticklabels(breeds, rotation=45, ha='right', fontsize=6)
    ax.set_title('breed（品種）', fontsize=8); ax.set_ylabel('割合 (%)', fontsize=7)
    ax.legend(fontsize=6); ax.grid(axis='y', ls='--', alpha=0.5)
    ax.text(0.97, 0.97, '（検定なし）', ha='right', va='top',
            transform=ax.transAxes, fontsize=7, color='gray')

    for ax_off in axes2[len(all_binary)+1:]:
        ax_off.set_visible(False)

    fig2.suptitle(f'{city}（{muniCd}）  病害・気象フラグ・品種\nHigh n={n_h} / Low n={n_l}  ※Fisher\'s exact test', fontsize=10)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, f'{tag}_disease_flag.png'), dpi=180, bbox_inches='tight')
    plt.close()

    # ==============================================================
    # 図3: 順序スコア変数 + 栽培変数（箱ひげ図 + MWU + Fisher's）
    # ==============================================================
    ORDINAL_LABELS = {
        'unripen': '青立ち\n(1=無〜4=多)',
        'lay':     '倒伏\n(1=無〜4=多)',
        'loss':    '収穫ロス\n(1=無〜4=多)',
    }
    CULTIVAR_LABELS = {
        'between_lines': '条間 (cm)',
        'between_stocks':'株間 (cm)',
    }
    all_box_cols = HARM_ORDINAL_COLS + CULTIVAR_COLS  # 3 + 2 = 5
    fig3, axes3 = plt.subplots(1, len(all_box_cols), figsize=(15, 5))

    for i, c in enumerate(all_box_cols):
        ax = axes3[i]
        h_v = df_h[c].dropna()
        l_v = df_l[c].dropna()

        # 箱ひげ図
        if len(h_v) >= 1 and len(l_v) >= 1:
            bp = ax.boxplot([h_v, l_v], labels=['High', 'Low'],
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', lw=2))
            bp['boxes'][0].set_facecolor(COL_HIGH); bp['boxes'][0].set_alpha(0.6)
            bp['boxes'][1].set_facecolor(COL_LOW);  bp['boxes'][1].set_alpha(0.6)
        else:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)

        is_ordinal = c in HARM_ORDINAL_COLS
        ax.set_title(ORDINAL_LABELS.get(c, CULTIVAR_LABELS.get(c, c)), fontsize=9)
        ax.set_ylabel('スコア（1〜4）' if is_ordinal else '値', fontsize=8)
        if is_ordinal:
            ax.set_yticks([1, 2, 3, 4])
        ax.grid(axis='y', ls='--', alpha=0.5)

        # MWU
        p_mwu, sig_mwu = safe_mwu(h_v, l_v)

        if is_ordinal:
            # Fisher's exact（≥2を被害あり）
            p_f, sig_f, h_rate, l_rate = safe_fisher(df_h[c], df_l[c], threshold=2)
            annotate(ax, p_mwu, sig_mwu,
                     extra=f'MWU\nχ²≥2: {sig_f} p={p_f:.3f}\nHigh {h_rate:.0f}%/Low {l_rate:.0f}%'
                     if not np.isnan(p_f) else 'MWU')
            all_stats_rows.append({
                'muniCd': muniCd, 'city': city,
                'variable': c, 'type': 'ordinal_harm',
                'n_high': n_h, 'n_low': n_l,
                'high_median': round(float(h_v.median()), 2) if len(h_v) > 0 else np.nan,
                'low_median':  round(float(l_v.median()), 2) if len(l_v) > 0 else np.nan,
                'mwu_p': p_mwu, 'sig': sig_mwu,
                'fisher_p': p_f, 'sig_fisher': sig_f,
                'high_damage_rate_pct': round(h_rate, 1) if not np.isnan(h_rate) else np.nan,
                'low_damage_rate_pct':  round(l_rate, 1) if not np.isnan(l_rate) else np.nan,
            })
        else:
            annotate(ax, p_mwu, sig_mwu,
                     extra=f'High中央値={h_v.median():.1f}\nLow中央値={l_v.median():.1f}'
                     if len(h_v) > 0 and len(l_v) > 0 else '')
            all_stats_rows.append({
                'muniCd': muniCd, 'city': city,
                'variable': c, 'type': 'cultivar_scalar',
                'n_high': n_h, 'n_low': n_l,
                'high_median': round(float(h_v.median()), 1) if len(h_v) > 0 else np.nan,
                'low_median':  round(float(l_v.median()), 1) if len(l_v) > 0 else np.nan,
                'mwu_p': p_mwu, 'sig': sig_mwu,
            })

    fig3.suptitle(f'{city}（{muniCd}）  順序スコア変数 + 栽培変数\nHigh n={n_h} / Low n={n_l}  ※MWU + Fisher\'s exact (≥2)', fontsize=10)
    plt.tight_layout()
    fig3.savefig(os.path.join(OUT_DIR, f'{tag}_ordinal_cultivar.png'), dpi=180, bbox_inches='tight')
    plt.close()

    print(f'    → 保存完了: {tag}_*.png')


# =============================================================================
# 全市町村まとめCSV
# =============================================================================
all_stats_df = pd.DataFrame(all_stats_rows)
csv_path = os.path.join(OUT_DIR, 'all_stats_intra.csv')
all_stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n  全市町村統計サマリ保存: {csv_path}')

# 有意変数のサマリ表示
print('\n  ★ 有意差あり（p<0.05）の変数一覧:')
sig_df = all_stats_df[all_stats_df['sig'].isin(['*', '**'])].copy()
for _, r in sig_df.iterrows():
    p_col = 'mwu_p' if r['type'] in ['ndvi_period', 'ordinal_harm', 'cultivar_scalar'] else 'fisher_p'
    p_val = r.get(p_col, np.nan)
    print(f'    {r["city"]:12s} | {r["variable"]:20s} | {r["sig"]:3s} | p={p_val:.4f}')

print()
print('='*60)
print('  同市町村内偏差比較 分析完了！')
print(f'  出力先: {OUT_DIR}')
print('='*60)
