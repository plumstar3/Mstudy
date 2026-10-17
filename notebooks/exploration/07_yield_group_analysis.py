"""
07_yield_group_analysis.py
============================
多収圃場 vs 低収圃場 の比較分析スクリプト（全体比較）。
同市町村内の地域内比較は 08_yield_intra_analysis.py で実施する。

[分析内容]
  A. 連続変数（気象・NDVI）の時系列比較
     - 播種〜収穫を相対生育ステージ（0〜100%）に正規化して揃える
     - 年別z-scoreで年の効果を除去
     - 平均値 ± 95%信頼区間を High/Low グループで重ね描き
     - 各時刻点でMann-Whitney U検定（FDR補正）→ 有意区間を帯で表示

  B. カテゴリカル変数（病害5種・品種）
     - χ²検定 + Cramér's V
     - High/Low グループの出現比率棒グラフ

  C. 栽培変数（条間・株間）
     - Mann-Whitney U検定（スカラー値の分布比較）

[入力]
  - data/processed/FieldData_fieldid.db   (収量・播種日・収穫日・品種)
  - data/processed/weather_database_fieldid.db (日次気象)
  - data/processed/ndvi_daily_smooth.csv  (平滑化済み日次NDVI)
  - outputs/reverse_geocode/field_Addresses.csv (muniCd)

[出力]
  outputs/yield_group_analysis/
    01_weather_ts_global.png
    02_ndvi_ts_global.png
    03_disease_breed_global.png
    04_stats_table_global.csv
"""

import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── フォント設定 ───────────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# =============================================================================
# パス設定
# =============================================================================
_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB   = os.path.join(_ROOT, 'data', 'processed', 'FieldData_fieldid.db')
WEATHER_DB = os.path.join(_ROOT, 'data', 'processed', 'weather_database_fieldid.db')
NDVI_CSV   = os.path.join(_ROOT, 'data', 'processed', 'ndvi_daily_smooth.csv')
GEOCODE_CSV= os.path.join(_ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT_DIR    = os.path.join(_ROOT, 'outputs', 'yield_group_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEARS   = [2015, 2016, 2017, 2018]
N_RESAMPLE     = 100   # 相対生育ステージの点数
GROUP_QUANTILE = 0.33  # 上下33%をHigh/Lowとする

# 気象変数（ユーザー指定の優先順位順）
WEATHER_COLS = ['APCPRA', 'TMP_mea', 'WIND', 'GSR', 'RH', 'TMP_max', 'TMP_min', 'SSD', 'SWE']
WEATHER_LABELS = {
    'APCPRA': '降水量 (mm)',   'TMP_mea': '日平均気温 (℃)',  'WIND': '風速 (m/s)',
    'GSR':    '全天日射量',     'RH':      '相対湿度 (%)',     'TMP_max': '日最高気温 (℃)',
    'TMP_min':'日最低気温 (℃)', 'SSD':    '日照時間 (h)',     'SWE':  '積雪水当量',
}

# ── 二値化して χ²検定する病害変数 ──────────────────────────────
# sick : 1=なし, 2=あり, 3=不明→NaN
# weed : 1=なし, 2=あり
# wet  : 1=なし, 2=あり
# bug  : 1=目立たなかった, 2=ほ場で目立った, 3=収穫物で目立った, 4=不明→NaN
#        ※2・3をまとめて「被害あり」とし、4はNaN扱い
HARM_BINARY_COLS = ['sick', 'weed', 'wet', 'bug']
# 列ごとの「不明」センチネル値 → NaN に変換してから二値化
HARM_BINARY_UNKNOWN = {'sick': [3], 'bug': [4]}

# ── 順序スコアのまま Mann-Whitney U 検定する病害変数 ───────────
# unripen : 1=無, 2=少, 3=中, 4=多
# lay     : 1=無, 2=少, 3=中, 4=多
# loss    : 1=無, 2=少, 3=中, 4=多
HARM_ORDINAL_COLS = ['unripen', 'lay', 'loss']

# ── True/False フラグ（χ²検定）───────────────────────────────────
# 除外: frost(2件), hail(5件), cold(4件), early_snow(2件) → サンプル少
HARM_FLAG_COLS = ['heavy_rain', 'drought', 'typhoon', 'gale', 'long_rain', 'few_solar']

# DB から読む列 = 全病害列（除外列もqueryには含め、処理後に無視）
HARM_ALL_DB_COLS = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS

# 後方互換用（build_stats_table などで参照）
HARM_COLS = HARM_ALL_DB_COLS

BREED_MIN_COUNT = 5

# =============================================================================
# データ読み込み
# =============================================================================
print('='*60)
print('  データ読み込み')
print('='*60)

# ── 収量・播種日・収穫日・品種・栽培設定 ──────────────────
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

# 病害データ（全列読み込み）
harm_col_str = ', '.join(HARM_ALL_DB_COLS)
harm_df = pd.read_sql(f"""
    SELECT field_id, year, {harm_col_str}
    FROM harm
    WHERE field_id IS NOT NULL
""", conn)
conn.close()

quest_df['field_id']     = quest_df['field_id'].astype(int)
quest_df['year']         = quest_df['year'].astype(int)
quest_df['seed_date']    = pd.to_datetime(quest_df['seed_date'])
quest_df['harvest_date'] = pd.to_datetime(quest_df['harvest_date'])
quest_df['growing_days'] = (quest_df['harvest_date'] - quest_df['seed_date']).dt.days

# 栽培変数を数値型に変換
CULTIVAR_COLS = ['between_lines', 'between_stocks']
CULTIVAR_LABELS = {'between_lines': '条間 (cm)', 'between_stocks': '株間 (cm)'}
for c in CULTIVAR_COLS:
    quest_df[c] = pd.to_numeric(quest_df[c], errors='coerce')

harm_df['field_id'] = harm_df['field_id'].astype(int)
harm_df['year']     = harm_df['year'].astype(int)
# 二値化変数・順序変数 → 数値型
for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS:
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce')
# フラグ変数 → 0/1 整数型
for c in HARM_FLAG_COLS:
    harm_df[c] = harm_df[c].replace({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
    harm_df[c] = pd.to_numeric(harm_df[c], errors='coerce').fillna(0).astype(int)

# 1圃場×年で集計（基本的に重複なし・念のため）
agg_dict = {}
agg_dict.update({c: 'mean' for c in HARM_BINARY_COLS + HARM_ORDINAL_COLS})
agg_dict.update({c: 'max'  for c in HARM_FLAG_COLS})
harm_df = harm_df.groupby(['field_id', 'year'])[HARM_ALL_DB_COLS].agg(agg_dict).reset_index()

print(f'  圃場数（質問票）: {len(quest_df)} 件')

# ── 逆ジオコーディング（muniCd） ──────────────────────────────────
geo_df = pd.read_csv(GEOCODE_CSV, encoding='utf-8-sig')
geo_df['field_id'] = geo_df['field_id'].astype(int)
geo_df['muniCd']   = pd.to_numeric(geo_df['muniCd'], errors='coerce').astype('Int64')
geo_df = geo_df[['field_id', 'muniCd']].drop_duplicates('field_id')
quest_df = quest_df.merge(geo_df, on='field_id', how='left')

# ── NDVI日次データ ────────────────────────────────────────────────
print('  NDVI日次データ読み込み...')
ndvi_df = pd.read_csv(NDVI_CSV)
ndvi_df['field_id'] = ndvi_df['field_id'].astype(int)
ndvi_df['year']     = ndvi_df['year'].astype(int)
ndvi_fields = set(zip(ndvi_df['field_id'], ndvi_df['year']))
print(f'  NDVI保有圃場×年: {len(ndvi_fields)} 件')

# NDVIデータのある圃場のみを対象とする（ユーザー指示）
quest_df['has_ndvi'] = quest_df.apply(
    lambda r: (int(r['field_id']), int(r['year'])) in ndvi_fields, axis=1
)
quest_df = quest_df[quest_df['has_ndvi']].copy()
print(f'  NDVI保有圃場に絞った後: {len(quest_df)} 件')

# 病害データを結合
quest_df = quest_df.merge(harm_df, on=['field_id', 'year'], how='left')

print(f'  最終サンプル数: {len(quest_df)} 件')

# ── 気象データ読み込み ────────────────────────────────────────────
print('  気象データ読み込み（全フィールド日次）...')
conn_w = sqlite3.connect(WEATHER_DB)
weather_raw = pd.read_sql(f"""
    SELECT field_id, date, {', '.join(WEATHER_COLS)}
    FROM weather_data
    WHERE field_id IN ({','.join(map(str, quest_df['field_id'].unique()))})
""", conn_w)
conn_w.close()

weather_raw['field_id'] = weather_raw['field_id'].astype(int)
weather_raw['date']     = pd.to_datetime(weather_raw['date'])
for c in WEATHER_COLS:
    weather_raw[c] = pd.to_numeric(weather_raw[c], errors='coerce')

print(f'  気象データ行数: {len(weather_raw):,} 行')


# =============================================================================
# 時系列正規化ユーティリティ
# =============================================================================
def normalize_ts_to_relative(field_id, year, seed_date, growing_days,
                              weather_raw, ndvi_df):
    """
    1圃場の気象・NDVI時系列を相対生育ステージ 0〜100% (N_RESAMPLE点) にリサンプル。
    Returns:
        weather_resampled: dict{col: ndarray(N_RESAMPLE)}
        ndvi_resampled:    ndarray(N_RESAMPLE) or None
    """
    relative_pts = np.linspace(0, 100, N_RESAMPLE)

    # 気象
    wf = weather_raw[weather_raw['field_id'] == field_id].copy()
    wf['DAP'] = (wf['date'] - seed_date).dt.days
    wf = wf[(wf['DAP'] >= 0) & (wf['DAP'] <= growing_days)]

    weather_resampled = {}
    if len(wf) >= 2:
        wf_dap = wf['DAP'].values
        relative_wf = wf_dap / growing_days * 100
        for c in WEATHER_COLS:
            vals = wf[c].values.astype(float)
            if np.isnan(vals).all():
                weather_resampled[c] = np.full(N_RESAMPLE, np.nan)
            else:
                # NANを除いてから補間
                mask = ~np.isnan(vals)
                if mask.sum() >= 2:
                    weather_resampled[c] = np.interp(
                        relative_pts, relative_wf[mask], vals[mask])
                else:
                    weather_resampled[c] = np.full(N_RESAMPLE, np.nan)
    else:
        for c in WEATHER_COLS:
            weather_resampled[c] = np.full(N_RESAMPLE, np.nan)

    # NDVI
    nf = ndvi_df[(ndvi_df['field_id'] == field_id) & (ndvi_df['year'] == year)]
    if len(nf) >= 2:
        nf_dap = nf['DAP'].values
        relative_nf = nf_dap / growing_days * 100
        ndvi_vals = nf['ndvi_smooth'].values.astype(float)
        ndvi_resampled = np.interp(relative_pts, relative_nf, ndvi_vals)
    else:
        ndvi_resampled = None

    return weather_resampled, ndvi_resampled


# =============================================================================
# 時系列データ構築（全圃場）
# =============================================================================
print()
print('='*60)
print('  時系列データ構築（相対生育ステージ正規化）')
print('='*60)

weather_ts = {c: [] for c in WEATHER_COLS}  # 各変数の (N圃場 × N_RESAMPLE) リスト
ndvi_ts    = []
fid_list   = []
year_list  = []

for _, row in tqdm(quest_df.iterrows(), total=len(quest_df), desc='時系列正規化'):
    fid          = int(row['field_id'])
    yr           = int(row['year'])
    seed_date    = row['seed_date']
    growing_days = int(row['growing_days'])

    w_res, n_res = normalize_ts_to_relative(
        fid, yr, seed_date, growing_days, weather_raw, ndvi_df)

    for c in WEATHER_COLS:
        weather_ts[c].append(w_res[c])
    ndvi_ts.append(n_res if n_res is not None else np.full(N_RESAMPLE, np.nan))
    fid_list.append(fid)
    year_list.append(yr)

# numpy配列化
for c in WEATHER_COLS:
    weather_ts[c] = np.array(weather_ts[c])  # (N, N_RESAMPLE)
ndvi_ts = np.array(ndvi_ts)                   # (N, N_RESAMPLE)

# 年別z-score正規化
print('  年別z-score正規化...')
year_arr = np.array(year_list)
for c in WEATHER_COLS:
    for yr in TARGET_YEARS:
        mask = year_arr == yr
        if mask.sum() == 0:
            continue
        mu  = np.nanmean(weather_ts[c][mask], axis=0)
        sig = np.nanstd(weather_ts[c][mask], axis=0)
        sig[sig == 0] = 1.0
        weather_ts[c][mask] = (weather_ts[c][mask] - mu) / sig

print('  完了')


# =============================================================================
# グループ定義・統計計算ユーティリティ
# =============================================================================
def define_groups(quest_df, mode='global'):
    """
    mode='global': 絶対収量の上下33%
    mode='intra' : muniCd内偏差（収量 - muniCd平均）の上下33%
    """
    df = quest_df.copy()
    if mode == 'global':
        lo_thr = df['yield'].quantile(GROUP_QUANTILE)
        hi_thr = df['yield'].quantile(1 - GROUP_QUANTILE)
        df['group'] = np.where(df['yield'] <= lo_thr, 'Low',
                      np.where(df['yield'] >= hi_thr, 'High', 'Mid'))
    else:
        muni_mean = df.groupby('muniCd')['yield'].transform('mean')
        df['yield_intra'] = df['yield'] - muni_mean
        lo_thr = df['yield_intra'].quantile(GROUP_QUANTILE)
        hi_thr = df['yield_intra'].quantile(1 - GROUP_QUANTILE)
        df['group'] = np.where(df['yield_intra'] <= lo_thr, 'Low',
                      np.where(df['yield_intra'] >= hi_thr, 'High', 'Mid'))
    return df


def ts_stats(ts_mat, mask_high, mask_low, alpha=0.05):
    """
    各時刻点でMann-Whitney U検定 → FDR補正 → 平均・95%CI を返す。
    """
    n_pts = ts_mat.shape[1]
    mu_h = np.nanmean(ts_mat[mask_high], axis=0)
    mu_l = np.nanmean(ts_mat[mask_low],  axis=0)
    se_h = np.nanstd(ts_mat[mask_high], axis=0) / np.sqrt(mask_high.sum())
    se_l = np.nanstd(ts_mat[mask_low],  axis=0) / np.sqrt(mask_low.sum())
    ci95_h = 1.96 * se_h
    ci95_l = 1.96 * se_l

    pvals = []
    for t in range(n_pts):
        h_vals = ts_mat[mask_high, t]
        l_vals = ts_mat[mask_low,  t]
        h_clean = h_vals[~np.isnan(h_vals)]
        l_clean = l_vals[~np.isnan(l_vals)]
        if len(h_clean) >= 3 and len(l_clean) >= 3:
            _, p = mannwhitneyu(h_clean, l_clean, alternative='two-sided')
        else:
            p = 1.0
        pvals.append(p)

    pvals = np.array(pvals)
    _, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')
    sig_mask = pvals_corr < alpha

    return mu_h, mu_l, ci95_h, ci95_l, sig_mask, pvals_corr


def cramers_v(contingency_table):
    """Cramér's V（χ²の効果量）を計算"""
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    v = np.sqrt(phi2 / min(r - 1, k - 1)) if min(r - 1, k - 1) > 0 else 0
    return float(v)


# =============================================================================
# 描画ユーティリティ
# =============================================================================
RELATIVE_PTS = np.linspace(0, 100, N_RESAMPLE)
COL_HIGH = '#E74C3C'   # 赤
COL_LOW  = '#3498DB'   # 青
COL_SIG  = '#AAAAAA'   # 有意差区間の帯

def plot_weather_grid(ts_dict, mask_high, mask_low, title, out_path,
                      n_high, n_low):
    """気象9変数を3×3グリッドでプロット"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()
    sig_results = {}

    for i, c in enumerate(WEATHER_COLS):
        ax = axes[i]
        mu_h, mu_l, ci_h, ci_l, sig, _ = ts_stats(
            ts_dict[c], mask_high, mask_low)
        sig_results[c] = sig
        print(f'  [有意時刻%] {c}: {np.mean(sig) * 100:.0f}%')

        # 有意差帯
        sig_start = None
        for t in range(N_RESAMPLE):
            if sig[t] and sig_start is None:
                sig_start = t
            elif not sig[t] and sig_start is not None:
                ax.axvspan(RELATIVE_PTS[sig_start], RELATIVE_PTS[t-1],
                           color=COL_SIG, alpha=0.3, zorder=0)
                sig_start = None
        if sig_start is not None:
            ax.axvspan(RELATIVE_PTS[sig_start], RELATIVE_PTS[-1],
                       color=COL_SIG, alpha=0.3, zorder=0)

        # 平均±CI
        ax.plot(RELATIVE_PTS, mu_h, color=COL_HIGH, lw=1.8, label=f'High (n={n_high})')
        ax.fill_between(RELATIVE_PTS, mu_h - ci_h, mu_h + ci_h,
                        color=COL_HIGH, alpha=0.2)
        ax.plot(RELATIVE_PTS, mu_l, color=COL_LOW, lw=1.8, label=f'Low (n={n_low})')
        ax.fill_between(RELATIVE_PTS, mu_l - ci_l, mu_l + ci_l,
                        color=COL_LOW, alpha=0.2)

        ax.set_title(WEATHER_LABELS.get(c, c), fontsize=10)
        ax.set_ylabel('標準化値', fontsize=8)
        if i >= 6:
            ax.set_xlabel('相対生育ステージ (%)', fontsize=8)
        ax.grid(True, ls='--', alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out_path}')
    return sig_results


def plot_ndvi_ts(ndvi_mat, mask_high, mask_low, title, out_path, n_high, n_low):
    """NDVI時系列比較グラフ"""
    mu_h, mu_l, ci_h, ci_l, sig, _ = ts_stats(ndvi_mat, mask_high, mask_low)

    fig, ax = plt.subplots(figsize=(10, 5))
    # 有意差帯
    sig_start = None
    for t in range(N_RESAMPLE):
        if sig[t] and sig_start is None:
            sig_start = t
        elif not sig[t] and sig_start is not None:
            ax.axvspan(RELATIVE_PTS[sig_start], RELATIVE_PTS[t-1],
                       color=COL_SIG, alpha=0.3, zorder=0, label='有意差区間 (FDR<0.05)' if sig_start == list(np.where(sig)[0])[0] else '')
            sig_start = None
    if sig_start is not None:
        ax.axvspan(RELATIVE_PTS[sig_start], RELATIVE_PTS[-1],
                   color=COL_SIG, alpha=0.3, zorder=0)

    ax.plot(RELATIVE_PTS, mu_h, color=COL_HIGH, lw=2, label=f'High yield (n={n_high})')
    ax.fill_between(RELATIVE_PTS, mu_h - ci_h, mu_h + ci_h, color=COL_HIGH, alpha=0.2)
    ax.plot(RELATIVE_PTS, mu_l, color=COL_LOW,  lw=2, label=f'Low yield  (n={n_low})')
    ax.fill_between(RELATIVE_PTS, mu_l - ci_l, mu_l + ci_l, color=COL_LOW, alpha=0.2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('相対生育ステージ (%)', fontsize=11)
    ax.set_ylabel('NDVI（SG平滑化済み）', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out_path}')


def plot_disease_breed(quest_sub, mask_high, mask_low, title, out_path, n_high, n_low):
    """全病害変数 + 品種の比較グラフ + 統計検定。
    出力: (stats_rows, ordinal_stats_rows) のタプル
    """
    df_h = quest_sub[mask_high]
    df_l = quest_sub[mask_low]
    stats_rows = []
    ordinal_stats_rows = []

    # ================================================================
    # 図1: 二値化変数（sick/weed/wet/bug） + フラグ（6種） + 品種
    # ================================================================
    # 4 + 3(順序: ≥2を被害ありとして率表示) + 6 = 13 変数 + 品種 = 14パネル
    all_binary_cols = HARM_BINARY_COLS + HARM_ORDINAL_COLS + HARM_FLAG_COLS
    n_cols_fig1 = len(all_binary_cols) + 1  # +1 for breed
    fig1, axes1 = plt.subplots(2, 7, figsize=(26, 10))
    axes1 = axes1.flatten()

    # ── 二値化変数（χ²検定）──────────────────────────────────────
    def _to_binary_harm(series, col):
        """不明センチネル値をNaNに変換してから 2 以上を被害ありとする。"""
        s = series.copy()
        for uval in HARM_BINARY_UNKNOWN.get(col, []):
            s = s.where(s != uval, other=np.nan)
        return s.fillna(1) >= 2

    for i, c in enumerate(all_binary_cols):
        ax = axes1[i]
        if c in HARM_ORDINAL_COLS:
            # 順序変数: 欠損を除外し、≥2 を被害ありとする（図2のχ²と同じ定義）
            h_bin = (df_h[c].dropna() >= 2)
            l_bin = (df_l[c].dropna() >= 2)
            h_rate = h_bin.mean() * 100
            l_rate = l_bin.mean() * 100
            h_cnt  = [int(h_bin.sum()), int((~h_bin).sum())]
            l_cnt  = [int(l_bin.sum()), int((~l_bin).sum())]
        elif c in HARM_BINARY_COLS:
            h_bin = _to_binary_harm(df_h[c], c)
            l_bin = _to_binary_harm(df_l[c], c)
            h_rate = h_bin.mean() * 100
            l_rate = l_bin.mean() * 100
            h_cnt  = [int(h_bin.sum()), int((~h_bin).sum())]
            l_cnt  = [int(l_bin.sum()), int((~l_bin).sum())]
        else:  # フラグ変数
            h_rate = (df_h[c] == 1).mean() * 100
            l_rate = (df_l[c] == 1).mean() * 100
            h_cnt  = [int((df_h[c] == 1).sum()), int((df_h[c] != 1).sum())]
            l_cnt  = [int((df_l[c] == 1).sum()), int((df_l[c] != 1).sum())]

        table = np.array([h_cnt, l_cnt])
        try:
            _, p, _, _ = chi2_contingency(table)
            v = cramers_v(table)
        except Exception:
            p, v = 1.0, 0.0
        sig_str = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')

        ax.bar(['High', 'Low'], [h_rate, l_rate],
               color=[COL_HIGH, COL_LOW], alpha=0.8, width=0.5)
        ax.set_title(c, fontsize=9)
        ax.set_ylabel('被害あり率 (%)', fontsize=8)
        ax.set_ylim(0, 100)
        ax.text(0.5, 0.92, f"χ²: {sig_str}\nV={v:.2f}, p={p:.3f}",
                ha='center', transform=ax.transAxes, fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        ax.grid(axis='y', ls='--', alpha=0.5)
        if c in HARM_ORDINAL_COLS:
            continue  # 統計値は図2側の ordinal_stats_rows に記録（重複防止）
        cat_type = 'binary_harm' if c in HARM_BINARY_COLS else 'flag_harm'
        stats_rows.append({'variable': c, 'type': cat_type,
                           'chi2_p': p, 'cramers_v': v, 'sig': sig_str})

    # ── 品種（χ²検定）───────────────────────────────────────────
    ax = axes1[len(all_binary_cols)]
    all_breeds = pd.concat([df_h[['breed']], df_l[['breed']]])
    all_breeds['breed'] = all_breeds['breed'].fillna('不明')
    breed_counts = all_breeds['breed'].value_counts()
    rare_breeds  = breed_counts[breed_counts < BREED_MIN_COUNT].index
    df_h_b = df_h['breed'].fillna('不明').replace(rare_breeds, 'その他')
    df_l_b = df_l['breed'].fillna('不明').replace(rare_breeds, 'その他')
    breeds = sorted(set(df_h_b.unique()) | set(df_l_b.unique()), key=str)
    h_cnt_b = pd.Series(df_h_b.value_counts()).reindex(breeds, fill_value=0)
    l_cnt_b = pd.Series(df_l_b.value_counts()).reindex(breeds, fill_value=0)
    x = np.arange(len(breeds)); w = 0.35
    ax.bar(x - w/2, h_cnt_b / h_cnt_b.sum() * 100, w, color=COL_HIGH, alpha=0.8, label=f'High (n={n_high})')
    ax.bar(x + w/2, l_cnt_b / l_cnt_b.sum() * 100, w, color=COL_LOW,  alpha=0.8, label=f'Low (n={n_low})')
    ax.set_xticks(x); ax.set_xticklabels(breeds, rotation=45, ha='right', fontsize=7)
    ax.set_title('品種（breed）', fontsize=9); ax.set_ylabel('構成比 (%)', fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis='y', ls='--', alpha=0.5)
    table_b = np.array([h_cnt_b.values, l_cnt_b.values])
    try:
        _, p_b, _, _ = chi2_contingency(table_b)
        v_b = cramers_v(pd.DataFrame(table_b))
    except Exception:
        p_b, v_b = 1.0, 0.0
    sig_str_b = '**' if p_b < 0.01 else ('*' if p_b < 0.05 else 'ns')
    ax.text(0.5, 0.92, f"χ²: {sig_str_b}\nV={v_b:.2f}, p={p_b:.3f}",
            ha='center', transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    stats_rows.append({'variable': 'breed', 'type': 'categorical',
                       'chi2_p': p_b, 'cramers_v': v_b, 'sig': sig_str_b})
    # 残りの空きaxをオフに
    for ax_off in axes1[len(all_binary_cols)+1:]:
        ax_off.set_visible(False)

    fig1.suptitle(title + '（二値化変数・フラグ・品種）', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out_path}')

    # ================================================================
    # 図2: 順序スコア変数（unripen/lay/loss） → 箱ひげ図 + MWU検定
    # ================================================================
    out_path2 = out_path.replace('.png', '_ordinal.png')
    fig2, axes2 = plt.subplots(1, len(HARM_ORDINAL_COLS), figsize=(12, 5))
    if len(HARM_ORDINAL_COLS) == 1:
        axes2 = [axes2]

    ORDINAL_LABELS = {
        'unripen': '青立ち\n(1=無〜4=多)',
        'lay':     '倒伏\n(1=無〜4=多)',
        'loss':    '収穫ロス\n(1=無〜4=多)',
    }
    for i, c in enumerate(HARM_ORDINAL_COLS):
        ax = axes2[i]
        h_vals = df_h[c].dropna()
        l_vals = df_l[c].dropna()

        # ── Mann-Whitney U検定（スコア分布の比較）──────────────────
        try:
            _, p_mwu = mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        except Exception:
            p_mwu = 1.0
        sig_mwu = '**' if p_mwu < 0.01 else ('*' if p_mwu < 0.05 else 'ns')

        # ── χ²検定（1=被害なし / 2,3,4=被害あり に二値化、欠損は除外）──
        h_bin = (h_vals >= 2)
        l_bin = (l_vals >= 2)
        h_rate = h_bin.mean() * 100
        l_rate = l_bin.mean() * 100
        table_ord = np.array([
            [int(h_bin.sum()), int((~h_bin).sum())],
            [int(l_bin.sum()), int((~l_bin).sum())],
        ])
        try:
            _, p_chi2, _, _ = chi2_contingency(table_ord)
            v_chi2 = cramers_v(table_ord)
        except Exception:
            p_chi2, v_chi2 = 1.0, 0.0
        sig_chi2 = '**' if p_chi2 < 0.01 else ('*' if p_chi2 < 0.05 else 'ns')

        # ── 箱ひげ図 ──────────────────────────────────────────────
        bp = ax.boxplot([h_vals, l_vals], labels=[f'High\n({h_rate:.0f}%)', f'Low\n({l_rate:.0f}%)'],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        bp['boxes'][0].set_facecolor(COL_HIGH); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(COL_LOW);  bp['boxes'][1].set_alpha(0.6)
        ax.set_title(ORDINAL_LABELS.get(c, c), fontsize=10)
        ax.set_ylabel('スコア（1=無〜4=多）', fontsize=8)
        ax.set_yticks([1, 2, 3, 4])
        ax.grid(axis='y', ls='--', alpha=0.5)
        # 注釈：MWU + χ²の両方を表示
        ax.text(0.5, 0.97,
                f'MWU: {sig_mwu} (p={p_mwu:.3f})\nχ²: {sig_chi2} (V={v_chi2:.2f}, p={p_chi2:.3f})\n≥2を被害あり: High {h_rate:.0f}% / Low {l_rate:.0f}%',
                ha='center', va='top', transform=ax.transAxes, fontsize=7.5,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        ordinal_stats_rows.append({
            'variable': c, 'type': 'ordinal_harm',
            'high_median': round(float(h_vals.median()), 2),
            'low_median':  round(float(l_vals.median()), 2),
            'mwu_p': p_mwu, 'sig': sig_mwu,
            'chi2_p': p_chi2, 'cramers_v': v_chi2, 'sig_chi2': sig_chi2,
            'high_n_damage': int(table_ord[0][0]), 'high_n_none': int(table_ord[0][1]),
            'low_n_damage': int(table_ord[1][0]),  'low_n_none': int(table_ord[1][1]),
            'high_damage_rate_pct': round(h_rate, 1),
            'low_damage_rate_pct':  round(l_rate, 1),
        })

    fig2.suptitle(title + '（順序スコア変数：箱ひげ図 + χ²検定）', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  保存: {out_path2}')

    return stats_rows, ordinal_stats_rows


def build_stats_table(sig_dict_weather, ndvi_sig, cat_stats, ordinal_stats=None):
    """全変数の統計結果をまとめたDataFrameを生成"""
    rows = []
    for c, sig in sig_dict_weather.items():
        n_sig = sig.sum()
        rows.append({'variable': c, 'type': 'weather_ts',
                     'n_sig_timepoints': int(n_sig),
                     'sig_timepoints_pct': round(n_sig / N_RESAMPLE * 100, 1)})
    ndvi_n_sig = ndvi_sig.sum()
    rows.append({'variable': 'ndvi_smooth', 'type': 'ndvi_ts',
                 'n_sig_timepoints': int(ndvi_n_sig),
                 'sig_timepoints_pct': round(ndvi_n_sig / N_RESAMPLE * 100, 1)})
    for r in cat_stats:
        rows.append(r)
    if ordinal_stats:
        for r in ordinal_stats:
            rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# 分析実行（全体比較のみ）
# 同市町村内の地域内比較は 08_yield_intra_analysis.py で実施する
# =============================================================================
mode  = 'global'
label = '全体比較'

print()
print('='*60)
print(f'  {label}')
print('='*60)

grouped_df = define_groups(quest_df, mode=mode)
mask_high  = (grouped_df['group'] == 'High').values
mask_low   = (grouped_df['group'] == 'Low').values
n_high, n_low = mask_high.sum(), mask_low.sum()
print(f'  High: {n_high} 件 / Low: {n_low} 件 / Mid（除外）: {(grouped_df["group"]=="Mid").sum()} 件')

title_base = f'多収 vs 低収圃場 — {label}'

# ── 1. 気象時系列グラフ ─────────────────────────────────────────
sig_dict_w = plot_weather_grid(
    weather_ts, mask_high, mask_low,
    title=f'{title_base}（気象時系列 / 年別z-score）',
    out_path=os.path.join(OUT_DIR, '01_weather_ts_global.png'),
    n_high=n_high, n_low=n_low
)

# ── 2. NDVI時系列グラフ ─────────────────────────────────────────
_, _, _, _, ndvi_sig, _ = ts_stats(ndvi_ts, mask_high, mask_low)
plot_ndvi_ts(
    ndvi_ts, mask_high, mask_low,
    title=f'{title_base}（NDVI時系列）',
    out_path=os.path.join(OUT_DIR, '02_ndvi_ts_global.png'),
    n_high=n_high, n_low=n_low
)

# ── 3. 病害・品種棒グラフ ────────────────────────────────────────
cal_stats, ordinal_stats = plot_disease_breed(
    grouped_df, mask_high, mask_low,
    title=f'{title_base}（病害・品種）',
    out_path=os.path.join(OUT_DIR, '03_disease_breed_global.png'),
    n_high=n_high, n_low=n_low
)

# ── 4. 栽培変数（条間・株間）のMann-Whitney U検定 ────────────────
from scipy.stats import mannwhitneyu
cultivar_stats = []
for c in CULTIVAR_COLS:
    h_vals = grouped_df.loc[mask_high, c].dropna()
    l_vals = grouped_df.loc[mask_low,  c].dropna()
    if len(h_vals) >= 3 and len(l_vals) >= 3:
        _, p_mwu = mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        sig_str  = '**' if p_mwu < 0.01 else ('*' if p_mwu < 0.05 else 'ns')
        h_med, l_med = float(h_vals.median()), float(l_vals.median())
    else:
        p_mwu, sig_str = 1.0, 'ns'
        h_med, l_med  = float('nan'), float('nan')
    cultivar_stats.append({
        'variable': c, 'type': 'cultivar_scalar',
        'high_median': round(h_med, 1) if not np.isnan(h_med) else np.nan,
        'low_median':  round(l_med, 1) if not np.isnan(l_med) else np.nan,
        'mwu_p': p_mwu, 'sig': sig_str,
    })
print('  栽培変数（条間・株間）Mann-Whitney U検定:')
for row in cultivar_stats:
    print(f'    {row["variable"]}: High中央値={row["high_median"]} / Low中央値={row["low_median"]} / p={row["mwu_p"]:.4f} ({row["sig"]})')

# ── 5. 統計サマリCSV ─────────────────────────────────────────────
stats_df = build_stats_table(sig_dict_w, ndvi_sig, cal_stats, ordinal_stats=ordinal_stats)
stats_df = pd.concat([stats_df, pd.DataFrame(cultivar_stats)], ignore_index=True)
csv_path = os.path.join(OUT_DIR, '04_stats_table_global.csv')
stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'  保存: {csv_path}')

print()
print('='*60)
print('  全体比較分析完了！')
print(f'  出力先: {OUT_DIR}')
print('='*60)
