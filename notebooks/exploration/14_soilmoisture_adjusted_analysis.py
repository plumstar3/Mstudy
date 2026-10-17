"""
14_soilmoisture_adjusted_analysis.py
====================================
VWC の多収/低収比較を「補正なし」「地域×年偏差」「年別」で並べて比較する。

[共通]
  単位は圃場×年（観測日ごとではなく圃場を単位に検定＝擬似反復を回避）。
  DAP% を 5% ビン化し、圃場×年×ビンで平均VWCを算出。

[A] 補正なし : 年ごとに収量上位/下位33%、生VWCを比較（11と同じ定義・圃場単位検定）
[B] 偏差     : 収量・VWCとも「市町村×年の平均」を引いた偏差を使用。
               収量偏差の上位/下位33%をHigh/Low、VWC偏差を比較。
               ＋ 連続値 Spearman(収量偏差, VWC偏差) も算出。
               (市町村×年で圃場数が MIN_CELL 未満のセルは除外)
[C] 年別     : 年ごとに [A] と同様に比較

[出力] outputs/soil_analysis/14_adjusted_*.png / .csv
"""
import os, sqlite3, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')
for _fn in ['IPAexGothic', 'IPAGothic', 'MS Gothic', 'Yu Gothic', 'Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB = os.path.join(ROOT, 'data', 'processed', 'FieldData_fieldid.db')
GEO = os.path.join(ROOT, 'outputs', 'reverse_geocode', 'field_Addresses.csv')
OUT = os.path.join(ROOT, 'outputs', 'soil_analysis')
os.makedirs(OUT, exist_ok=True)

Q = 0.33
BIN_W = 5
MIN_N = 5          # 検定に必要な各群の最小圃場数
MIN_CELL = 3       # 偏差計算に必要な 市町村×年 の最小圃場数
BIN_RANGE = (25, 90)
C_H, C_L = '#E74C3C', '#3498DB'

# ── データ ───────────────────────────────────────────────────
conn = sqlite3.connect(DB)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
qd = pd.read_sql('''SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire
    WHERE yield IS NOT NULL AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND field_id IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
conn.close()
sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce'); sm = sm.dropna(subset=['field_id'])
sm['field_id'] = sm['field_id'].astype(int); sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce'); sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])
qd['field_id'] = qd['field_id'].astype(int); qd['year'] = qd['year'].astype(int)
qd['seed_date'] = pd.to_datetime(qd['seed_date']); qd['harvest_date'] = pd.to_datetime(qd['harvest_date'])
qd['gd'] = (qd['harvest_date'] - qd['seed_date']).dt.days
qd = qd[qd['gd'] > 0]
geo = pd.read_csv(GEO, encoding='utf-8-sig'); geo['field_id'] = geo['field_id'].astype(int)
geo = geo[['field_id', 'muniCd']].drop_duplicates('field_id')

d = sm.merge(qd, on=['field_id', 'year']).merge(geo, on='field_id')
d['dap'] = (d['date'] - d['seed_date']).dt.days / d['gd'] * 100
d = d[(d['dap'] >= BIN_RANGE[0]) & (d['dap'] < BIN_RANGE[1])]
d['bin'] = (np.floor(d['dap'] / BIN_W) * BIN_W).astype(int)

fb = d.groupby(['field_id', 'year', 'muniCd', 'bin']).agg(vwc=('VWC', 'mean'), y=('yield', 'first')).reset_index()
bins = sorted(fb['bin'].unique())
print(f'圃場×年: {fb[["field_id","year"]].drop_duplicates().shape[0]} 件, ビン {bins[0]}-{bins[-1]+BIN_W}%')


def hl_groups(df, col):
    """df(圃場×年、1行1圃場)に col の上位/下位Q で High/Low を付与。"""
    lo, hi = df[col].quantile(Q), df[col].quantile(1 - Q)
    return np.where(df[col] >= hi, 'High', np.where(df[col] <= lo, 'Low', 'Mid'))


def compare(df_fb, val_col, label, group_col='grp'):
    rows = []
    for b in bins:
        s = df_fb[df_fb['bin'] == b]
        h, l = s[s[group_col] == 'High'][val_col], s[s[group_col] == 'Low'][val_col]
        if len(h) >= MIN_N and len(l) >= MIN_N:
            p = mannwhitneyu(h, l)[1]
        else:
            p = np.nan
        se = np.sqrt(h.var(ddof=1) / len(h) + l.var(ddof=1) / len(l)) if len(h) > 1 and len(l) > 1 else np.nan
        rows.append(dict(setting=label, bin=b + BIN_W / 2, n_high=len(h), n_low=len(l),
                         diff=h.mean() - l.mean(), se=se, p=p))
    return pd.DataFrame(rows)


results = []

# [A] 補正なし（年ごとに収量グループ、全体プール）
fy = fb[['field_id', 'year', 'y']].drop_duplicates(['field_id', 'year']).copy()
fy['grp'] = 'Mid'
for yr, g in fy.groupby('year'):
    fy.loc[g.index, 'grp'] = hl_groups(g, 'y')
A = fb.merge(fy[['field_id', 'year', 'grp']], on=['field_id', 'year'])
results.append(compare(A, 'vwc', 'A: 補正なし(全体)'))

# [B] 地域×年の偏差
B = fb.copy()
cell_n = B.groupby(['muniCd', 'year'])['field_id'].transform('nunique')
B = B[cell_n >= MIN_CELL].copy()
print(f'[B] 市町村×年 圃場数>={MIN_CELL}: 圃場×年 {B[["field_id","year"]].drop_duplicates().shape[0]} 件, '
      f'セル数 {B[["muniCd","year"]].drop_duplicates().shape[0]}')
fy_b = B[['field_id', 'year', 'muniCd', 'y']].drop_duplicates(['field_id', 'year']).copy()
fy_b['y_dev'] = fy_b['y'] - fy_b.groupby(['muniCd', 'year'])['y'].transform('mean')
fy_b['grp'] = hl_groups(fy_b, 'y_dev')
B = B.merge(fy_b[['field_id', 'year', 'y_dev', 'grp']], on=['field_id', 'year'])
B['v_dev'] = B['vwc'] - B.groupby(['muniCd', 'year', 'bin'])['vwc'].transform('mean')
results.append(compare(B, 'v_dev', 'B: 地域×年偏差'))

# [B'] 連続値 Spearman
sp_rows = []
for b in bins:
    s = B[B['bin'] == b]
    if len(s) >= 15:
        r, p = spearmanr(s['y_dev'], s['v_dev'])
        sp_rows.append(dict(bin=b + BIN_W / 2, n=len(s), rho=r, p=p))
SP = pd.DataFrame(sp_rows)

# [C] 年別
for yr in [2016, 2017, 2018]:
    sub = A[A['year'] == yr]
    results.append(compare(sub, 'vwc', f'C: {yr}年のみ(地域未補正)'))

R = pd.concat(results, ignore_index=True)
R.to_csv(os.path.join(OUT, '14_adjusted_stats.csv'), index=False, encoding='utf-8-sig')
SP.to_csv(os.path.join(OUT, '14_adjusted_spearman.csv'), index=False, encoding='utf-8-sig')

# ── 要約表示 ─────────────────────────────────────────────────
print('\n=== High−Low の VWC差（DAP40〜80%ビンの平均・有意ビン数） ===')
for s, g in R.groupby('setting', sort=False):
    w = g[(g['bin'] >= 40) & (g['bin'] <= 80)]
    nt = int(w['p'].notna().sum()); ns = int((w['p'] < 0.05).sum())
    print(f"{s:24s} 平均差={w['diff'].mean():+.4f}  負のビン={int((w['diff']<0).sum())}/{len(w)}  "
          f"有意(p<.05)={ns}/{nt}  最小p={w['p'].min():.3f}  n(H/L)={int(w['n_high'].median())}/{int(w['n_low'].median())}")
print('\n=== [B] 連続値 Spearman(収量偏差, VWC偏差) ===')
print(SP.round(3).to_string(index=False))
w = SP[(SP['bin'] >= 40) & (SP['bin'] <= 80)]
print(f"DAP40-80 の rho 平均={w['rho'].mean():+.3f}  有意ビン={int((w['p']<0.05).sum())}/{len(w)}")

# ── 図 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(20, 5.5))
cols = {'A: 補正なし(全体)': '#34495E', 'B: 地域×年偏差': '#27AE60'}
for s, c in cols.items():
    g = R[R['setting'] == s]
    ax[0].plot(g['bin'], g['diff'], color=c, lw=2.2, marker='o', ms=4, label=s)
    ax[0].fill_between(g['bin'], g['diff'] - 1.96 * g['se'], g['diff'] + 1.96 * g['se'], color=c, alpha=0.15)
    sig = g[g['p'] < 0.05]
    ax[0].scatter(sig['bin'], sig['diff'], s=90, facecolors='none', edgecolors='k', zorder=5)
ax[0].axhline(0, color='gray', ls='--')
ax[0].set_title('High − Low の VWC差（95%CI、○=p<0.05）\n補正なし vs 地域×年偏差')
ax[0].set_xlabel('DAP%'); ax[0].set_ylabel('VWC差 (High − Low)'); ax[0].legend(); ax[0].grid(ls='--', alpha=.3)

for yr, c in zip([2016, 2017, 2018], ['#E67E22', '#8E44AD', '#16A085']):
    g = R[R['setting'] == f'C: {yr}年のみ(地域未補正)']
    ax[1].plot(g['bin'], g['diff'], color=c, lw=2, marker='o', ms=4, label=f'{yr}年')
    sig = g[g['p'] < 0.05]
    ax[1].scatter(sig['bin'], sig['diff'], s=90, facecolors='none', edgecolors='k', zorder=5)
ax[1].axhline(0, color='gray', ls='--')
ax[1].set_title('年別の High − Low の VWC差（地域は未補正）')
ax[1].set_xlabel('DAP%'); ax[1].legend(); ax[1].grid(ls='--', alpha=.3)

ax[2].bar(SP['bin'], SP['rho'], width=BIN_W * 0.8, color=np.where(SP['p'] < 0.05, '#C0392B', '#95A5A6'))
ax[2].axhline(0, color='gray', ls='--')
ax[2].set_title('[B] Spearman ρ(収量偏差, VWC偏差)（赤=p<0.05）')
ax[2].set_xlabel('DAP%'); ax[2].set_ylabel('ρ'); ax[2].grid(ls='--', alpha=.3)
plt.tight_layout()
p = os.path.join(OUT, '14_adjusted_comparison.png')
plt.savefig(p, dpi=170, bbox_inches='tight'); plt.close()
print('\n図保存:', p)
