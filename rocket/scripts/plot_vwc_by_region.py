"""
同じ地域(muniCd)・同じ年の圃場について、VWC の推移を播種後日数(dap)で重ねてプロットする。
色: 低収=青, 多収=赤, 中間(ラベルなし)=灰 (label_muniyear に基づく)

対象: 地域x年のグループ圃場数が MIN_GROUP 以上のもの
出力: rocket/outputs/vwc_plots/vwc_{muniCd}_{year}.png  と  vwc_overview.png

使い方:
  python plot_vwc_by_region.py            # 横軸 = 播種後日数(dap)  -> outputs/vwc_plots/
  python plot_vwc_by_region.py --x date   # 横軸 = 実時間(date)      -> outputs/vwc_plots_date/
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--x', choices=['dap', 'date'], default='dap')
XMODE = ap.parse_args().x

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(R)
P = os.path.join(R, 'data', 'processed')
OUT_DIR = os.path.join(R, 'outputs', 'vwc_plots' if XMODE == 'dap' else 'vwc_plots_date')
MIN_GROUP = 4

COL = {0: '#1f77b4', 1: '#d62728'}
MID = '#9e9e9e'

st = pd.read_csv(os.path.join(P, 'field_year_labeled.csv'), dtype={'muniCd': str})
ts = pd.read_csv(os.path.join(P, 'soilmoisture_timeseries.csv'), parse_dates=['date'])
addr = pd.read_csv(os.path.join(ROOT, 'outputs', 'reverse_geocode', 'field_addresses.csv'), dtype={'muniCd': str})
names = addr.drop_duplicates('muniCd').set_index('muniCd').apply(lambda r: f"{r['prefecture']}{r['city']}", axis=1)

ts = ts.merge(st[['field_id', 'year', 'muniCd', 'yield', 'label_muniyear']], on=['field_id', 'year'])
groups = st.groupby(['muniCd', 'year']).size().rename('n').reset_index()
groups = groups[groups.n >= MIN_GROUP].sort_values(['n', 'muniCd', 'year'], ascending=[False, True, True])


XCOL = XMODE
XLABEL = '播種後日数 (dap)' if XMODE == 'dap' else '日付 (date)'


def draw(ax, muni, year, annotate=True):
    d = ts[(ts.muniCd == muni) & (ts.year == year)]
    for fid, g in d.groupby('field_id'):
        lab = g['label_muniyear'].iloc[0]
        c = COL.get(lab, MID) if pd.notna(lab) else MID
        z = 3 if pd.notna(lab) else 1
        ax.plot(g[XCOL], g['VWC'], color=c, lw=1.4 if pd.notna(lab) else 1.0,
                alpha=0.9 if pd.notna(lab) else 0.6, zorder=z)
        if annotate:
            ax.text(g[XCOL].iloc[-1], g['VWC'].iloc[-1], f" {int(g['yield'].iloc[0])}",
                    color=c, fontsize=7, va='center')
    ax.set_xlabel(XLABEL)
    ax.set_ylabel('VWC')
    ax.grid(alpha=0.3)
    if XMODE == 'date':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.tick_params(axis='x', labelsize=8)


os.makedirs(OUT_DIR, exist_ok=True)
for _, r in groups.iterrows():
    muni, year, n = r.muniCd, int(r.year), int(r.n)
    fig, ax = plt.subplots(figsize=(9, 5))
    draw(ax, muni, year)
    ax.set_title(f'{names.get(muni, muni)} ({muni}) {year}年  圃場数={n}   [線端の数字=収量]')
    ax.plot([], [], color=COL[1], label='多収')
    ax.plot([], [], color=COL[0], label='低収')
    ax.plot([], [], color=MID, label='中間(ラベルなし)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'vwc_{muni}_{year}.png'), dpi=130)
    plt.close(fig)

# 一覧図
k = len(groups)
ncol = 5
nrow = int(np.ceil(k / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharey=True)
axes = np.atleast_1d(axes).ravel()
for ax, (_, r) in zip(axes, groups.iterrows()):
    draw(ax, r.muniCd, int(r.year), annotate=False)
    ax.set_title(f"{names.get(r.muniCd, r.muniCd)} {int(r.year)} (n={int(r.n)})", fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
for ax in axes[k:]:
    ax.axis('off')
fig.suptitle(f'地域x年ごとの VWC 推移 (赤=多収, 青=低収, 灰=中間)  横軸: {"播種後日数" if XMODE == "dap" else "実時間(日付)"}', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT_DIR, 'vwc_overview.png'), dpi=110)
plt.close(fig)
print(f'{k}グループの図を保存: {OUT_DIR}')
