"""
VWC の観測状況を図にする。
  上: 播種後日数(dap)ごとの観測圃場×年の数(全件 / label_muniyear 付き94件)
  下: 系列長(観測日数)の分布

出力: rocket/outputs/vwc_coverage/
  vwc_coverage.png        図
  n_by_dap.csv            dap ごとの観測件数
  series_length.csv       圃場×年ごとの観測開始・終了 dap と系列長
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'vwc_coverage')

WINDOWS = {'案C 30–89': (30, 89), '案A 50–99': (50, 99)}
C_ALL, C_LAB = '#eb6834', '#2a78d6'  # 全件 / label_muniyear 付き
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig')
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig',
                      dtype={'muniCd': str})
    keys94 = lab.loc[lab['label_muniyear'].notna(), ['field_id', 'year']]

    sl = ts.groupby(['field_id', 'year'])['dap'].agg(start_dap='min', end_dap='max', n_obs='size').reset_index()
    sl['span'] = sl['end_dap'] - sl['start_dap'] + 1
    sl['gap_days'] = sl['span'] - sl['n_obs']
    sl = sl.merge(keys94.assign(labeled=1), on=['field_id', 'year'], how='left').fillna({'labeled': 0})
    sl.to_csv(os.path.join(OUT_DIR, 'series_length.csv'), index=False, encoding='utf-8-sig')

    t94 = ts.merge(keys94, on=['field_id', 'year'])
    dap = np.arange(ts['dap'].min(), ts['dap'].max() + 1)
    n_all = ts.groupby('dap').size().reindex(dap, fill_value=0)
    n_lab = t94.groupby('dap').size().reindex(dap, fill_value=0)
    pd.DataFrame({'dap': dap, 'n_all': n_all.values, 'n_labeled94': n_lab.values}).to_csv(
        os.path.join(OUT_DIR, 'n_by_dap.csv'), index=False, encoding='utf-8-sig')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [3, 2]})
    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # 上: dap ごとの観測件数
    for i, (name, (lo, hi)) in enumerate(WINDOWS.items()):
        ax1.axvspan(lo, hi, ymin=0, ymax=1, color='#9a9890', alpha=0.10 + 0.06 * i, lw=0)
        ax1.text((lo + hi) / 2, 1.0 + 0.045 * (1 - i), name, transform=ax1.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=9, color=MUTED)
    n_tot, n94 = len(sl), int(sl['labeled'].sum())
    ax1.plot(dap, n_all.values, color=C_ALL, lw=2, label=f'全件({n_tot} 圃場×年)')
    ax1.plot(dap, n_lab.values, color=C_LAB, lw=2, label=f'label_muniyear 付き({n94} 件)')
    for s, c in [(n_all, C_ALL), (n_lab, C_LAB)]:
        m = s.idxmax()
        ax1.annotate(f'最大 {s.max()}(dap {m})', (m, s.max()), xytext=(6, 4),
                     textcoords='offset points', fontsize=9, color=INK)
    ax1.set_xlabel('播種後日数(dap)', color=INK)
    ax1.set_ylabel('観測がある圃場×年の数', color=INK)
    ax1.set_xlim(dap.min(), dap.max())
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, loc='upper right', labelcolor=INK)

    # 下: 系列長の分布
    bins = np.arange(0, sl['n_obs'].max() + 11, 10)
    ax2.hist(sl['n_obs'], bins=bins, histtype='step', lw=2, color=C_ALL,
             label=f'全件(中央値 {sl.n_obs.median():.0f} 日)')
    ax2.hist(sl.loc[sl.labeled == 1, 'n_obs'], bins=bins, histtype='stepfilled', lw=0,
             color=C_LAB, alpha=0.85, label=f'label_muniyear 付き(中央値 {sl.loc[sl.labeled == 1, "n_obs"].median():.0f} 日)')
    ax2.set_xlabel('系列長(観測日数)', color=INK)
    ax2.set_ylabel('圃場×年の数', color=INK)
    ax2.legend(frameon=False, loc='upper left', labelcolor=INK)

    fig.suptitle('VWC の観測状況', fontsize=14, color=INK, x=0.06, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'vwc_coverage.png'), dpi=150)

    # 数値の要約
    print('欠けのある系列:', int((sl.gap_days > 0).sum()), '/', len(sl))
    for name, d in [('全件', sl), ('label_muniyear 付き', sl[sl.labeled == 1])]:
        print(f'[{name}] n={len(d)}')
        for c in ['n_obs', 'start_dap', 'end_dap']:
            print(f'  {c}: ' + ', '.join(f'{k} {v:.0f}' for k, v in d[c].describe()[['min', '25%', '50%', '75%', 'max']].items()))
    print('年別の系列長(全件 中央値/最小/最大):')
    print(sl.groupby('year')['n_obs'].agg(['size', 'median', 'min', 'max']).to_string())
    print('年別の系列長(94件):')
    print(sl[sl.labeled == 1].groupby('year')['n_obs'].agg(['size', 'median', 'min', 'max']).to_string())
    print(f'出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
