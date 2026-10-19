"""
field_year_static.csv と soilmoisture_timeseries.csv の品質チェック。
使えない/注意が必要な圃場x年を洗い出し、outputs/quality_check.csv に保存する。
"""
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(R, 'data', 'processed')
st = pd.read_csv(os.path.join(P, 'field_year_static.csv'), dtype={'muniCd': str},
                 parse_dates=['seed_date', 'harvest_date'])
ts = pd.read_csv(os.path.join(P, 'soilmoisture_timeseries.csv'), parse_dates=['date'])

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

print('=== 基本統計 ===')
print(st.describe().T.to_string())

# ---------- 静的値 ----------
st['growing_days'] = (st['harvest_date'] - st['seed_date']).dt.days
st['seed_doy'] = st['seed_date'].dt.dayofyear
st['seed_year_ok'] = st['seed_date'].dt.year == st['year']
st['harv_year_ok'] = st['harvest_date'].dt.year == st['year']

print('\n=== 収量 ===')
print(st['yield'].describe().to_string())
print('yield<=0:', int((st['yield'] <= 0).sum()))
print('yield 下位10:\n', st.nsmallest(10, 'yield')[['field_id', 'year', 'muniCd', 'yield', 'n_obs']].to_string(index=False))
print('yield 上位10:\n', st.nlargest(10, 'yield')[['field_id', 'year', 'muniCd', 'yield', 'n_obs']].to_string(index=False))
print('yield 値の丸め(10の倍数割合):', round((st['yield'] % 10 == 0).mean(), 2))

print('\n=== 栽培期間 ===')
print(st['growing_days'].describe().to_string())
print('seed_date の年が year と不一致:', int((~st.seed_year_ok).sum()))
print('harvest_date の年が year と不一致:', int((~st.harv_year_ok).sum()))
print('growing_days<=0:', int((st.growing_days <= 0).sum()))
print('growing_days 短い下位8:\n', st.nsmallest(8, 'growing_days')[['field_id', 'year', 'seed_date', 'harvest_date', 'growing_days']].to_string(index=False))
print('growing_days 長い上位8:\n', st.nlargest(8, 'growing_days')[['field_id', 'year', 'seed_date', 'harvest_date', 'growing_days']].to_string(index=False))
print('播種日 最早/最遅:', st.seed_date.min().date(), st.seed_date.max().date())
print('収穫日 最早/最遅:', st.harvest_date.min().date(), st.harvest_date.max().date())

print('\n=== 位置 ===')
print('lat 範囲:', st.lat.min(), st.lat.max(), ' lon 範囲:', st.lon.min(), st.lon.max())
print('日本域外(lat<24 or >46 or lon<122 or >146):',
      int(((st.lat < 24) | (st.lat > 46) | (st.lon < 122) | (st.lon > 146)).sum()))

print('\n=== カテゴリ・栽培条件 ===')
print('irrigation:\n', st.irrigation.value_counts(dropna=False).to_string())
print('breed:\n', st.breed.value_counts(dropna=False).to_string())
for c in ['depth', 'between_lines', 'between_stocks']:
    print(f'{c}:\n', st[c].describe().to_string())

# ---------- 時系列 ----------
print('\n=== 時系列(圃場x年ごと) ===')
ts = ts.merge(st[['field_id', 'year', 'seed_date', 'harvest_date', 'growing_days']], on=['field_id', 'year'])
ts['dap_ratio'] = ts['dap'] / ts['growing_days']
print('dap<0 の観測:', int((ts.dap < 0).sum()), ' 収穫後(date>harvest)の観測:', int((ts.date > ts.harvest_date).sum()))
print('VWC<0:', int((ts.VWC < 0).sum()), ' VWC==0:', int((ts.VWC == 0).sum()), ' VWC>0.6:', int((ts.VWC > 0.6).sum()))
print('VWC 全体:', ts.VWC.describe().round(3).to_dict())


def summ(g):
    d = g.sort_values('date')
    gaps = d['date'].diff().dt.days.dropna()
    dup = d['date'].duplicated().sum()
    inside = d[(d.dap >= 0) & (d.date <= d.harvest_date)]
    return pd.Series({
        'n_total': len(d),
        'n_in_season': len(inside),
        'first_dap': d.dap.min(),
        'last_dap': d.dap.max(),
        'cover_ratio': (d.dap.max() - d.dap.min()) / d.growing_days.iloc[0],
        'max_gap_days': gaps.max() if len(gaps) else np.nan,
        'median_gap_days': gaps.median() if len(gaps) else np.nan,
        'dup_dates': dup,
        'vwc_mean': d.VWC.mean(),
        'vwc_std': d.VWC.std(),
        'vwc_min': d.VWC.min(),
        'vwc_max': d.VWC.max(),
        'n_neg': int((d.VWC < 0).sum()),
        'n_zero': int((d.VWC == 0).sum()),
        'max_jump': d.VWC.diff().abs().max(),
        'n_unique_vwc': d.VWC.nunique(),
    })


S = ts.groupby(['field_id', 'year']).apply(summ, include_groups=False).reset_index()
q = st.merge(S, on=['field_id', 'year'])
print('\nn_obs と n_total が一致しない:', int((q.n_obs != q.n_total).sum()))
print(q[['n_obs', 'n_in_season', 'cover_ratio', 'max_gap_days', 'median_gap_days', 'vwc_std', 'max_jump']].describe().round(3).T.to_string())

# ---------- フラグ ----------
flags = pd.DataFrame(index=q.index)
flags['few_obs(<30)'] = q.n_obs < 30
flags['short_cover(<0.5)'] = q.cover_ratio < 0.5
flags['large_gap(>14d)'] = q.max_gap_days > 14
flags['flat_vwc(std<0.005)'] = q.vwc_std < 0.005
flags['neg_vwc'] = q.n_neg > 0
flags['many_zero(>5%)'] = (q.n_zero / q.n_total) > 0.05
flags['big_jump(>0.2)'] = q.max_jump > 0.2
flags['dup_dates'] = q.dup_dates > 0
flags['yield_out(IQR)'] = False
Q1, Q3 = q['yield'].quantile([.25, .75])
iqr = Q3 - Q1
flags['yield_out(IQR)'] = (q['yield'] < Q1 - 1.5 * iqr) | (q['yield'] > Q3 + 1.5 * iqr)
flags['growing_days_odd(<70|>160)'] = (q.growing_days < 70) | (q.growing_days > 160)
flags['seed/harv_year_mismatch'] = ~(q.seed_year_ok & q.harv_year_ok)
flags['obs_outside_season(>10%)'] = (q.n_total - q.n_in_season) / q.n_total > 0.10

print('\n=== フラグ別の該当数(圃場x年) ===')
print(flags.sum().to_string())
q['n_flags'] = flags.sum(axis=1)
q['flag_names'] = flags.apply(lambda r: ','.join(r.index[r]), axis=1)
print('\nフラグ数の分布:', q.n_flags.value_counts().sort_index().to_dict())

cols = ['field_id', 'year', 'muniCd', 'yield', 'n_obs', 'growing_days', 'cover_ratio', 'max_gap_days', 'vwc_std', 'flag_names']
print('\n=== フラグが付いた圃場x年 ===')
print(q[q.n_flags > 0].sort_values('n_flags', ascending=False)[cols].round(3).to_string(index=False))

print('\n=== n_obs の分布 ===')
print(pd.cut(q.n_obs, [0, 10, 30, 60, 90, 120, 160]).value_counts().sort_index().to_string())
print('year別 n_obs 中央値:', q.groupby('year').n_obs.median().to_dict())

out = os.path.join(R, 'outputs', 'quality_check.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
q.to_csv(out, index=False, encoding='utf-8-sig')
print('\n保存:', out)
