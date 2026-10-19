import os
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(R, 'data', 'processed')
st = pd.read_csv(os.path.join(P, 'field_year_static.csv'), dtype={'muniCd': str}, parse_dates=['seed_date', 'harvest_date'])
ts = pd.read_csv(os.path.join(P, 'soilmoisture_timeseries.csv'), parse_dates=['date'])
pd.set_option('display.width', 250)

# 1) 土壌水分系列(日付+VWC)が完全一致する圃場x年のグループ
sig = ts.sort_values('date').groupby(['field_id', 'year']).apply(
    lambda g: hash(tuple(zip(g['date'].dt.strftime('%Y%m%d'), g['VWC'].round(6)))), include_groups=False)
sig = sig.rename('sig').reset_index()
sz = sig.groupby('sig').size().rename('n_share')
sig = sig.merge(sz, on='sig')
dup = sig[sig.n_share > 1].merge(st, on=['field_id', 'year'])
print('同一VWC系列を共有する圃場x年:', len(dup), '/ グループ数', dup.sig.nunique())
print('共有グループサイズ分布:', dup.groupby('sig').size().value_counts().sort_index().to_dict())
g = dup.groupby('sig').agg(n=('field_id', 'size'), year=('year', 'first'), muni=('muniCd', 'first'),
                           ymin=('yield', 'min'), ymax=('yield', 'max'), fids=('field_id', lambda s: list(s)))
g['yield_range'] = g.ymax - g.ymin
print(g.sort_values('n', ascending=False).head(30).round(1).to_string())
print('共有グループ内の収量レンジ 中央値:', round(g.yield_range.median(), 1))
print('ユニークなVWC系列数:', sig.sig.nunique(), '/ 圃場x年', len(sig))

# 2) 同一 place(土壌センサ地点)単位の確認用: 緯度経度が同一の圃場
ll = st.groupby(['year', 'lat', 'lon']).size()
print('\n同一(year,lat,lon)に複数圃場:', int((ll > 1).sum()), 'グループ /', int(ll[ll > 1].sum()), '圃場')

# 3) harvest/seed 年不一致
st['gd'] = (st.harvest_date - st.seed_date).dt.days
print('\n年不一致:\n', st[st.harvest_date.dt.year != st.year][['field_id', 'year', 'seed_date', 'harvest_date', 'gd', 'yield', 'n_obs']].to_string(index=False))

# 4) 圃場x年内の収量が同一(整数/小数)
st['int_yield'] = (st['yield'] % 1 == 0)
print('\n収量が整数:', int(st.int_yield.sum()), ' 小数:', int((~st.int_yield).sum()))

# 5) 観測開始/終了のdap
d = ts.merge(st[['field_id', 'year', 'seed_date', 'harvest_date']], on=['field_id', 'year'])
f = d.groupby(['field_id', 'year']).agg(first_date=('date', 'min'), last_date=('date', 'max')).reset_index().merge(st, on=['field_id', 'year'])
f['start_dap'] = (f.first_date - f.seed_date).dt.days
f['end_to_harv'] = (f.last_date - f.harvest_date).dt.days
print('\n観測開始の播種後日数 (負=播種前から観測):', f.start_dap.describe().round(1).to_dict())
print('観測終了-収穫日(日):', f.end_to_harv.describe().round(1).to_dict())
print('観測開始が播種後30日超:', int((f.start_dap > 30).sum()), ' 観測終了が収穫の30日以上前:', int((f.end_to_harv < -30).sum()))
print('観測終了日 最頻:', f.last_date.dt.strftime('%m-%d').value_counts().head(5).to_dict())
print('観測開始日 最頻:', f.first_date.dt.strftime('%m-%d').value_counts().head(5).to_dict())

# 6) field_id=5 の詳細
print('\nfield_id=5:\n', st[st.field_id == 5].T.to_string())
print(ts[ts.field_id == 5].to_string(index=False))
