import os, sqlite3, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
q = pd.read_sql('''SELECT field_id, year, yield, seed_date, harvest_date FROM Questionaire WHERE yield IS NOT NULL
 AND seed_date IS NOT NULL AND harvest_date IS NOT NULL AND field_id IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
conn.close()
sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce'); sm = sm.dropna(subset=['field_id']); sm['field_id'] = sm['field_id'].astype(int)
sm['year'] = sm['year'].astype(int); sm['date'] = pd.to_datetime(sm['date'], errors='coerce'); sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['date', 'VWC'])
q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date']); q['harvest_date'] = pd.to_datetime(q['harvest_date'])
q['gd'] = (q['harvest_date'] - q['seed_date']).dt.days; q = q[q['gd'] > 0]
geo = pd.read_csv('outputs/reverse_geocode/field_Addresses.csv', encoding='utf-8-sig')
geo['field_id'] = geo['field_id'].astype(int)
geo = geo[['field_id', 'muniCd', 'city']].drop_duplicates('field_id')

d = sm.merge(q, on=['field_id', 'year']).merge(geo, on='field_id')
d['dap'] = (d['date'] - d['seed_date']).dt.days / d['gd'] * 100
w = d[(d['dap'] >= 40) & (d['dap'] <= 80)]
# 圃場×年 単位の窓平均VWC
f = w.groupby(['field_id', 'year', 'muniCd', 'city']).agg(vwc=('VWC', 'mean'), y=('yield', 'first'), n=('VWC', 'size')).reset_index()
print('圃場×年(DAP40-80%):', len(f), ' 窓内の観測数/圃場 中央値', f['n'].median())

# A) 全体、年別33% → 圃場単位検定（旧分析の定義）
f['grpA'] = 'Mid'
for yr, g in f.groupby('year'):
    lo, hi = g['y'].quantile(.33), g['y'].quantile(.67)
    f.loc[g.index[g['y'] <= lo], 'grpA'] = 'Low'; f.loc[g.index[g['y'] >= hi], 'grpA'] = 'High'
h, l = f[f.grpA == 'High']['vwc'], f[f.grpA == 'Low']['vwc']
print(f'\n[A] 全体・圃場単位: High n={len(h)} mean={h.mean():.4f} / Low n={len(l)} mean={l.mean():.4f} / diff={h.mean()-l.mean():+.4f} / MWU p={mannwhitneyu(h,l)[1]:.4f}')

# B) 地域(市町村)×年偏差で同一地域内に補正
f['y_dev'] = f['y'] - f.groupby(['muniCd', 'year'])['y'].transform('mean')
f['v_dev'] = f['vwc'] - f.groupby(['muniCd', 'year'])['vwc'].transform('mean')
print('\n[B] 地域×年平均との偏差（地域差・年差を除去）:')
r = np.corrcoef(f['y_dev'], f['v_dev'])[0, 1]
from scipy.stats import spearmanr
print(f'  全体 spearman(yield偏差, VWC偏差) = {spearmanr(f.y_dev, f.v_dev)[0]:+.3f} (p={spearmanr(f.y_dev, f.v_dev)[1]:.3f}), n={len(f)}')
print(f'  spearman(yield, vwc) 素の値      = {spearmanr(f.y, f.vwc)[0]:+.3f} (p={spearmanr(f.y, f.vwc)[1]:.3f})')

# C) 市町村ごとの相関符号
rows = []
for (m, c), g in f.groupby(['muniCd', 'city']):
    if len(g) >= 6:
        rho = spearmanr(g['y'], g['vwc'])[0]
        rows.append((c, len(g), rho, g['vwc'].mean(), g['y'].mean()))
t = pd.DataFrame(rows, columns=['city', 'n', 'rho_y_vwc', 'vwc_mean', 'yield_mean']).sort_values('n', ascending=False)
print('\n[C] 市町村内 spearman(収量, VWC) (n>=6):'); print(t.round(3).to_string(index=False))
print(f'  負の市町村 {int((t.rho_y_vwc<0).sum())} / 正 {int((t.rho_y_vwc>0).sum())}')
print(f'\n[D] 市町村間: spearman(市町村平均VWC, 市町村平均収量) = {spearmanr(t.vwc_mean, t.yield_mean)[0]:+.3f}')
