import sqlite3, os
import numpy as np, pandas as pd

DB = os.path.join('data', 'processed', 'FieldData_fieldid.db')
EDGES = list(range(30, 85, 5))
COLS = [f'vwc_dap{lo:03d}' for lo in EDGES[:-1]]

conn = sqlite3.connect(DB)
q = pd.read_sql('''SELECT field_id, year, yield, breed, seed_date, harvest_date FROM Questionaire
 WHERE field_id IS NOT NULL AND yield IS NOT NULL AND seed_date IS NOT NULL
 AND harvest_date IS NOT NULL AND year BETWEEN 2016 AND 2018''', conn)
ids = pd.read_sql('SELECT DISTINCT field_id, year FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
conn.close()

q['field_id'] = q['field_id'].astype(int); q['year'] = q['year'].astype(int)
q['seed_date'] = pd.to_datetime(q['seed_date'], errors='coerce')
q['harvest_date'] = pd.to_datetime(q['harvest_date'], errors='coerce')
q['growing_days'] = (q['harvest_date'] - q['seed_date']).dt.days
q = q[q['growing_days'] > 0]
ids['field_id'] = pd.to_numeric(ids['field_id'], errors='coerce').dropna().astype(int)
key = set(zip(ids['field_id'], ids['year'].astype(int)))
q = q[[(a, b) in key for a, b in zip(q['field_id'], q['year'])]].reset_index(drop=True)

# 分類ラベル
lab = pd.Series(np.nan, index=q.index)
for yr, g in q.groupby('year'):
    hi = g['yield'].quantile(0.67); lo = g['yield'].quantile(0.33)
    lab[g.index[g['yield'] >= hi]] = 1; lab[g.index[g['yield'] <= lo]] = 0
q['label'] = lab

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
sm = sm.dropna(subset=['field_id', 'date', 'VWC']); sm['field_id'] = sm['field_id'].astype(int)
m = sm.merge(q[['field_id', 'year', 'seed_date', 'growing_days']], on=['field_id', 'year'])
m['dap'] = (m['date'] - m['seed_date']).dt.days / m['growing_days'] * 100

rows = []
for (f, y), g in m.groupby(['field_id', 'year']):
    r = {'field_id': f, 'year': y, 'n_obs': len(g)}
    for lo, c in zip(EDGES[:-1], COLS):
        mk = (g['dap'] >= lo) & (g['dap'] < lo + 5)
        r[c] = g['VWC'][mk].mean() if mk.sum() else np.nan
    rows.append(r)
F = pd.DataFrame(rows)
d = q.merge(F, on=['field_id', 'year'], how='inner')

def rep(name, df):
    X = df[COLS]
    tot = X.size; nm = int(X.isna().sum().sum())
    print(f'\n== {name}: {len(df)}件 x {len(COLS)}ビン = {tot}セル')
    print(f'  欠損セル: {nm} ({nm/tot*100:.1f}%)')
    print(f'  欠損が1つ以上ある圃場: {int(X.isna().any(axis=1).sum())}件')
    print(f'  全ビン欠損の圃場: {int(X.isna().all(axis=1).sum())}件')
    print('  ビン別欠損率(%):', {c[-3:]: round(X[c].isna().mean()*100, 1) for c in COLS})
    print('  年別件数:', df.groupby('year').size().to_dict())
    print('  圃場あたり観測数 中央値/最小/最大:', df['n_obs'].median(), df['n_obs'].min(), df['n_obs'].max())

rep('回帰(全件)', d)
rep('分類(中間除外後)', d.dropna(subset=['label']))
