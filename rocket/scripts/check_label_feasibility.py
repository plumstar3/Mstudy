import os
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(R, 'data', 'processed', 'field_year_static.csv'), dtype={'muniCd': str})

# 地域x年 のグループサイズ
g = df.groupby(['muniCd', 'year']).size().rename('n').reset_index()
print('地域x年グループ数:', len(g))
print('グループサイズ分布:\n', g.n.value_counts().sort_index().to_string())
for k in [3, 4, 5, 6]:
    ok = g[g.n >= k]
    print(f'n>={k}: グループ {len(ok)} / 圃場 {ok.n.sum()} ({ok.n.sum()/len(df)*100:.0f}%)')


def label(d, key, lo=1/3, hi=2/3):
    out = pd.Series(np.nan, index=d.index)
    for _, x in d.groupby(key):
        if len(x) < 3:
            continue
        r = x['yield'].rank(method='first', pct=True)
        out[x.index[r <= lo]] = 0
        out[x.index[r > hi]] = 1
    return out


for name, key in [('地域x年', ['muniCd', 'year']), ('年のみ', ['year'])]:
    l = label(df, key)
    print(f'\n[{name}] ラベル付与: {int(l.notna().sum())}件 (低収 {int((l==0).sum())} / 多収 {int((l==1).sum())})')
    print('  年別:', df.assign(l=l).dropna(subset=['l']).groupby('year').size().to_dict())

# 地域内の収量ばらつきと地域間ばらつき
print('\n収量 全体SD:', round(df['yield'].std(), 1))
print('地域x年内SD(平均, n>=3):', round(df.groupby(['muniCd', 'year'])['yield'].std()[g.set_index(['muniCd','year']).n>=3].mean(), 1))
print('年別平均収量:', df.groupby('year')['yield'].mean().round(0).to_dict())
print('地域x年内の収量レンジ(max-min)中央値(n>=4):',
      df.groupby(['muniCd', 'year'])['yield'].agg(lambda s: s.max()-s.min())[g.set_index(['muniCd','year']).n>=4].median())

# 同一圃場の複数年
print('field_idユニーク/行:', df.field_id.nunique(), len(df))
print('irrigation:', df.irrigation.value_counts().to_dict())
