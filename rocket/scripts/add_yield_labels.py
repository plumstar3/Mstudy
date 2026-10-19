"""
field_year_static.csv に多収・低収ラベルを付与する。

label_muniyear : 同一 muniCd x 同一年 の中で収量の上位/下位 約1/3 (グループ4圃場以上のみ)
label_year     : 同一年 の中で収量の上位/下位 約1/3 (比較用)
  0 = 低収, 1 = 多収, 空欄 = ラベルなし(中間 or 対象外)

グループ内 n 件のとき、上位と下位から同数 k = round(n/3) 件ずつ採る(残りは中間として除外)。
収量が同値の場合は並び順(field_id昇順)で決める。

出力: rocket/data/processed/field_year_labeled.csv
"""
import os

import numpy as np
import pandas as pd

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_static.csv')
OUT_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_labeled.csv')

MIN_GROUP_SIZE = 4  # 地域x年ラベルを付ける最小グループ圃場数


def assign_label(df, keys, min_size):
    """keys でグループ化し、各グループで yield の下位k件=0, 上位k件=1 を付ける。"""
    label = pd.Series(np.nan, index=df.index, dtype='float')
    for _, g in df.groupby(keys):
        n = len(g)
        if n < min_size:
            continue
        k = int(np.floor(n / 3 + 0.5))
        if k < 1:
            continue
        order = g.sort_values(['yield', 'field_id']).index
        label[order[:k]] = 0
        label[order[-k:]] = 1
    return label


def main():
    df = pd.read_csv(IN_CSV, dtype={'muniCd': str})

    df['label_muniyear'] = assign_label(df, ['muniCd', 'year'], MIN_GROUP_SIZE)
    df['label_year'] = assign_label(df, ['year'], 1)

    # 欠損可能な整数列として保存
    for c in ['label_muniyear', 'label_year']:
        df[c] = df[c].astype('Int64')

    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'出力: {OUT_CSV} ({len(df)}行)')

    for c in ['label_muniyear', 'label_year']:
        s = df[c]
        print(f'\n[{c}] ラベル付与 {int(s.notna().sum())}件 (低収 {int((s == 0).sum())} / 多収 {int((s == 1).sum())})')
        print('  年別:', df[s.notna()].groupby('year').size().to_dict())
        print('  付与された地域数:', df.loc[s.notna(), 'muniCd'].nunique())
        print('  平均収量 低収/多収:',
              round(df.loc[s == 0, 'yield'].mean(), 1), '/', round(df.loc[s == 1, 'yield'].mean(), 1))

    both = df['label_muniyear'].notna() & df['label_year'].notna()
    agree = (df.loc[both, 'label_muniyear'] == df.loc[both, 'label_year']).mean()
    print(f'\n両ラベルが付いている{int(both.sum())}件での一致率: {agree:.2f}')


if __name__ == '__main__':
    main()
