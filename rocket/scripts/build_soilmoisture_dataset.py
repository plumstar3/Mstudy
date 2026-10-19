"""
土壌水分量による多収・低収分類のためのデータセットを2テーブルで作成する。

1) soilmoisture_timeseries.csv  (1行 = 1圃場 x 1年 x 1観測日)
     field_id, year, date, dap(播種後日数), VWC
2) field_year_static.csv        (1行 = 1圃場 x 1年)
     field_id, year, muniCd, yield, seed_date, harvest_date, breed, irrigation,
     lat, lon, depth, between_lines, between_stocks, n_obs
結合キー: (field_id, year)。muniCd のみ field_id で field_addresses.csv と結合する。
ラベル(多収/低収)の付与は行わない。

出力先: rocket/data/processed/
"""
import os
import sqlite3

import pandas as pd

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(ROCKET_DIR)
DB = os.path.join(ROOT_DIR, 'data', 'processed', 'FieldData_fieldid.db')
ADDR = os.path.join(ROOT_DIR, 'outputs', 'reverse_geocode', 'field_addresses.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'data', 'processed')
OUT_TS = os.path.join(OUT_DIR, 'soilmoisture_timeseries.csv')
OUT_STATIC = os.path.join(OUT_DIR, 'field_year_static.csv')

STATIC_COLS = ['yield', 'seed_date', 'harvest_date', 'breed', 'irrigation',
               'lat', 'lon', 'depth', 'between_lines', 'between_stocks']


def main():
    conn = sqlite3.connect(DB)
    q = pd.read_sql(
        f"SELECT field_id, year, {', '.join(STATIC_COLS)} FROM Questionaire", conn)
    sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture', conn)
    conn.close()
    print(f'Questionaire: {len(q)}行 / SolidMoisture: {len(sm)}行')

    # field_id が NULL の行は結合できないため除外
    q = q.dropna(subset=['field_id'])
    sm = sm.dropna(subset=['field_id'])
    q['field_id'] = pd.to_numeric(q['field_id'], errors='coerce')
    sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce')
    q = q.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})
    sm = sm.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})
    print(f'field_id NULL除外後: Questionaire {len(q)}行 / SolidMoisture {len(sm)}行')

    # (field_id, year) の重複確認(重複があれば結合で行が増えるため)
    dup = q.duplicated(['field_id', 'year'], keep=False)
    print(f'Questionaire の (field_id, year) 重複行: {int(dup.sum())}')
    q = q.drop_duplicates(['field_id', 'year'], keep='first')

    # 時系列の整形
    sm['date'] = pd.to_datetime(sm['date'], errors='coerce')
    sm['VWC'] = pd.to_numeric(sm['VWC'], errors='coerce')
    sm = sm.dropna(subset=['date', 'VWC'])
    print(f'VWC 負値: {int((sm["VWC"] < 0).sum())}行 (そのまま保持。解析時に要処理)')

    # 圃場x年の静的属性を結合
    d = sm.merge(q, on=['field_id', 'year'], how='inner')
    print(f'Questionaire と結合できた観測: {len(d)}行')

    # yield 欠損は分類に使えないため除外
    d = d.dropna(subset=['yield'])

    # 地域コード
    addr = pd.read_csv(ADDR, encoding='utf-8-sig', dtype={'muniCd': str})
    addr['field_id'] = addr['field_id'].astype(int)
    addr = addr[['field_id', 'muniCd']].drop_duplicates('field_id')
    d = d.merge(addr, on='field_id', how='left')

    # 播種後日数
    d['seed_date'] = pd.to_datetime(d['seed_date'], errors='coerce')
    d['harvest_date'] = pd.to_datetime(d['harvest_date'], errors='coerce')
    d['dap'] = (d['date'] - d['seed_date']).dt.days

    # 時系列テーブル
    ts = d[['field_id', 'year', 'date', 'dap', 'VWC']]
    ts = ts.sort_values(['field_id', 'year', 'date']).reset_index(drop=True)

    # 圃場x年の静的テーブル(1行 = 1圃場x1年)
    n_obs = ts.groupby(['field_id', 'year']).size().rename('n_obs').reset_index()
    static_cols = ['field_id', 'year', 'muniCd', 'yield', 'seed_date', 'harvest_date',
                   'breed', 'irrigation', 'lat', 'lon', 'depth',
                   'between_lines', 'between_stocks']
    st = d[static_cols].drop_duplicates(['field_id', 'year'])
    st = st.merge(n_obs, on=['field_id', 'year']).sort_values(['field_id', 'year']).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts.to_csv(OUT_TS, index=False, encoding='utf-8-sig')
    st.to_csv(OUT_STATIC, index=False, encoding='utf-8-sig')

    print(f'\n出力: {OUT_TS}\n      {OUT_STATIC}')
    print(f'時系列: {len(ts)}行 / 静的: {len(st)}行(圃場x年)')
    print('年別 圃場x年:', st.groupby('year').size().to_dict())
    print('静的テーブルの各列の欠損数:\n', st.isna().sum().to_string())
    print('圃場x年あたり観測数 (min/中央/max):', st.n_obs.min(), st.n_obs.median(), st.n_obs.max())


if __name__ == '__main__':
    main()
