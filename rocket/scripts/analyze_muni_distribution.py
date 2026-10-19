"""
field_year_static.csv から自治体(muniCd)別の圃場数・年別内訳を集計するスクリプト。
出力: rocket/outputs/muni_field_distribution.csv
"""
import os
import pandas as pd

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(ROCKET_DIR)

STATIC_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_static.csv')
ADDR_CSV = os.path.join(ROOT_DIR, 'outputs', 'reverse_geocode', 'field_addresses.csv')
OUT_CSV = os.path.join(ROCKET_DIR, 'outputs', 'muni_field_distribution.csv')


def main():
    df = pd.read_csv(STATIC_CSV, dtype={'muniCd': str})
    addr = pd.read_csv(ADDR_CSV, dtype={'muniCd': str})

    # muniCd -> 都道府県・市区町村のマッピング
    muni_info = addr.drop_duplicates('muniCd').set_index('muniCd')[['prefecture', 'city', 'town']].fillna('')

    # クロス集計 (行: muniCd, 列: year)
    ct = pd.crosstab(df['muniCd'], df['year'])
    for yr in [2016, 2017, 2018]:
        if yr not in ct.columns:
            ct[yr] = 0
    ct = ct[[2016, 2017, 2018]]

    # 合計列
    ct['合計圃場数'] = ct.sum(axis=1)

    # 住所情報結合
    res = ct.join(muni_info, how='left')

    # カラム並び替えとソート
    cols = ['prefecture', 'city', 2016, 2017, 2018, '合計圃場数']
    res = res[cols].sort_values(by=['合計圃場数', 'prefecture', 'city'], ascending=[False, True, True])

    # 合計行の追加
    total_row = pd.DataFrame([{
        'prefecture': '全国合計',
        'city': f'{len(res)} 市町村',
        2016: int(df[df["year"] == 2016]["field_id"].count()),
        2017: int(df[df["year"] == 2017]["field_id"].count()),
        2018: int(df[df["year"] == 2018]["field_id"].count()),
        '合計圃場数': len(df)
    }], index=['合計'])
    
    # 保存用
    res_to_save = pd.concat([res, total_row])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    res_to_save.to_csv(OUT_CSV, encoding='utf-8-sig', index_label='muniCd')
    print(f"集計結果を保存しました: {OUT_CSV}")

    # コンソール表示用（markdownフォーマットで出力）
    print("\n### 自治体(muniCd)別 圃場数集計一覧")
    print(res.to_markdown())


if __name__ == '__main__':
    main()
