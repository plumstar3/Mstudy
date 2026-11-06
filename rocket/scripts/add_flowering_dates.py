"""
field_year_labeled.csv に開花日の列を追加する(add_yield_labels.py の後に実行する)。

flower_date_obs  : Questionaire.flower_date(実測。無ければ空欄)
flower_date_pred : flowering_pred_64.csv の pred_flower_date(実測が無い64件のみ。EXP-016)
flower_date      : 解析に使う開花日(実測を優先し、無ければ予測)
flower_source    : 'obs' / 'pred'

実測の読み込み元は rocket/data/raw/FieldData_fieldid.db。
予測は scripts/predict_flowering_targets.py が作る rocket/data/processed/flowering_pred_64.csv。
既に開花日の列がある場合は作り直す(何度実行しても同じ結果になる)。

出力: rocket/data/processed/field_year_labeled.csv(上書き)
"""
import os
import sqlite3

import pandas as pd

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROCKET_DIR, 'data', 'raw', 'FieldData_fieldid.db')
PRED_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'flowering_pred_64.csv')
LABELED_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_labeled.csv')

FLOWER_COLS = ['flower_date_obs', 'flower_date_pred', 'flower_date', 'flower_source']


def load_observed():
    conn = sqlite3.connect(DB)
    q = pd.read_sql('SELECT field_id, year, flower_date FROM Questionaire', conn)
    conn.close()

    # field_id が NULL の行は結合できないため除外(build_soilmoisture_dataset.py と同じ扱い)
    q['field_id'] = pd.to_numeric(q['field_id'], errors='coerce')
    q = q.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})

    q['flower_date_obs'] = pd.to_datetime(q['flower_date'], errors='coerce')
    dup = q.groupby(['field_id', 'year'])['flower_date_obs'].nunique()
    if (dup > 1).any():
        raise ValueError(f'同じ圃場x年で実測開花日が食い違う: {dup[dup > 1].index.tolist()}')
    return q.drop_duplicates(['field_id', 'year'])[['field_id', 'year', 'flower_date_obs']]


def load_predicted():
    p = pd.read_csv(PRED_CSV, encoding='utf-8-sig')
    p['flower_date_pred'] = pd.to_datetime(p['pred_flower_date'])
    return p[['field_id', 'year', 'flower_date_pred']]


def main():
    # round_trip: 上書き保存で yield などの小数の末尾桁が変わらないようにする
    df = pd.read_csv(LABELED_CSV, dtype={'muniCd': str}, encoding='utf-8-sig',
                     float_precision='round_trip')
    df = df.drop(columns=[c for c in FLOWER_COLS if c in df.columns])
    # ラベル列は欠損可能な整数のまま保つ(add_yield_labels.py と同じ)
    for c in [c for c in df.columns if c.startswith('label_')]:
        df[c] = pd.to_numeric(df[c]).astype('Int64')
    n0 = len(df)

    df = df.merge(load_observed(), on=['field_id', 'year'], how='left')
    df = df.merge(load_predicted(), on=['field_id', 'year'], how='left')
    assert len(df) == n0, '結合で行数が変わった'

    obs = df['flower_date_obs'].notna()
    pred = df['flower_date_pred'].notna()
    if (obs & pred).any():
        print(f'注意: 実測と予測の両方がある件 {int((obs & pred).sum())}件(実測を使う)')
    missing = df.loc[~obs & ~pred, ['field_id', 'year']]
    if len(missing):
        print(f'注意: 開花日が無い件 {len(missing)}件 -> {missing.values.tolist()}')

    df['flower_date'] = df['flower_date_obs'].fillna(df['flower_date_pred'])
    df['flower_source'] = pd.Series(pd.NA, index=df.index, dtype='string')
    df.loc[pred, 'flower_source'] = 'pred'
    df.loc[obs, 'flower_source'] = 'obs'

    for c in ['flower_date_obs', 'flower_date_pred', 'flower_date']:
        df[c] = df[c].dt.strftime('%Y-%m-%d')

    df.to_csv(LABELED_CSV, index=False, encoding='utf-8-sig')
    print(f'出力: {LABELED_CSV} ({len(df)}行)')

    print('\n開花日の出どころ:', df['flower_source'].value_counts(dropna=False).to_dict())
    for c in ['label_muniyear', 'label_year', 'label_muniyear_median_min2']:
        s = df.loc[df[c].notna(), 'flower_source']
        print(f'  [{c}] {len(s)}件:', s.value_counts(dropna=False).to_dict())

    d2f = (pd.to_datetime(df['flower_date']) - pd.to_datetime(df['seed_date'])).dt.days
    print('\n播種から開花までの日数(実測 / 予測):')
    for src in ['obs', 'pred']:
        print(f'  {src}:', d2f[df['flower_source'] == src].describe()[['count', 'min', '50%', 'max']].round(1).to_dict())


if __name__ == '__main__':
    main()
