"""
開花日の無い 64 件(収量あり・SolidMoisture あり)の開花日を予測する。

採用モデル: 播種日回帰 d2f ~ 播種日の通日(scripts/eval_flowering_raw_seed.py の評価で、GroupKFold・
  leave-one-region-out とも MAE 最良または同等。EXP-016)。
学習データ: 開花日と seed_date_raw のある 463 件(D1〜D4・2015 を含める)。
比較用に RF(緯度・播種日・品種、ハイパーパラメータは学習データ全体での GroupKFold(地域×年)の内側 CV)と
  統計モデル(播種日 + 緯度 + 40 日平均気温)でも予測する。
播種日: seed_date_raw を使う。空の件は補完済みの seed_date を使い、seed_imputed = True とする。
pred_flower_date = 播種日 + round(pred_d2f) 日。
region_mae: 採用モデルの GroupKFold(地域×年、D を含める)での地域別 MAE
  (outputs/flowering_raw_seed/cv_metrics.csv)。

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db、rocket/data/processed/daylength.csv、
                    rocket/outputs/flowering_raw_seed/cv_metrics.csv
出力:
  rocket/data/processed/flowering_pred_64.csv
      field_id, year, place, breed, seed_date_used, seed_imputed, pred_d2f, pred_flower_date,
      pred_rf, pred_stat, region_mae
  rocket/outputs/flowering_raw_seed/targets_check.md
      播種日が学習データの範囲から外れる件、モデル間の食い違い、土壌水分のそろい方、D を除いた場合の差
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import eval_flowering_raw_seed as ev  # noqa: E402
import fit_flowering_gdd as gdd  # noqa: E402
import fit_flowering_rf as rfm  # noqa: E402

CV_METRICS = os.path.join(ev.OUT_DIR, 'cv_metrics.csv')
OUT_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'flowering_pred_64.csv')
OUT_MD = os.path.join(ev.OUT_DIR, 'targets_check.md')

DISAGREE_DAYS = 5
WINDOW = 60          # 開花日(0 日目)から +59 日まで
OUT_COLS = ['field_id', 'year', 'place', 'breed', 'seed_date_used', 'seed_imputed', 'pred_d2f',
            'pred_flower_date', 'pred_rf', 'pred_stat', 'region_mae']


def target_frame(q, w, sm):
    """64 件と、予測に使う特徴量(播種日の通日・緯度・40 日平均気温・品種)。"""
    _, tg = gdd.no_flower_targets(q, sm)
    tg = tg.copy()
    tg['seed_imputed'] = tg['seed_date_raw'].isna()
    tg['seed_date_used'] = tg['seed_date_raw'].fillna(tg['seed_date_imputed'])
    tg['sd'] = pd.to_datetime(tg['seed_date_used'])
    tg['seed_doy'] = tg['sd'].dt.dayofyear
    tg['region'] = tg['place'].str[0]
    tg['breed_rf'] = tg['breed'].replace('(欠損)', '不明')
    tw = w.set_index(['field_id', 'date'])['TMP_mea']
    tm = []
    for r in tg.itertuples():
        days = pd.date_range(r.sd + pd.Timedelta(days=1), periods=rfm.STAT_DAYS)
        v = tw.reindex(pd.MultiIndex.from_product([[r.field_id], days])).values
        assert np.isfinite(v).all(), f'気温に欠け: field_id={r.field_id}'
        tm.append(v.mean())
    tg['tmean40'] = tm
    return tg.reset_index(drop=True)


def fit_seed_reg(train):
    return np.linalg.lstsq(rfm.X_seed(train), train['d2f'].values, rcond=None)[0]


def soil_coverage(tg, harvest):
    """予測開花日から +59 日までの 60 日に VWC がそろっているか(収穫日以前に限る場合も)。"""
    conn = sqlite3.connect(f'file:{gdd.DB}?mode=ro', uri=True)
    vwc = pd.read_sql('SELECT field_id, date, VWC FROM SolidMoisture WHERE VWC IS NOT NULL', conn)
    conn.close()
    vwc['date'] = pd.to_datetime(vwc['date'])
    have = set(zip(vwc['field_id'], vwc['date']))
    rows = []
    for r in tg.itertuples():
        f0 = pd.Timestamp(r.pred_flower_date)
        days = pd.date_range(f0, periods=WINDOW)
        n_ok = sum((r.field_id, d) in have for d in days)
        g = vwc[vwc['field_id'] == r.field_id]['date']
        hd = harvest.get(r.field_id)
        rows.append({'field_id': r.field_id, 'n_days_ok': n_ok, 'full': n_ok == WINDOW,
                     'before_harvest': pd.notna(hd) and days[-1] <= pd.Timestamp(hd),
                     'vwc_first': g.min().date(), 'vwc_last': g.max().date(),
                     'harvest_date': hd})
    return pd.DataFrame(rows)


def main():
    q, w, sm, dl = ev.load_raw()
    train, _ = ev.build(q, w, dl)
    tg = target_frame(q, w, sm)

    # 採用モデル(播種日回帰)と比較用の 2 モデル
    b = fit_seed_reg(train)
    tg['pred_d2f'] = rfm.X_seed(tg) @ b
    tg['pred_stat'] = rfm.linear(rfm.X_stat(train), train['d2f'].values, rfm.X_stat(tg))
    gs = GridSearchCV(rfm.rf_pipeline(), rfm.RF_GRID, scoring='neg_root_mean_squared_error',
                      cv=rfm.inner_cv('group', train), n_jobs=-1, refit=True)
    gs.fit(train[rfm.RF_COLS], train['d2f'])
    tg['pred_rf'] = gs.predict(tg[rfm.RF_COLS])
    tg['pred_flower_date'] = (tg['sd'] + pd.to_timedelta(np.round(tg['pred_d2f']), unit='D')).dt.strftime('%Y-%m-%d')

    met = pd.read_csv(CV_METRICS)
    rmae = met[(met['group_by'] == 'region') & (met['variant'] == 'incl_D') & (met['scheme'] == 'group')
               & (met['model'] == 'seed_reg')].set_index('region')['MAE']
    tg['region_mae'] = tg['region'].map(rmae)

    out = tg[OUT_COLS].copy()
    for c in ['pred_d2f', 'pred_rf', 'pred_stat', 'region_mae']:
        out[c] = out[c].round(2)
    out.sort_values(['year', 'place']).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # ---- 確認
    md = gdd.md_table
    # (1) 播種日が学習データの範囲(地域ごと)から外れる件
    rng = train.groupby('region')['seed_doy'].agg(['min', 'max', 'size'])
    tg['train_min'] = tg['region'].map(rng['min'])
    tg['train_max'] = tg['region'].map(rng['max'])
    tg['n_train_region'] = tg['region'].map(rng['size'])
    doy2s = lambda d: (pd.Timestamp('2017-01-01') + pd.Timedelta(days=int(d) - 1)).strftime('%m-%d')  # noqa: E731
    tg['out_region'] = (tg['seed_doy'] < tg['train_min']) | (tg['seed_doy'] > tg['train_max'])
    tg['out_all'] = (tg['seed_doy'] < train['seed_doy'].min()) | (tg['seed_doy'] > train['seed_doy'].max())
    oor = tg[tg['out_region']].copy()
    oor['地域の学習範囲'] = oor['train_min'].map(doy2s) + '〜' + oor['train_max'].map(doy2s)
    oor['外れ'] = np.where(oor['seed_doy'] > oor['train_max'], '遅い', '早い')
    oor['外れの日数'] = np.where(oor['seed_doy'] > oor['train_max'], oor['seed_doy'] - oor['train_max'],
                             oor['train_min'] - oor['seed_doy'])
    # (2) モデル間の食い違い
    P = tg[['pred_d2f', 'pred_rf', 'pred_stat']]
    tg['spread'] = P.max(axis=1) - P.min(axis=1)
    dis = tg[tg['spread'] >= DISAGREE_DAYS]
    # (3) 土壌水分
    conn = sqlite3.connect(f'file:{gdd.DB}?mode=ro', uri=True)
    hv = pd.read_sql('SELECT field_id, harvest_date, harvest_date_raw FROM Questionaire WHERE field_id IS NOT NULL', conn)
    conn.close()
    harvest = dict(zip(hv['field_id'].astype(int), hv['harvest_date_raw']))
    cov = soil_coverage(tg, harvest)
    tg = tg.merge(cov[['field_id', 'n_days_ok', 'full', 'before_harvest']], on='field_id')
    # (4) D1〜D4・2015 を除いて学習した場合の差
    b_ex = fit_seed_reg(train[~train['flag_D2015']])
    diff_ex = rfm.X_seed(tg) @ b_ex - tg['pred_d2f']

    cols_o = ['field_id', 'place', 'year', 'breed', 'seed_date_used', '地域の学習範囲', 'n_train_region',
              '外れ', '外れの日数', 'pred_d2f', 'pred_rf', 'pred_stat', 'pred_flower_date']
    o78 = tg[tg['field_id'].isin([568, 571])]
    L = ['# 64 件の開花日予測の確認', '',
         '- スクリプト: `rocket/scripts/predict_flowering_targets.py`、予測表: `rocket/data/processed/flowering_pred_64.csv`',
         f'- 採用モデル: 播種日回帰 d2f = {b[0]:.3f} {b[1]:+.4f} × 播種日の通日(学習 {len(train)} 件、D1〜D4・2015 を含める)',
         f'- 比較: RF(ハイパーパラメータ {gs.best_params_})、統計モデル',
         f'- seed_imputed = True の件: {int(tg["seed_imputed"].sum())} 件', '',
         '## 1. 播種日が学習データの範囲から外れる件', '',
         f'- 学習データ全体の播種日の範囲: {doy2s(train["seed_doy"].min())}〜{doy2s(train["seed_doy"].max())}。'
         f'これより外れる件: {int(tg["out_all"].sum())} 件',
         f'- 地域ごとの範囲から外れる件: {len(oor)} 件', '']
    if len(oor):
        L += [md(oor[cols_o].sort_values(['place', 'year']), '.1f'), '']
    L += ['O7・O8・2016(date_flag = harvest_next_year)の確認:', '',
          md(o78.assign(地域の学習範囲=o78['train_min'].map(doy2s) + '〜' + o78['train_max'].map(doy2s))
             [['field_id', 'place', 'year', 'breed', 'seed_date_used', '地域の学習範囲', 'pred_d2f', 'pred_rf',
               'pred_stat', 'pred_flower_date', 'n_days_ok', 'full']], '.1f'), '',
          '## 2. 予測が 5 日以上食い違う件(播種日回帰・RF・統計モデル)', '',
          f'{len(dis)} 件 / {len(tg)} 件', '']
    if len(dis):
        L += [md(dis[['field_id', 'place', 'year', 'breed', 'seed_date_used', 'pred_d2f', 'pred_rf', 'pred_stat',
                      'spread', 'out_region']].sort_values('spread', ascending=False), '.1f'), '']
    L += ['## 3. 土壌水分のそろい方(予測開花日 〜 +59 日の 60 日)', '',
          f'- 60 日すべてに VWC がある件: **{int(tg["full"].sum())} 件** / {len(tg)} 件',
          f'- そのうち、+59 日が収穫日(harvest_date_raw)以前の件: {int((tg["full"] & tg["before_harvest"]).sum())} 件'
          f'(harvest_date_raw が空の件は数えていない: {int(tg["field_id"].map(harvest).isna().sum())} 件)', '',
          md(tg.groupby(['year', 'region']).agg(n=('field_id', 'size'), full=('full', 'sum'),
                                                full_before_harvest=('before_harvest', lambda s: int((s & tg.loc[s.index, 'full']).sum())))
             .reset_index()), '',
          'そろわない件:', '',
          md(tg[~tg['full']].merge(cov[['field_id', 'vwc_first', 'vwc_last']], on='field_id')
             [['field_id', 'place', 'year', 'pred_flower_date', 'n_days_ok', 'vwc_first', 'vwc_last']]
             .sort_values(['year', 'place'])), '',
          '## 4. D1〜D4・2015 を除いて学習した場合', '',
          f'- 係数: {b_ex[0]:.3f} {b_ex[1]:+.4f} × 播種日の通日',
          f'- 64 件の予測の差(除く − 含める): {diff_ex.min():+.2f}〜{diff_ex.max():+.2f} 日'
          f'(四捨五入した開花日が変わる件: {int((np.round(tg["pred_d2f"] + diff_ex) != np.round(tg["pred_d2f"])).sum())} 件)', '']
    with open(OUT_MD, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'保存: {OUT_CSV}\n保存: {OUT_MD}')


if __name__ == '__main__':
    main()
