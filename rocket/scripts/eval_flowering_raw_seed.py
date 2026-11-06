"""
開花日予測を、補完前の播種日(Questionaire.seed_date_raw)で評価し直す。

学習データ: 開花日があり、seed_date_raw もある件(463 件)。補完済みの seed_date は使わない。
  D1〜D4・2015(ミヤギシロメ、d2f 72〜75 日、date_flag = long_seed_to_flower)は
  含める(incl_D)/ 除外する(excl_D)の両方で評価する。
モデル: 播種日回帰 / 統計モデル(播種日 + 緯度 + 40 日平均気温)/ GDD (a) / RF(緯度・播種日・品種)
  部品は scripts/fit_flowering_rf.py・fit_flowering_gdd.py のものをそのまま使う。
交差検証: メイン GroupKFold(5)(地域×年)、確認用 leave-one-region-out。
  RF のハイパーパラメータは外側の各 fold の学習データで、外側と同じ方式の内側 CV で決める。
指標: MAE(主)、RMSE、バイアス、±3 日以内・±7 日以内の割合。全体と地域別。
比較: 補完値を含んでいた EXP-015(outputs/flowering_rf/cv_metrics.csv)と並べ、差(今回 − EXP-015)を出す。

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db、rocket/data/processed/daylength.csv、
                    rocket/outputs/flowering_rf/cv_metrics.csv
出力: rocket/outputs/flowering_raw_seed/
  params.csv, cv_predictions.csv, cv_metrics.csv, comparison_exp015.md, mae_by_model.png, summary.md
"""
import os
import sqlite3
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import fit_flowering_gdd as gdd  # noqa: E402
import fit_flowering_rf as rfm  # noqa: E402

EXP015_METRICS = os.path.join(ROCKET_DIR, 'outputs', 'flowering_rf', 'cv_metrics.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'flowering_raw_seed')

SCHEMES = ['group', 'loro']
SCHEME_LABEL = {'group': 'GroupKFold(5, 地域×年)', 'loro': 'leave-one-region-out'}
MODELS = ['seed_reg', 'stat', 'gdd_a', 'rf']
LABEL = {'seed_reg': '播種日回帰', 'stat': '統計モデル', 'gdd_a': 'GDD (a)', 'rf': 'RF'}
COLORS = {'seed_reg': '#eb6834', 'stat': '#1baf7a', 'gdd_a': '#2a78d6', 'rf': '#6250d6'}
VARIANTS = ['incl_D', 'excl_D']
VARIANT_LABEL = {'incl_D': 'D1〜D4・2015 を含める', 'excl_D': 'D1〜D4・2015 を除く'}
D_PLACES = ['D1', 'D2', 'D3', 'D4']
FOCUS_REGIONS = ['O', 'N', 'C']


def load_raw():
    """gdd.load() と同じ形で、seed_date を seed_date_raw に差し替えたデータを返す。"""
    q, w, sm, dl = gdd.load()
    conn = sqlite3.connect(f'file:{gdd.DB}?mode=ro', uri=True)
    raw = pd.read_sql('SELECT field_id, year, seed_date_raw, date_flag FROM Questionaire '
                      'WHERE field_id IS NOT NULL', conn)
    conn.close()
    raw['field_id'] = raw['field_id'].astype(int)
    q = q.merge(raw, on=['field_id', 'year'], how='left', validate='one_to_one')
    q['seed_date_imputed'] = q['seed_date']
    q['seed_date'] = q['seed_date_raw']
    return q, w, sm, dl


def build(q, w, dl):
    """開花日と seed_date_raw のある件の学習データ(gdd.build_cases に渡す)。"""
    qq = q[q['flower_date'].notna() & q['seed_date'].notna()]
    cases, T = gdd.build_cases(qq, w, dl)
    cases['tmean40'] = T[:, :rfm.STAT_DAYS].mean(axis=1)
    cases['region_year'] = cases['region'] + '_' + cases['year'].astype(str)
    cases['breed_rf'] = cases['breed'].replace('(欠損)', '不明')
    cases['flag_D2015'] = cases['place'].isin(D_PLACES) & (cases['year'] == 2015)
    return cases, T


def run_cv(cases, T, scheme):
    Gs = gdd.cum_gdd(T)
    d2f = cases['d2f'].values
    preds, params = [], []
    for k, (tr_i, te_i) in enumerate(rfm.outer_splits(scheme, cases)):
        tr, te = cases.iloc[tr_i], cases.iloc[te_i].copy()
        te['fold'] = k
        te['pred_rf'], best, inner = rfm.fit_predict_rf(scheme, tr, te)
        b_seed = np.linalg.lstsq(rfm.X_seed(tr), tr['d2f'].values, rcond=None)[0]
        te['pred_seed_reg'] = rfm.X_seed(te) @ b_seed
        te['pred_stat'] = rfm.linear(rfm.X_stat(tr), tr['d2f'].values, rfm.X_stat(te))
        te['pred_gdd_a'] = rfm.predict_gdd_a(Gs, tr_i, te_i, d2f)
        params.append({'fold': k, 'n_train': len(tr), 'n_test': len(te),
                       'test_regions': ','.join(sorted(te['region'].unique())),
                       'seed_b0': b_seed[0], 'seed_b1': b_seed[1],
                       **{k2.replace('rf__', ''): v for k2, v in best.items()}, 'rf_inner_rmse': inner})
        preds.append(te)
    return pd.concat(preds), pd.DataFrame(params)


def metrics(cv, by=None):
    rows = []
    keys = ['variant', 'scheme'] + ([by] if by else [])
    for key, d in cv.groupby(keys, sort=False):
        for m in MODELS:
            e = (d[f'pred_{m}'] - d['d2f']).dropna()
            r = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            r.update({'group_by': by or 'all', 'model': m, 'n': len(e),
                      'MAE': float(e.abs().mean()), 'RMSE': float(np.sqrt((e ** 2).mean())),
                      'bias': float(e.mean()), 'within3': float((e.abs() <= 3).mean()),
                      'within7': float((e.abs() <= 7).mean())})
            rows.append(r)
    return pd.DataFrame(rows)


def plot_mae(met, path):
    d = met[met['group_by'] == 'all']
    combos = [(v, s) for s in SCHEMES for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    gdd.style(ax)
    ys = np.arange(len(combos))
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for k, m in enumerate(MODELS):
        xs = [d[(d['variant'] == v) & (d['scheme'] == s) & (d['model'] == m)]['MAE'].iloc[0] for v, s in combos]
        ax.scatter(xs, ys + off[k], s=55, color=COLORS[m], edgecolor=gdd.SURF, lw=1.5,
                   label=LABEL[m], zorder=3)
        for x, y in zip(xs, ys + off[k]):
            ax.text(x + 0.05, y, f'{x:.2f}', va='center', fontsize=8, color=gdd.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{SCHEME_LABEL[s]}\n{VARIANT_LABEL[v]}' for v, s in combos], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('MAE [日]')
    ax.set_title('補完前の播種日での MAE(モデル × CV × D1〜D4・2015 の扱い)', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_exp015(met):
    """今回(D を含める)と EXP-015 の全体の MAE / RMSE を並べ、差を出す。"""
    old = pd.read_csv(EXP015_METRICS)
    old = old[(old['group_by'] == 'all') & old['scheme'].isin(SCHEMES) & old['model'].isin(MODELS)]
    new = met[(met['group_by'] == 'all') & (met['variant'] == 'incl_D')]
    c = new.merge(old[['scheme', 'model', 'n', 'MAE', 'RMSE', 'bias']], on=['scheme', 'model'],
                  suffixes=('', '_exp015'))
    c['dMAE'] = c['MAE'] - c['MAE_exp015']
    c['dRMSE'] = c['RMSE'] - c['RMSE_exp015']
    c['scheme'] = c['scheme'].map(SCHEME_LABEL)
    c['model'] = c['model'].map(LABEL)
    return c[['scheme', 'model', 'n_exp015', 'n', 'MAE_exp015', 'MAE', 'dMAE',
              'RMSE_exp015', 'RMSE', 'dRMSE']]


def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    q, w, sm, dl = load_raw()
    cases_all, T_all = build(q, w, dl)
    n_old = int(q['flower_date'].notna().sum())

    preds, params = [], []
    for v in VARIANTS:
        keep = np.ones(len(cases_all), bool) if v == 'incl_D' else ~cases_all['flag_D2015'].values
        cases = cases_all[keep].reset_index(drop=True)
        T = T_all[keep]
        for s in SCHEMES:
            p, prm = run_cv(cases, T, s)
            preds.append(p.assign(variant=v, scheme=s))
            params.append(prm.assign(variant=v, scheme=s))
            print(f'{v} {s} 完了({time.time() - t0:.0f} 秒)', flush=True)
    cv = pd.concat(preds, ignore_index=True)
    params = pd.concat(params, ignore_index=True)
    met = pd.concat([metrics(cv), metrics(cv, 'region')], ignore_index=True)
    comp = comparison_exp015(met)

    params.to_csv(os.path.join(OUT_DIR, 'params.csv'), index=False, encoding='utf-8-sig')
    cv[['variant', 'scheme', 'fold', 'field_id', 'place', 'region', 'year', 'breed', 'lat',
        'seed_date', 'flower_date', 'd2f', 'seed_doy', 'tmean40', 'flag_D2015']
       + [f'pred_{m}' for m in MODELS]].to_csv(os.path.join(OUT_DIR, 'cv_predictions.csv'),
                                               index=False, encoding='utf-8-sig')
    met.to_csv(os.path.join(OUT_DIR, 'cv_metrics.csv'), index=False, encoding='utf-8-sig')
    plot_mae(met, os.path.join(OUT_DIR, 'mae_by_model.png'))
    md = gdd.md_table
    with open(os.path.join(OUT_DIR, 'comparison_exp015.md'), 'w', encoding='utf-8') as fh:
        fh.write('# EXP-015(補完値を含む 465 件)と今回(seed_date_raw、D1〜D4・2015 を含める 463 件)の比較\n\n'
                 '差 = 今回 − EXP-015(日)。負なら今回のほうが誤差が小さい。\n\n' + md(comp) + '\n')

    # ---- 要約
    dropped = q[q['flower_date'].notna() & q['seed_date'].isna()]
    L = ['# 開花日予測:補完前の播種日(seed_date_raw)での再評価', '',
         '- スクリプト: `rocket/scripts/eval_flowering_raw_seed.py`',
         f'- 学習データ: 開花日と seed_date_raw のある {len(cases_all)} 件(補完値を含む従来の {n_old} 件から '
         f'{n_old - len(cases_all)} 件減: ' + ', '.join(f'{r.place}・{r.year}' for r in dropped.itertuples()) + ')',
         f'- D1〜D4・2015({int(cases_all["flag_D2015"].sum())} 件)を含める / 除く の両方で評価',
         '- CV: メイン GroupKFold(5)(地域×年)、確認用 leave-one-region-out。RF のハイパーパラメータは外側の各 fold の'
         '学習データで、外側と同じ方式の内側 CV で決めた', '',
         '## 1. 全体の指標', '']
    allm = met[met['group_by'] == 'all']
    for s in SCHEMES:
        for v in VARIANTS:
            t = allm[(allm['scheme'] == s) & (allm['variant'] == v)][
                ['model', 'n', 'MAE', 'RMSE', 'bias', 'within3', 'within7']].copy()
            t['model'] = t['model'].map(LABEL)
            t['within3'] = t['within3'].map('{:.1%}'.format)
            t['within7'] = t['within7'].map('{:.1%}'.format)
            L += [f'### {SCHEME_LABEL[s]}、{VARIANT_LABEL[v]}', '',
                  md(t.rename(columns={'within3': '±3 日以内', 'within7': '±7 日以内'})), '']
    L += ['## 2. EXP-015 との比較(D1〜D4・2015 を含める)', '', '差 = 今回 − EXP-015(日)', '', md(comp), '']
    L += ['## 3. 地域別の MAE / RMSE / バイアス(D1〜D4・2015 を含める)', '']
    rm = met[(met['group_by'] == 'region') & (met['variant'] == 'incl_D')]
    for s in SCHEMES:
        d = rm[rm['scheme'] == s]
        cell = d.assign(v=d['MAE'].map('{:.2f}'.format) + ' / ' + d['RMSE'].map('{:.2f}'.format)
                        + ' / ' + d['bias'].map('{:+.2f}'.format))
        t = cell.pivot(index='region', columns='model', values='v')[MODELS]
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', d[d['model'] == 'rf'].set_index('region')['n'])
        mae = d.pivot(index='region', columns='model', values='MAE')
        t['最小 MAE'] = mae[MODELS].idxmin(axis=1).map(LABEL)
        t = t.reset_index().rename(columns={'region': '地域'})
        t['_f'] = ~t['地域'].isin(FOCUS_REGIONS)
        t = t.sort_values(['_f', '地域']).drop(columns='_f')
        L += [f'### {SCHEME_LABEL[s]}(MAE / RMSE / バイアス)', '', md(t), '']
    L += ['## 4. RF のハイパーパラメータ・播種日回帰の係数(fold ごと)', '',
          md(params.assign(max_depth=params['max_depth'].map(lambda v: 'None' if pd.isna(v) else str(int(v))),
                           max_features=params['max_features'].astype(str))
             [['variant', 'scheme', 'fold', 'n_train', 'test_regions', 'seed_b0', 'seed_b1',
               'max_depth', 'max_features', 'min_samples_leaf', 'rf_inner_rmse']], '.3f'), '']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'保存先: {OUT_DIR}({time.time() - t0:.0f} 秒)')


if __name__ == '__main__':
    main()
