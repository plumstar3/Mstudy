"""
段階2: 時系列の中身を使わない情報だけでラベルが当たるかの確認(リーク確認と非時系列ベースライン)。

対象: label_muniyear が付いた94件のうち
  案A = dap 50–99 を覆う85件 / 案C = dap 30–89 を覆う62件
特徴量ごとに StandardScaler → LogisticRegression(L2) を評価し、置換検定も行う。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage2\\run_stage2.py [--n-perm 200]
出力: rocket/outputs/stage2/*.csv
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
from src.evaluation import evaluate, make_splits, permutation_test, split_table  # noqa: E402

ROOT_DIR = os.path.dirname(ROCKET_DIR)
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DB = os.path.join(ROOT_DIR, 'data', 'processed', 'FieldData_fieldid.db')
GDD = os.path.join(ROOT_DIR, 'outputs', 'gdd', 'gdd_daily.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage2')

WINDOWS = {'A': (50, 99), 'C': (30, 89)}  # dap の区間(両端を含む)
FEATURE_SETS = {
    'a_obs_shape': ['start_dap', 'end_dap', 'n_obs'],
    'b_has_flower': ['has_flower'],
    'c_seed_doy': ['seed_doy'],
    'd_gdd_dap50': ['gdd_dap50'],
    'e_vwc_stats': ['vwc_mean', 'vwc_sd', 'vwc_min', 'vwc_max'],
}
FEATURE_SETS['f_all'] = sum(FEATURE_SETS.values(), [])

N_SPLITS, N_REPEATS = 5, 20


def load():
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig',
                      dtype={'muniCd': str})
    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig',
                     parse_dates=['date'])
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    q = pd.read_sql('SELECT field_id, year, flower_date FROM Questionaire WHERE field_id IS NOT NULL', conn)
    conn.close()
    q['field_id'] = pd.to_numeric(q['field_id'], errors='coerce')
    q = q.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})
    q = q.drop_duplicates(['field_id', 'year'])  # build_soilmoisture_dataset.py と同じく先頭を採る
    gdd = pd.read_csv(GDD, encoding='utf-8-sig', parse_dates=['date'],
                      usecols=['field_id', 'year', 'date', '累積GDD'])
    return lab, ts, q, gdd


def build_base(lab, ts, q, gdd):
    """label_muniyear 付き94件の、窓に依存しない特徴量。"""
    d = lab[lab['label_muniyear'].notna()].copy()
    d['y'] = d['label_muniyear'].astype(int)
    d['muniyear'] = d['muniCd'] + '_' + d['year'].astype(str)
    d = d.merge(q, on=['field_id', 'year'], how='left')
    d['has_flower'] = pd.to_datetime(d['flower_date'], errors='coerce').notna().astype(int)
    d['seed_doy'] = pd.to_datetime(d['seed_date']).dt.dayofyear

    t = ts.merge(d[['field_id', 'year']], on=['field_id', 'year'])
    shape = t.groupby(['field_id', 'year'])['dap'].agg(start_dap='min', end_dap='max', n_obs='size')
    d = d.drop(columns=['n_obs']).merge(shape.reset_index(), on=['field_id', 'year'])

    seed = pd.to_datetime(d['seed_date'])
    g = gdd.merge(d[['field_id', 'year']].assign(d50=seed + pd.Timedelta(days=50)),
                  on=['field_id', 'year'])
    g = g[g['date'] == g['d50']][['field_id', 'year', '累積GDD']].rename(columns={'累積GDD': 'gdd_dap50'})
    d = d.merge(g, on=['field_id', 'year'], how='left')
    return d, t


def window_subset(base, t, lo, hi):
    """dap lo..hi の全日に観測がある件と、窓内 VWC の要約統計。"""
    w = t[(t['dap'] >= lo) & (t['dap'] <= hi)]
    st = w.groupby(['field_id', 'year'])['VWC'].agg(
        n_win='size', vwc_mean='mean', vwc_sd='std', vwc_min='min', vwc_max='max').reset_index()
    st = st[st['n_win'] == hi - lo + 1]
    return base.merge(st, on=['field_id', 'year']).reset_index(drop=True)


def holm(p):
    """Holm 法で補正した p 値。"""
    p = np.asarray(p, dtype=float)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=200)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    lab, ts, q, gdd = load()
    base, t = build_base(lab, ts, q, gdd)

    # 1. ラベル定義の確認: 市町村×年ごとの多収/低収の件数
    tabs = [base.groupby(['muniCd', 'year'])['y'].agg(n_high='sum', n='size').assign(set='all94')]
    # 4. flower_date 欠測の偏り
    miss = base[base['has_flower'] == 0]
    miss[['field_id', 'year', 'muniCd', 'y']].to_csv(
        os.path.join(OUT_DIR, 'flower_missing_cases.csv'), index=False, encoding='utf-8-sig')
    ct = pd.crosstab(base['has_flower'], base['y'])
    odds, p = fisher_exact(ct.values)
    fisher = [{'set': 'all94', 'missing_low': ct.loc[0, 0], 'missing_high': ct.loc[0, 1],
               'present_low': ct.loc[1, 0], 'present_high': ct.loc[1, 1], 'odds_ratio': odds, 'p': p}]

    summary_rows, split_rows, fold_rows = [], [], []
    for name, (lo, hi) in WINDOWS.items():
        d = window_subset(base, t, lo, hi)
        d.to_csv(os.path.join(OUT_DIR, f'features_{name}.csv'), index=False, encoding='utf-8-sig')
        tabs.append(d.groupby(['muniCd', 'year'])['y'].agg(n_high='sum', n='size').assign(set=name))
        ct = pd.crosstab(d['has_flower'], d['y']).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        odds, p = fisher_exact(ct.values)
        fisher.append({'set': name, 'missing_low': ct.loc[0, 0], 'missing_high': ct.loc[0, 1],
                       'present_low': ct.loc[1, 0], 'present_high': ct.loc[1, 1], 'odds_ratio': odds, 'p': p})

        y, groups = d['y'].values, d['muniCd'].values
        splits = make_splits(y, groups, N_SPLITS, N_REPEATS, seed0=0)
        split_rows.append(split_table(splits, y, groups).assign(set=name))
        print(f'[{name}] dap {lo}-{hi}: n={len(d)} (多収{y.sum()}/低収{len(y) - y.sum()}) 地域{len(set(groups))}')

        for fs, cols in FEATURE_SETS.items():
            X = d[cols].values.astype(float)
            assert not np.isnan(X).any(), f'{name} {fs} に欠損あり'
            res, null = permutation_test(model(), X, y, splits, d['muniyear'].values,
                                         n_perm=args.n_perm, seed=0)
            fold_rows.append(evaluate(model(), X, y, splits).assign(set=name, features=fs))
            summary_rows.append({'set': name, 'window': f'{lo}-{hi}', 'n': len(d), 'features': fs, **res})
            print(f'  {fs:14s} bacc {res["bacc_mean"]:.3f}±{res["bacc_sd_fold"]:.3f} (p={res["bacc_p"]:.3f})'
                  f'  auc {res["auc_mean"]:.3f}±{res["auc_sd_fold"]:.3f} (p={res["auc_p"]:.3f})')

    pd.concat(tabs).reset_index().to_csv(os.path.join(OUT_DIR, 'label_by_muniyear.csv'),
                                         index=False, encoding='utf-8-sig')
    pd.DataFrame(fisher).to_csv(os.path.join(OUT_DIR, 'flower_missing_fisher.csv'),
                                index=False, encoding='utf-8-sig')
    pd.concat(split_rows).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat(fold_rows)
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    folds.groupby(['set', 'features', 'repeat'])[['bacc', 'auc']].mean().reset_index().to_csv(
        os.path.join(OUT_DIR, 'repeats.csv'), index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary_rows)
    # 多重比較: 全検定(集合×特徴量×指標)をまとめた補正と、指標ごとの補正
    both = holm(np.r_[summary['bacc_p'], summary['auc_p']])
    summary['bacc_p_holm24'], summary['auc_p_holm24'] = both[:len(summary)], both[len(summary):]
    summary['bacc_p_holm12'], summary['auc_p_holm12'] = holm(summary['bacc_p']), holm(summary['auc_p'])
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(f'出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
