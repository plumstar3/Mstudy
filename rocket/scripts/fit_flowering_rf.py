"""
大豆の開花日予測: RF 回帰(中村 2026 卒論の再現)と、交差検証の方式による比較。

RF: 入力 = 緯度・播種日の通日・品種(one-hot、欠損は「不明」の 1 カテゴリ。名寄せはしない)、
    目的変数 = d2f(播種〜開花の日数)。RandomForestRegressor(n_estimators=300)。
    ハイパーパラメータは外側の各 fold の学習データの中だけで、外側と同じ方式の内側 CV により
    GridSearchCV(RMSE 最小)で決める。
交差検証(外側):
  kfold : KFold(5, shuffle=True, random_state=0)
  group : GroupKFold(5)、グループ = 地域(place の先頭文字)× 年
  loyo  : leave-one-year-out
内側: kfold → KFold(5, shuffle=True, random_state=0)、group → GroupKFold(5)(地域×年)、
      loyo → 残りの年での leave-one-year-out
比較するモデル(3 方式すべてで): 播種日回帰、統計モデル(播種日 + 緯度 + 40 日平均気温)、GDD (a)
光熱モデルは LOYO だけ(手順3の outputs/flowering_photothermal/cv_predictions.csv を読んで並べる)。

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db、rocket/data/processed/daylength.csv、
                    rocket/outputs/flowering_photothermal/cv_predictions.csv
出力: rocket/outputs/flowering_rf/
  params.csv               RF で選ばれたハイパーパラメータ(方式 × fold)
  cv_predictions.csv       件ごとの予測(方式ごと)
  cv_metrics.csv           RMSE, MAE, バイアス(全体・fold 別・地域別・品種別)
  rmse_by_cv.png           モデル × CV 方式の RMSE
  pred_vs_obs.png          RF と播種日回帰の予測値と実測値(方式ごと)
  residual_diagnostics.png RF の残差と緯度・播種日・日長(方式ごと)
  summary.md               要約
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import fit_flowering_gdd as gdd  # noqa: E402  データの読み込み・GDD (a)・図の体裁を再利用

PT_PRED = os.path.join(ROCKET_DIR, 'outputs', 'flowering_photothermal', 'cv_predictions.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'flowering_rf')

SEED = 0
N_SPLITS = 5
STAT_DAYS = 40
RF_GRID = {'rf__max_depth': [None, 5, 10],
           'rf__min_samples_leaf': [1, 3, 5, 10],
           'rf__max_features': [1.0, 'sqrt', 0.5]}
SCHEMES = ['kfold', 'group', 'loyo', 'loro']
SCHEME_LABEL = {'kfold': '① KFold(5, shuffle)', 'group': '② GroupKFold(5, 地域×年)', 'loyo': '③ LOYO',
                'loro': '④ leave-one-region-out'}
THESIS_MAE = 2.45  # 中村 2026 卒論の MAE(日)
MODELS = ['rf', 'seed_reg', 'stat', 'gdd_a']
PT_MODELS = ['pt_sun', 'pt_sun_kb']
LABEL = {'rf': 'RF', 'seed_reg': '播種日回帰', 'stat': '統計モデル', 'gdd_a': 'GDD (a)',
         'pt_sun': '光熱(共通, sun)', 'pt_sun_kb': '光熱(K 品種別, sun)'}
COLORS = {'rf': '#6250d6', 'seed_reg': '#eb6834', 'stat': '#1baf7a', 'gdd_a': '#2a78d6',
          'pt_sun': '#eda100', 'pt_sun_kb': '#e87ba4'}
FOCUS_REGIONS = ['O', 'N', 'C']


# ---------------------------------------------------------------- 分割
def outer_splits(scheme, cases):
    if scheme == 'kfold':
        return list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(cases))
    if scheme == 'group':
        return list(GroupKFold(N_SPLITS).split(cases, groups=cases['region_year']))
    if scheme == 'loro':
        return list(LeaveOneGroupOut().split(cases, groups=cases['region']))
    return list(LeaveOneGroupOut().split(cases, groups=cases['year']))


def inner_cv(scheme, tr):
    """内側の CV(外側と同じ方式)。GridSearchCV に渡す分割のリスト。"""
    if scheme == 'kfold':
        return list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr))
    if scheme == 'group':
        return list(GroupKFold(N_SPLITS).split(tr, groups=tr['region_year']))
    if scheme == 'loro':
        return list(LeaveOneGroupOut().split(tr, groups=tr['region']))
    return list(LeaveOneGroupOut().split(tr, groups=tr['year']))


# ---------------------------------------------------------------- モデル
RF_COLS = ['lat', 'seed_doy', 'breed_rf']


def rf_pipeline():
    pre = ColumnTransformer([('breed', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['breed_rf'])],
                            remainder='passthrough')
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1)
    return Pipeline([('pre', pre), ('rf', rf)])


def fit_predict_rf(scheme, tr, te):
    gs = GridSearchCV(rf_pipeline(), RF_GRID, scoring='neg_root_mean_squared_error',
                      cv=inner_cv(scheme, tr), n_jobs=-1, refit=True)
    gs.fit(tr[RF_COLS], tr['d2f'])
    return gs.predict(te[RF_COLS]), gs.best_params_, -gs.best_score_


def linear(Xtr, ytr, Xte):
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    return Xte @ beta


def X_seed(d):
    return np.column_stack([np.ones(len(d)), d['seed_doy']])


def X_stat(d):
    return np.column_stack([np.ones(len(d)), d['seed_doy'], d['lat'], d['tmean40']])


def predict_gdd_a(Gs, tr_i, te_i, d2f):
    tb, K, _ = gdd.fit_common(Gs[:, tr_i], d2f[tr_i])
    return gdd.predict_d2f(Gs[int(np.argmin(np.abs(gdd.TB_GRID - tb)))][te_i], K)


# ---------------------------------------------------------------- 評価
def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def metrics(cv, by=None):
    rows = []
    for scheme, d0 in cv.groupby('scheme', sort=False):
        models = MODELS + (PT_MODELS if scheme == 'loyo' else [])
        groups = [('全体', d0)] if by is None else list(d0.groupby(by))
        for g, d in groups:
            for m in models:
                e = (d[f'pred_{m}'] - d['d2f']).dropna()
                rows.append({'scheme': scheme, 'group_by': by or 'all', 'group': g, 'model': m,
                             'n': len(e), 'RMSE': rmse(e), 'MAE': float(e.abs().mean()),
                             'bias': float(e.mean())})
    return pd.DataFrame(rows)


def leak_share(cases, splits):
    """テスト件のうち、学習側に同じ地域×年×播種日の件がある割合。"""
    key = cases['region_year'] + '_' + cases['seed_date']
    hit = 0
    for tr_i, te_i in splits:
        hit += key.iloc[te_i].isin(set(key.iloc[tr_i])).sum()
    return hit / len(cases)


# ---------------------------------------------------------------- 図
def plot_rmse(met, path):
    d = met[met['group_by'] == 'all']
    models = MODELS + PT_MODELS
    fig, ax = plt.subplots(figsize=(9, 4.8))
    gdd.style(ax)
    ys = np.arange(len(SCHEMES))
    off = np.linspace(-0.3, 0.3, len(models))
    for k, m in enumerate(models):
        v = [d[(d['scheme'] == s) & (d['model'] == m)]['RMSE'] for s in SCHEMES]
        xs = [x.iloc[0] if len(x) else np.nan for x in v]
        ax.scatter(xs, ys + off[k], s=60, color=COLORS[m], edgecolor=gdd.SURF, lw=1.5,
                   label=LABEL[m], zorder=3)
        for x, y in zip(xs, ys + off[k]):
            if np.isfinite(x):
                ax.text(x + 0.08, y, f'{x:.2f}', va='center', fontsize=8, color=gdd.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([SCHEME_LABEL[s] for s in SCHEMES])
    ax.invert_yaxis()
    ax.set_xlabel('RMSE [日]')
    ax.set_title('交差検証の方式ごとの RMSE(光熱モデルは LOYO のみ)', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_obs(cv, path):
    models = ['rf', 'seed_reg']
    fig, axes = plt.subplots(len(models), len(SCHEMES), figsize=(17, 8.6), sharex=True, sharey=True)
    lo, hi = cv['d2f'].min() - 5, cv['d2f'].max() + 5
    for i, m in enumerate(models):
        for j, s in enumerate(SCHEMES):
            ax = axes[i, j]
            gdd.style(ax)
            d = cv[cv['scheme'] == s]
            e = d[f'pred_{m}'] - d['d2f']
            ax.scatter(d['d2f'], d[f'pred_{m}'], s=12, color=COLORS[m], alpha=0.55,
                       edgecolor=gdd.SURF, lw=0.4)
            ax.plot([lo, hi], [lo, hi], color=gdd.INK, lw=1, ls='--')
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f'{LABEL[m]} {SCHEME_LABEL[s]}\nRMSE {rmse(e):.2f} 日, バイアス {e.mean():+.2f} 日',
                         loc='left', fontsize=10)
            if i == len(models) - 1:
                ax.set_xlabel('実測 d2f [日]')
            if j == 0:
                ax.set_ylabel('予測 d2f [日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_corr(cv):
    rows = []
    for s, d in cv.groupby('scheme', sort=False):
        for m in MODELS + (PT_MODELS if s == 'loyo' else []):
            e = d[f'pred_{m}'] - d['d2f']
            for x, _ in gdd.DIAG:
                r, p = stats.pearsonr(d[x], e)
                rows.append({'scheme': s, 'model': m, 'x': x, 'pearson_r': r, 'pearson_p': p})
    return pd.DataFrame(rows)


def plot_residuals(cv, corr, path):
    fig, axes = plt.subplots(len(SCHEMES), 3, figsize=(15, 14.5), sharey=True)
    for i, s in enumerate(SCHEMES):
        d = cv[cv['scheme'] == s]
        e = d['pred_rf'] - d['d2f']
        for j, (x, xl) in enumerate(gdd.DIAG):
            ax = axes[i, j]
            gdd.style(ax)
            ax.scatter(d[x], e, s=10, color=COLORS['rf'], alpha=0.5, edgecolor=gdd.SURF, lw=0.4)
            ax.axhline(0, color=gdd.INK, lw=1, ls='--')
            b1, b0 = np.polyfit(d[x], e, 1)
            xs = np.linspace(d[x].min(), d[x].max(), 10)
            ax.plot(xs, b0 + b1 * xs, color='#0b0b0b', lw=1.5)
            c = corr[(corr['scheme'] == s) & (corr['model'] == 'rf') & (corr['x'] == x)].iloc[0]
            ax.set_title(f'RF {SCHEME_LABEL[s]}  r = {c.pearson_r:+.2f}(p = {c.pearson_p:.2g})',
                         loc='left', fontsize=10)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel('残差(予測 − 実測)[日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    q, w, sm, dl = gdd.load()
    cases, T = gdd.build_cases(q, w, dl)
    cases['tmean40'] = T[:, :STAT_DAYS].mean(axis=1)
    # 地域 = place の先頭文字。2018 年のゼロ埋め(A01 など)は先頭文字に影響しないことを確かめる
    norm = cases['place'].str.replace(r'^([A-Z])0*(\d+)$', r'\1\2', regex=True)
    assert (norm.str[0] == cases['region']).all()
    cases['region_year'] = cases['region'] + '_' + cases['year'].astype(str)
    # gdd.load() で品種の欠損は '(欠損)' になっている。RF では「不明」の 1 カテゴリとして扱う
    cases['breed_rf'] = cases['breed'].replace('(欠損)', '不明')
    Gs = gdd.cum_gdd(T)
    d2f = cases['d2f'].values

    preds, params, leaks = [], [], {}
    for s in SCHEMES:
        splits = outer_splits(s, cases)
        leaks[s] = leak_share(cases, splits)
        for k, (tr_i, te_i) in enumerate(splits):
            tr, te = cases.iloc[tr_i], cases.iloc[te_i].copy()
            te['scheme'], te['fold'] = s, k
            te['pred_rf'], best, inner = fit_predict_rf(s, tr, te)
            te['pred_seed_reg'] = linear(X_seed(tr), tr['d2f'].values, X_seed(te))
            te['pred_stat'] = linear(X_stat(tr), tr['d2f'].values, X_stat(te))
            te['pred_gdd_a'] = predict_gdd_a(Gs, tr_i, te_i, d2f)
            params.append({'scheme': s, 'fold': k, 'n_train': len(tr), 'n_test': len(te),
                           'test_years': ','.join(map(str, sorted(te['year'].unique()))),
                           **{k2.replace('rf__', ''): v for k2, v in best.items()},
                           'inner_rmse': inner})
            preds.append(te)
            print(f'{s} fold {k} 完了({time.time() - t0:.0f} 秒)', flush=True)

    cv = pd.concat(preds, ignore_index=True)
    # 光熱モデル(LOYO のみ、手順3の予測)を同じ件に結合する
    pt = pd.read_csv(PT_PRED, usecols=['field_id', 'year'] + [f'pred_{m}' for m in PT_MODELS])
    cv = cv.merge(pt.assign(scheme='loyo'), on=['field_id', 'year', 'scheme'], how='left')
    params = pd.DataFrame(params)
    met = pd.concat([metrics(cv)] + [metrics(cv, by) for by in ['fold', 'region', 'breed']],
                    ignore_index=True)
    corr = residual_corr(cv)

    params.to_csv(os.path.join(OUT_DIR, 'params.csv'), index=False, encoding='utf-8-sig')
    cv[['scheme', 'fold', 'field_id', 'place', 'region', 'year', 'breed', 'lat', 'seed_date',
        'flower_date', 'd2f', 'seed_doy', 'mean_daylength', 'tmean40']
       + [f'pred_{m}' for m in MODELS + PT_MODELS]].to_csv(
        os.path.join(OUT_DIR, 'cv_predictions.csv'), index=False, encoding='utf-8-sig')
    met.to_csv(os.path.join(OUT_DIR, 'cv_metrics.csv'), index=False, encoding='utf-8-sig')
    plot_rmse(met, os.path.join(OUT_DIR, 'rmse_by_cv.png'))
    plot_pred_obs(cv, os.path.join(OUT_DIR, 'pred_vs_obs.png'))
    plot_residuals(cv, corr, os.path.join(OUT_DIR, 'residual_diagnostics.png'))

    # ---- 要約
    md = gdd.md_table
    allm = met[met['group_by'] == 'all']
    L = ['# 開花日予測:RF 回帰(中村 2026 卒論の再現)と交差検証の方式の比較', '',
         '- スクリプト: `rocket/scripts/fit_flowering_rf.py`',
         f'- 対象: 開花日のある {len(cases)} 件。目的変数 d2f(播種〜開花の日数)',
         '- RF: 入力 = 緯度・播種日の通日・品種(one-hot、欠損は「不明」。名寄せなし)。'
         'RandomForestRegressor(n_estimators=300, random_state=0)',
         f'- RF のハイパーパラメータ: 外側の各 fold の学習データで、外側と同じ方式の内側 CV による GridSearchCV(RMSE)。'
         f'候補 max_depth {RF_GRID["rf__max_depth"]}, min_samples_leaf {RF_GRID["rf__min_samples_leaf"]}, '
         f'max_features {RF_GRID["rf__max_features"]}',
         '- CV: ① KFold(5, shuffle=True, random_state=0)、② GroupKFold(5)(グループ = 地域 × 年、'
         f'{cases["region_year"].nunique()} グループ)、③ leave-one-year-out',
         '- 比較: 播種日回帰・統計モデル・GDD (a) は 3 方式とも同じ分割で再計算。光熱モデルは手順3の LOYO の予測を転記', '',
         '## 1. CV の方式ごとの指標(全体)', '']
    for stat_name in ['RMSE', 'MAE', 'bias']:
        t = allm.pivot_table(index='model', columns='scheme', values=stat_name)[SCHEMES]
        t = t.reindex(MODELS + PT_MODELS)
        t.index = [LABEL[m] for m in t.index]
        t.columns = [SCHEME_LABEL[c] for c in t.columns]
        L += [f'### {stat_name}', '', md(t.reset_index().rename(columns={'index': 'モデル'})), '']
    L += ['### MAE の比較(RF と播種日回帰、卒論の MAE 2.45 日と比べる)', '',
          '| 方式 | RF | 播種日回帰 | RF − 卒論 |', '|---|---|---|---|']
    for s in SCHEMES:
        a = allm[allm['scheme'] == s].set_index('model')['MAE']
        L.append(f'| {SCHEME_LABEL[s]} | {a["rf"]:.2f} | {a["seed_reg"]:.2f} | {a["rf"] - THESIS_MAE:+.2f} |')
    L += ['']
    L += ['### 分割の漏れ(テスト件のうち、学習側に同じ地域×年×播種日の件がある割合)', '',
          '| 方式 | 割合 |', '|---|---|'] + [f'| {SCHEME_LABEL[s]} | {leaks[s]:.1%} |' for s in SCHEMES] + ['']

    L += ['## 2. fold ごとの RMSE', '']
    for s in SCHEMES:
        d = met[(met['group_by'] == 'fold') & (met['scheme'] == s)]
        t = d.pivot_table(index='group', columns='model', values='RMSE')[MODELS]
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', d[d['model'] == 'rf'].set_index('group')['n'])
        t = t.astype(object)
        t.loc['平均±SD'] = [np.nan] + [f'{t[c].mean():.2f}±{t[c].std():.2f}' for c in t.columns[1:]]
        t = t.reset_index().rename(columns={'group': 'fold'})
        t['n'] = t['n'].map(lambda v: '' if pd.isna(v) else str(int(v)))
        L += [f'### {SCHEME_LABEL[s]}', '', md(t), '']

    L += ['## 3. leave-one-region-out の地域ごとの指標(テスト地域 = その地域の全件)', '']
    d = met[(met['group_by'] == 'region') & (met['scheme'] == 'loro')]
    for stat_name in ['RMSE', 'MAE', 'bias']:
        t = d.pivot_table(index='group', columns='model', values=stat_name)[MODELS]
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', d[d['model'] == 'rf'].set_index('group')['n'])
        t['RF − 播種日回帰'] = t['RF'] - t['播種日回帰']
        t = t.reset_index().rename(columns={'group': '地域'})
        t['_f'] = ~t['地域'].isin(FOCUS_REGIONS)
        t = t.sort_values(['_f', '地域']).drop(columns='_f')
        L += [f'### {stat_name}', '', md(t), '']
    L += ['## 3b. 地域別(O, N, C)、方式ごと', '']
    d = met[(met['group_by'] == 'region') & met['group'].isin(FOCUS_REGIONS)]
    for s in SCHEMES:
        ds = d[d['scheme'] == s]
        models = MODELS + (PT_MODELS if s == 'loyo' else [])
        t = ds.pivot_table(index='group', columns='model', values='RMSE')[models]
        b = ds.pivot_table(index='group', columns='model', values='bias')[models]
        t = t.round(2).astype(str) + ' / ' + b.round(2).map('{:+.2f}'.format)
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', ds[ds['model'] == 'rf'].set_index('group')['n'])
        L += [f'### {SCHEME_LABEL[s]}(RMSE / バイアス)', '', md(t.reset_index().rename(columns={'group': '地域'})), '']

    L += ['## 4. 品種別の RMSE(15 件以上)', '']
    d = met[met['group_by'] == 'breed']
    for s in SCHEMES:
        ds = d[d['scheme'] == s]
        t = ds.pivot_table(index='group', columns='model', values='RMSE')[MODELS]
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', ds[ds['model'] == 'rf'].set_index('group')['n'])
        t = t[t['n'] >= gdd.MIN_N].sort_values('n', ascending=False)
        L += [f'### {SCHEME_LABEL[s]}', '', md(t.reset_index().rename(columns={'group': '品種'})), '']

    L += ['## 5. RF で選ばれたハイパーパラメータ', '',
          md(params.assign(max_depth=params['max_depth'].map(lambda v: 'None' if pd.isna(v) else str(int(v))),
                           max_features=params['max_features'].astype(str)), '.2f'), '',
          '`inner_rmse` は内側 CV での最良の RMSE。', '']

    rc = corr[corr['model'].isin(['rf', 'seed_reg'])].copy()
    rc['model'] = rc['model'].map(LABEL)
    rc['scheme'] = rc['scheme'].map(SCHEME_LABEL)
    rc['pearson_p'] = rc['pearson_p'].map('{:.2g}'.format)
    L += ['## 6. 残差の相関(RF と播種日回帰)', '', md(rc, '.3f'), '',
          '全モデルの値は `residual_diagnostics.png`(RF のみ図示)と本スクリプトの `residual_corr()` を参照。', '']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'保存先: {OUT_DIR}({time.time() - t0:.0f} 秒)')


if __name__ == '__main__':
    main()
