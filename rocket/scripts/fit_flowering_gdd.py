"""
大豆の開花日予測 手順2: 有効積算温度(GDD)モデル(ベースライン)。

モデル: 播種日の翌日から日ごとに max(TMP_mea − Tb, 0) を積算し、累積が K 以上になった
最初の日を予測開花日とする。d2f(播種〜開花の日数)で予測・評価する。
  Tb: 0〜15℃ を 0.5℃ 刻みでグリッド探索(学習データでの RMSE 最小)
  K : 学習データで「播種日の翌日〜実際の開花日」の積算値の平均
比較するモデル:
  (a) 全品種共通の (Tb, K)
  (b) 品種ごとの (Tb, K)。対象は開花日ありが 15 件以上の 6 品種。fold 内の学習件数が
      15 件未満なら (a) を使う。それ以外の品種・品種欠損も (a)
  (c) Tb は共通、K だけ品種ごと(同じ 6 品種、fold 内 15 件以上)。それ以外は学習データ全件の平均 K。
      Tb は、この組み合わせ全体の学習 RMSE が最小になる値
評価: leave-one-year-out。Tb と K は各 fold の学習データだけで決める。
残差(予測 − 実測)を、緯度・播種日の通日・播種翌日〜開花日の平均日長(day_length_sun)と比べる。

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db(Questionaire, Weather, SolidMoisture)
                    rocket/data/processed/daylength.csv(scripts/compute_daylength.py で作成)
出力: rocket/outputs/flowering_gdd/
  params.csv               fold ごと(と全データ)の Tb, K
  rmse_curve.csv / .png    学習データでの RMSE と Tb の関係
  cv_predictions.csv       LOYO での件ごとの予測
  cv_metrics.csv           RMSE, MAE, バイアス(全体・年別・品種別・地域別)
  residual_diagnostics.png 残差と緯度・播種日・日長の散布図
  pred_vs_obs.png          予測値と実測値の散布図
  summary.md               要約
"""
import os
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROCKET_DIR)
from src.phenology import first_reach  # noqa: E402

DB = os.path.join(ROCKET_DIR, 'data', 'raw', 'FieldData_fieldid.db')
DAYLENGTH = os.path.join(ROCKET_DIR, 'data', 'processed', 'daylength.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'flowering_gdd')

TB_GRID = np.arange(0, 15.01, 0.5)
MIN_N = 15          # 品種ごとにパラメータを推定する最小件数
HORIZON = 200       # 播種翌日から何日分の気温を使うか
MODELS = ['a', 'b', 'c']
MODEL_LABEL = {'a': '(a) 共通 Tb・K', 'b': '(b) 品種別 Tb・K', 'c': '(c) 共通 Tb・品種別 K'}
COLORS = {'a': '#2a78d6', 'b': '#eb6834', 'c': '#1baf7a'}
YEAR_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, GRID, SURF = '#52514e', '#e4e3df', '#fcfcfb'


# ---------------------------------------------------------------- データ
def load():
    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    q = pd.read_sql('SELECT field_id, place, year, breed, lat, yield, seed_date, flower_date '
                    'FROM Questionaire WHERE field_id IS NOT NULL', conn)
    w = pd.read_sql('SELECT field_id, date, TMP_mea FROM Weather', conn)
    sm = pd.read_sql('SELECT DISTINCT field_id, year FROM SolidMoisture', conn)
    conn.close()
    q['field_id'] = q['field_id'].astype(int)
    q['region'] = q['place'].str[0]
    q['breed'] = q['breed'].fillna('(欠損)')
    w['date'] = pd.to_datetime(w['date'])
    dl = pd.read_csv(DAYLENGTH, usecols=['field_id', 'date', 'day_length_sun'], parse_dates=['date'])
    return q, w, sm, dl


def build_cases(q, w, dl):
    """開花日のある行について、播種翌日からの気温行列 T (n, HORIZON) と付帯情報を作る。"""
    f = q[q['flower_date'].notna()].copy().reset_index(drop=True)
    f['sd'] = pd.to_datetime(f['seed_date'])
    f['fd'] = pd.to_datetime(f['flower_date'])
    f['d2f'] = (f['fd'] - f['sd']).dt.days
    f['seed_doy'] = f['sd'].dt.dayofyear
    tw = w.set_index(['field_id', 'date'])['TMP_mea']
    dls = dl.set_index(['field_id', 'date'])['day_length_sun']
    T = np.full((len(f), HORIZON), np.nan)
    mean_dl = np.empty(len(f))
    for i, r in f.iterrows():
        days = pd.date_range(r['sd'] + pd.Timedelta(days=1), periods=HORIZON)
        T[i] = tw.reindex(pd.MultiIndex.from_product([[r['field_id']], days])).values
        to_fl = pd.date_range(r['sd'] + pd.Timedelta(days=1), r['fd'])
        mean_dl[i] = dls.reindex(pd.MultiIndex.from_product([[r['field_id']], to_fl])).mean()
    f['mean_daylength'] = mean_dl
    # 播種翌日〜開花日の気温に欠けがあってはならない(手順1で確認済み)
    assert all(np.isfinite(T[i, :d]).all() for i, d in enumerate(f['d2f'])), '気温に欠けがある'
    return f, T


def cum_gdd(T):
    """Tb ごとの累積 GDD。形は (len(TB_GRID), n, HORIZON)。範囲外の欠けは 0 として扱う。"""
    Tz = np.nan_to_num(T, nan=-np.inf)
    return np.stack([np.cumsum(np.maximum(Tz - tb, 0), axis=1) for tb in TB_GRID])


# ---------------------------------------------------------------- モデル
def gdd_at_flower(G, d2f):
    """各件の、播種翌日〜開花日(d2f 日目、列 d2f-1)の積算値。G は (n, HORIZON)。"""
    return G[np.arange(len(d2f)), np.asarray(d2f) - 1]


def predict_d2f(G, K):
    """累積が K に初めて達した日(播種後日数)。HORIZON 日で達しなければ NaN。K は件ごとでもよい。"""
    K = np.broadcast_to(K, (len(G),))
    return np.array([first_reach(g, k) + 1 for g, k in zip(G, K)])


def scored(pred):
    """学習 RMSE 用。予測できなかった件は HORIZON 日として罰する。"""
    return np.where(np.isnan(pred), HORIZON, pred)


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def fit_common(Gs, d2f):
    """(Tb, K) を共通で決める。Gs は (nTb, n, H)。戻り値: Tb, K, RMSE 曲線。"""
    curve = []
    for G in Gs:
        K = gdd_at_flower(G, d2f).mean()
        curve.append((K, rmse(scored(predict_d2f(G, K)) - d2f)))
    curve = np.array(curve)
    j = int(np.argmin(curve[:, 1]))
    return TB_GRID[j], curve[j, 0], curve[:, 1]


def target_breeds(tr):
    vc = tr['breed'].value_counts()
    return [b for b in vc.index if b != '(欠損)' and vc[b] >= MIN_N]


def fit_fold(tr, Gs_tr):
    """学習データで (a)(b)(c) のパラメータを決める。"""
    d2f = tr['d2f'].values
    breeds = target_breeds(tr)
    p = {'a': {}, 'b': {}, 'c': {}}
    tb, K, curve_a = fit_common(Gs_tr, d2f)
    p['a']['_all'] = (tb, K)
    p['b']['_all'] = (tb, K)
    curves = {'a': curve_a}
    for b in breeds:
        m = (tr['breed'] == b).values
        tb_b, K_b, curves[f'b:{b}'] = fit_common(Gs_tr[:, m], d2f[m])
        p['b'][b] = (tb_b, K_b)
    # (c): Tb 共通、K は品種ごと(対象外は全件平均)
    curve_c, Ks_c = [], []
    for G in Gs_tr:
        at_fl = gdd_at_flower(G, d2f)
        Kmap = {'_all': at_fl.mean()}
        Kmap.update({b: at_fl[(tr['breed'] == b).values].mean() for b in breeds})
        Kvec = tr['breed'].map(lambda b: Kmap.get(b, Kmap['_all'])).values
        curve_c.append(rmse(scored(predict_d2f(G, Kvec)) - d2f))
        Ks_c.append(Kmap)
    j = int(np.argmin(curve_c))
    curves['c'] = np.array(curve_c)
    p['c'] = {k: (TB_GRID[j], v) for k, v in Ks_c[j].items()}
    return p, curves


def predict(params, cases, Gs):
    """モデルごとに、品種に応じたパラメータで予測する。"""
    out = {}
    for m in MODELS:
        pm = params[m]
        pred = np.empty(len(cases))
        for i, b in enumerate(cases['breed']):
            tb, K = pm.get(b, pm['_all'])
            j = int(np.argmin(np.abs(TB_GRID - tb)))
            pred[i] = predict_d2f(Gs[j, i:i + 1], K)[0]
        out[m] = pred
    return out


def param_rows(fold, params, tr):
    rows = []
    for m in MODELS:
        for b, (tb, K) in params[m].items():
            n = len(tr) if b == '_all' else int((tr['breed'] == b).sum())
            rows.append({'fold': fold, 'model': m, 'breed': '(共通)' if b == '_all' else b,
                         'n_train': n, 'Tb': tb, 'K': K})
    return rows


# ---------------------------------------------------------------- 評価
def metrics(df, by=None):
    rows = []
    groups = [('全体', df)] if by is None else list(df.groupby(by))
    for g, d in groups:
        for m in MODELS:
            e = d[f'pred_{m}'] - d['d2f']
            ok = e.notna()
            rows.append({'group_by': by or 'all', 'group': g, 'model': m, 'n': int(ok.sum()),
                         'n_unreached': int((~ok).sum()),
                         'RMSE': rmse(e[ok]) if ok.any() else np.nan,
                         'MAE': float(e[ok].abs().mean()), 'bias': float(e[ok].mean())})
    return pd.DataFrame(rows)


def flatness(curve):
    """RMSE 曲線の平らさ: 最小値 + 0.1 日以内に入る Tb の範囲と、曲線の振れ幅。"""
    near = TB_GRID[curve <= curve.min() + 0.1]
    return near.min(), near.max(), curve.max() - curve.min()


# ---------------------------------------------------------------- 図
def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(INK)
    ax.tick_params(colors=INK, labelsize=9)


def plot_rmse_curve(curve_df, path):
    breeds = [k for k in curve_df.columns if k.startswith('b:')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, key, title in [(axes[0], 'a', MODEL_LABEL['a']), (axes[1], 'c', MODEL_LABEL['c'])]:
        style(ax)
        for k, (fold, d) in enumerate(curve_df[curve_df['fold'] != 'all'].groupby('fold')):
            ax.plot(d['Tb'], d[key], color=YEAR_COLORS[k], lw=1.5, label=f'{fold}年を除く')
        d = curve_df[curve_df['fold'] == 'all']
        ax.plot(d['Tb'], d[key], color='#0b0b0b', lw=2.2, label='全データ')
        ax.set_title(f'{title}:学習 RMSE', loc='left', fontsize=11)
        ax.set_xlabel('Tb [℃]')
        ax.set_ylabel('学習 RMSE [日]')
        ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    style(ax)
    d = curve_df[curve_df['fold'] == 'all']
    pal = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
    for k, b in enumerate(breeds):
        ax.plot(d['Tb'], d[b], color=pal[k % len(pal)], lw=1.8, label=b[2:])
    ax.set_title('(b) 品種別:学習 RMSE(全データ)', loc='left', fontsize=11)
    ax.set_xlabel('Tb [℃]')
    ax.set_ylabel('学習 RMSE [日]')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_obs(cv, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    lo, hi = cv['d2f'].min() - 5, cv['d2f'].max() + 5
    for ax, m in zip(axes, MODELS):
        style(ax)
        e = cv[f'pred_{m}'] - cv['d2f']
        ax.scatter(cv['d2f'], cv[f'pred_{m}'], s=14, color=COLORS[m], alpha=0.55,
                   edgecolor=SURF, lw=0.4)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f'{MODEL_LABEL[m]}\nRMSE {rmse(e.dropna()):.2f} 日, バイアス {e.mean():+.2f} 日',
                     loc='left', fontsize=10)
        ax.set_xlabel('実測 d2f [日]')
    axes[0].set_ylabel('予測 d2f(LOYO)[日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


DIAG = [('lat', '緯度 [°]'), ('seed_doy', '播種日の通日'), ('mean_daylength', '播種〜開花の平均日長 [h]')]


def residual_corr(cv):
    rows = []
    for m in MODELS:
        e = cv[f'pred_{m}'] - cv['d2f']
        for x, _ in DIAG:
            ok = e.notna()
            r, pr = stats.pearsonr(cv.loc[ok, x], e[ok])
            rho, ps = stats.spearmanr(cv.loc[ok, x], e[ok])
            slope = np.polyfit(cv.loc[ok, x], e[ok], 1)[0]
            rows.append({'model': m, 'x': x, 'pearson_r': r, 'pearson_p': pr,
                         'spearman_rho': rho, 'spearman_p': ps, 'slope': slope})
    return pd.DataFrame(rows)


def plot_residuals(cv, corr, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey='row')
    for i, m in enumerate(MODELS):
        e = cv[f'pred_{m}'] - cv['d2f']
        for j, (x, xl) in enumerate(DIAG):
            ax = axes[i, j]
            style(ax)
            ax.scatter(cv[x], e, s=12, color=COLORS[m], alpha=0.5, edgecolor=SURF, lw=0.4)
            ax.axhline(0, color=INK, lw=1, ls='--')
            ok = e.notna()
            b1, b0 = np.polyfit(cv.loc[ok, x], e[ok], 1)
            xs = np.linspace(cv[x].min(), cv[x].max(), 10)
            ax.plot(xs, b0 + b1 * xs, color='#0b0b0b', lw=1.5)
            c = corr[(corr['model'] == m) & (corr['x'] == x)].iloc[0]
            ax.set_title(f'{MODEL_LABEL[m]}  r = {c.pearson_r:+.2f}(p = {c.pearson_p:.2g})',
                         loc='left', fontsize=10)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel('残差(予測 − 実測)[日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 要約
def md_table(df, floatfmt='.2f'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(format(v, floatfmt) if isinstance(v, float) else str(v)
                                       for v in r) + ' |')
    return '\n'.join(lines)


def pivot_metrics(met, by):
    d = met[met['group_by'] == by]
    t = d.pivot_table(index='group', columns='model', values='RMSE')
    t.columns = [f'RMSE_{c}' for c in t.columns]
    t.insert(0, 'n', d[d['model'] == 'a'].set_index('group')['n'])
    b = d.pivot_table(index='group', columns='model', values='bias')
    for c in b.columns:
        t[f'bias_{c}'] = b[c]
    return t.reset_index()


def no_flower_targets(q, sm):
    """開花日が無く、収量があり、SolidMoisture に同じ (field_id, year) の行がある件。"""
    nf = q[q['flower_date'].isna()]
    key = pd.MultiIndex.from_frame(sm[['field_id', 'year']].astype(int))
    has_sm = pd.MultiIndex.from_frame(nf[['field_id', 'year']]).isin(key)
    t = nf[nf['yield'].notna() & has_sm]
    return nf, t


def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
    os.makedirs(OUT_DIR, exist_ok=True)
    q, w, sm, dl = load()
    cases, T = build_cases(q, w, dl)
    Gs = cum_gdd(T)
    years = sorted(cases['year'].unique())

    param_list, curve_rows, preds = [], [], []
    for fold in years + ['all']:
        tr_m = (cases['year'] != fold).values if fold != 'all' else np.ones(len(cases), bool)
        tr = cases[tr_m]
        params, curves = fit_fold(tr, Gs[:, tr_m])
        param_list += param_rows(fold, params, tr)
        cd = pd.DataFrame({'Tb': TB_GRID, **curves})
        cd.insert(0, 'fold', fold)
        curve_rows.append(cd)
        if fold != 'all':
            te_m = ~tr_m
            te = cases[te_m].copy()
            for m, p in predict(params, te, Gs[:, te_m]).items():
                te[f'pred_{m}'] = p
            preds.append(te)

    params = pd.DataFrame(param_list)
    curve_df = pd.concat(curve_rows, ignore_index=True)
    cv = pd.concat(preds).sort_index()
    met = pd.concat([metrics(cv)] + [metrics(cv, by) for by in ['year', 'breed', 'region']],
                    ignore_index=True)
    corr = residual_corr(cv)
    # 比較用の基準: d2f の平均値、d2f を播種日の通日で線形回帰(phenology.py の (ii) と同じ形)
    ref = {'平均 d2f': [], '播種日の通日で回帰': []}
    for y in years:
        tr, te = cv[cv['year'] != y], cv[cv['year'] == y]
        ref['平均 d2f'] += list(tr['d2f'].mean() - te['d2f'])
        b1, b0 = np.polyfit(tr['seed_doy'], tr['d2f'], 1)
        ref['播種日の通日で回帰'] += list(b0 + b1 * te['seed_doy'] - te['d2f'])
    ref = pd.DataFrame([{'model': k, 'n': len(v), 'RMSE': rmse(v), 'MAE': float(np.mean(np.abs(v))),
                         'bias': float(np.mean(v))} for k, v in ref.items()])

    params.to_csv(os.path.join(OUT_DIR, 'params.csv'), index=False, encoding='utf-8-sig')
    curve_df.to_csv(os.path.join(OUT_DIR, 'rmse_curve.csv'), index=False, encoding='utf-8-sig')
    cv[['field_id', 'place', 'region', 'year', 'breed', 'lat', 'seed_date', 'flower_date', 'd2f',
        'seed_doy', 'mean_daylength'] + [f'pred_{m}' for m in MODELS]].to_csv(
        os.path.join(OUT_DIR, 'cv_predictions.csv'), index=False, encoding='utf-8-sig')
    met.to_csv(os.path.join(OUT_DIR, 'cv_metrics.csv'), index=False, encoding='utf-8-sig')
    plot_rmse_curve(curve_df, os.path.join(OUT_DIR, 'rmse_curve.png'))
    plot_pred_obs(cv, os.path.join(OUT_DIR, 'pred_vs_obs.png'))
    plot_residuals(cv, corr, os.path.join(OUT_DIR, 'residual_diagnostics.png'))

    # ---- 要約
    L = ['# 開花日予測 手順2:GDD モデル(ベースライン)', '',
         '- スクリプト: `rocket/scripts/fit_flowering_gdd.py`',
         f'- 対象: 開花日のある {len(cases)} 件(field_id あり)。d2f = 開花日 − 播種日',
         '- 積算は播種日の翌日から max(TMP_mea − Tb, 0)。累積が K 以上になった最初の日を予測開花日とする',
         f'- Tb は {TB_GRID[0]:.0f}〜{TB_GRID[-1]:.0f}℃(0.5℃刻み)を学習 RMSE 最小で選び、K は学習データの開花日までの積算値の平均',
         f'- (b)(c) の品種別推定は、fold 内の学習件数が {MIN_N} 件以上の品種のみ(それ以外は共通値)',
         '- 評価は leave-one-year-out。Tb と K は各 fold の学習データだけで決めた', '',
         '## 1. CV の評価指標(全体)', '',
         md_table(met[met['group_by'] == 'all'][['model', 'n', 'n_unreached', 'RMSE', 'MAE', 'bias']]), '',
         '比較用の基準(同じ LOYO):', '', md_table(ref), '',
         '参考: 既往研究の誤差は約 1.4〜2.2 日(中野ら 2015)、秋田県のリュウホウで RMSE 2.1 日。', '',
         '## 2. 選ばれたパラメータ', '',
         '(a) と (c) の Tb(fold ごと):', '',
         md_table(params[(params['breed'] == '(共通)') & params['model'].isin(['a', 'c'])]
                  [['fold', 'model', 'n_train', 'Tb', 'K']], '.1f'), '',
         '品種別のパラメータ(全データで推定):', '',
         md_table(params[(params['fold'] == 'all') & (params['breed'] != '(共通)')]
                  [['model', 'breed', 'n_train', 'Tb', 'K']], '.1f'), '',
         'fold ごとの全件は `params.csv`。', '',
         '## 3. RMSE 曲線の平らさ', '',
         '学習 RMSE が最小値 + 0.1 日以内に入る Tb の範囲と、曲線全体の振れ幅(最大 − 最小)。', '',
         '| fold | 曲線 | 最小 RMSE | 選ばれた Tb | +0.1 日以内の Tb | 振れ幅 [日] |', '|---|---|---|---|---|---|']
    for fold, d in curve_df.groupby('fold', sort=False):
        for k in [c for c in d.columns if c not in ('fold', 'Tb')]:
            c = d[k].values
            lo, hi, amp = flatness(c)
            L.append(f'| {fold} | {k} | {c.min():.2f} | {TB_GRID[np.argmin(c)]:.1f} | '
                     f'{lo:.1f}〜{hi:.1f} | {amp:.2f} |')
    L += ['', '## 4. 年別・品種別・地域別の CV 指標(RMSE とバイアス、日)', '']
    for by, name in [('year', '年(テスト年)'), ('breed', '品種'), ('region', '地域')]:
        t = pivot_metrics(met, by).sort_values('n', ascending=False)
        L += [f'### {name}', '', md_table(t), '']
    L += ['## 5. 残差の診断(残差 = 予測 − 実測)', '',
          md_table(corr.assign(pearson_p=corr['pearson_p'].map('{:.2g}'.format),
                               spearman_p=corr['spearman_p'].map('{:.2g}'.format)), '.3f'), '',
          'slope は残差の回帰係数(日 / 単位)。図は `residual_diagnostics.png`。', '']

    nf, tgt = no_flower_targets(q, sm)
    L += ['## 付録:開花日の予測が必要な件', '',
          f'開花日が無く field_id のある {len(nf)} 件のうち、収量があり、SolidMoisture に同じ '
          f'(field_id, year) の行がある件: **{len(tgt)} 件**', '',
          md_table(tgt.groupby('year').size().rename('n').reset_index()), '',
          md_table(tgt.groupby('region').size().rename('n').reset_index()), '',
          md_table(tgt.groupby('breed').size().rename('n').sort_values(ascending=False).reset_index()), '']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'保存先: {OUT_DIR}')


if __name__ == '__main__':
    main()
