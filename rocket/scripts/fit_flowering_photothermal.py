"""
大豆の開花日予測 手順3: 光熱モデル(温度 × 日長)。

日ごとの発育速度 r = f(T) × g(DL)
  f(T)  = max(TMP_mea − Tb, 0)
  g(DL) = 1(DL ≦ CSDL)、max(0, 1 − PPSEN × (DL − CSDL))(DL > CSDL)
播種日の翌日から r を積算し、K に達した最初の日を予測開花日とする。

パラメータの推定(各 LOYO fold の学習データだけで):
  Tb(0〜12℃、0.5℃刻み)と CSDL(0.25 時間刻み)はグリッド。各点で PPSEN(0〜1 /時間)と K を
  scipy.optimize.minimize(Powell、範囲付き)で決める。整数日の RMSE は階段状で最適化しにくいので、
  最適化では K を超えた日を線形補間した連続の日数で RMSE を測り、グリッド点の比較は整数日の RMSE で行う。
  K だけ品種別のモデルは、共通モデルの Tb・CSDL・PPSEN を使い、6 品種(fold 内 15 件以上)の K だけを
  その品種の学習データで決め直す(それ以外の品種・品種欠損は共通の K)。
DL は day_length_sun(主)と day_length_civil(比較)。CSDL の範囲は sun で 12〜15 時間、
civil は日長が約 1 時間長いので 12〜16 時間にした。

比較対象(同じ LOYO):
  gdd_a     : 手順2の GDD (a)(scripts/fit_flowering_gdd.py)
  seed_reg  : d2f ~ 播種日の通日
  stat      : d2f ~ 播種日の通日 + 緯度 + 播種翌日から 40 日間の平均気温
  pt_sun / pt_sun_kb     : 光熱モデル(共通 / K だけ品種別)、DL = day_length_sun
  pt_civil / pt_civil_kb : 同上、DL = day_length_civil

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db、rocket/data/processed/daylength.csv
出力: rocket/outputs/flowering_photothermal/
  params.csv               fold ごとのパラメータ(範囲の端に張り付いたかの印つき)
  rmse_grid.csv            Tb × CSDL の各点での学習 RMSE と PPSEN, K
  rmse_curve.png           学習 RMSE と Tb・CSDL の関係
  cv_predictions.csv, cv_metrics.csv, residual_diagnostics.png, pred_vs_obs.png, summary.md
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import fit_flowering_gdd as gdd  # noqa: E402  手順2の読み込み・GDD (a)・図の体裁を再利用

DAYLENGTH = gdd.DAYLENGTH
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'flowering_photothermal')

HORIZON = gdd.HORIZON
MIN_N = gdd.MIN_N
TB_GRID = np.arange(0, 12.01, 0.5)
CSDL_GRID = {'sun': np.arange(12, 15.001, 0.25), 'civil': np.arange(12, 16.001, 0.25)}
PPSEN_BOUNDS = (0.0, 1.0)
PPSEN_STARTS = (0.05, 0.4)
STAT_DAYS = 40

MODELS = ['gdd_a', 'seed_reg', 'stat', 'pt_sun', 'pt_sun_kb', 'pt_civil', 'pt_civil_kb']
LABEL = {'gdd_a': 'GDD (a)', 'seed_reg': '播種日回帰', 'stat': '統計モデル',
         'pt_sun': '光熱(共通, sun)', 'pt_sun_kb': '光熱(K 品種別, sun)',
         'pt_civil': '光熱(共通, civil)', 'pt_civil_kb': '光熱(K 品種別, civil)'}
PLOT_MODELS = ['gdd_a', 'seed_reg', 'stat', 'pt_sun', 'pt_sun_kb']
COLORS = {'gdd_a': '#2a78d6', 'seed_reg': '#eb6834', 'stat': '#1baf7a',
          'pt_sun': '#eda100', 'pt_sun_kb': '#e87ba4'}
FOCUS_REGIONS = ['O', 'N', 'C']


# ---------------------------------------------------------------- データ
def daylength_matrix(cases, col):
    """播種翌日から HORIZON 日分の日長行列 (n, HORIZON)。範囲外は NaN。"""
    dl = pd.read_csv(DAYLENGTH, usecols=['field_id', 'date', col], parse_dates=['date'])
    s = dl.set_index(['field_id', 'date'])[col]
    D = np.full((len(cases), HORIZON), np.nan)
    for i, r in cases.iterrows():
        days = pd.date_range(r['sd'] + pd.Timedelta(days=1), periods=HORIZON)
        D[i] = s.reindex(pd.MultiIndex.from_product([[r['field_id']], days])).values
    return D


# ---------------------------------------------------------------- 光熱モデル
class Photothermal:
    """f(T) × g(DL) の積算。F[k] は TB_GRID[k] での f(T) の行列。"""

    def __init__(self, T, D):
        Tz = np.nan_to_num(T, nan=-np.inf)            # 範囲外の日は発育しない
        self.F = np.stack([np.maximum(Tz - tb, 0) for tb in TB_GRID])
        self.D = np.nan_to_num(D, nan=0.0)            # 範囲外の日は g = 1(F が 0 なので影響しない)

    def cum(self, k_tb, csdl, ppsen, rows=slice(None)):
        D = self.D[rows]
        g = np.where(D <= csdl, 1.0, np.maximum(0.0, 1 - ppsen * (D - csdl)))
        return np.cumsum(self.F[k_tb][rows] * g, axis=1)


def cross(C, K):
    """累積 C (n, H) が K に初めて達した日。戻り値: 整数日(未到達は NaN)、連続の日数(未到達は HORIZON)。"""
    n = len(C)
    K = np.broadcast_to(np.asarray(K, float), (n,))
    reached = C >= K[:, None]
    ok = reached.any(axis=1)
    j = reached.argmax(axis=1)
    idx = np.arange(n)
    prev = np.where(j > 0, C[idx, np.maximum(j - 1, 0)], 0.0)
    cur = C[idx, j]
    frac = np.clip((K - prev) / np.maximum(cur - prev, 1e-9), 0, 1)
    d_int = np.where(ok, j + 1, np.nan)
    d_cont = np.where(ok, j + frac + 0.5, HORIZON)     # ceil に合わせて +0.5
    return d_int, d_cont


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def int_rmse(d_int, d2f):
    return rmse(np.where(np.isnan(d_int), HORIZON, d_int) - d2f)


def fit_pk(pt, k_tb, csdl, rows, d2f):
    """Tb・CSDL を固定して PPSEN と K を決める。戻り値: PPSEN, K, 整数日 RMSE。"""
    def obj(x):
        C = pt.cum(k_tb, csdl, x[0], rows)
        return rmse(cross(C, x[1])[1] - d2f)

    best = None
    for p0 in PPSEN_STARTS:
        C0 = pt.cum(k_tb, csdl, p0, rows)
        k0 = C0[np.arange(len(d2f)), d2f - 1].mean()
        res = optimize.minimize(obj, [p0, k0], method='Powell',
                                bounds=[PPSEN_BOUNDS, (0.2 * k0, 3 * k0)],
                                options={'xtol': 1e-3, 'ftol': 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    p, K = best.x
    return p, K, int_rmse(cross(pt.cum(k_tb, csdl, p, rows), K)[0], d2f)


def fit_k(C, d2f, k_start):
    """累積を固定して K だけ決める(品種別の K 用)。"""
    res = optimize.minimize_scalar(lambda K: rmse(cross(C, K)[1] - d2f),
                                   bounds=(0.2 * k_start, 3 * k_start), method='bounded')
    return res.x


def fit_photothermal(pt, rows, tr, csdl_grid):
    """学習データ(rows)で共通の Tb, CSDL, PPSEN, K と、品種別の K を決める。"""
    d2f = tr['d2f'].values
    grid = []
    for k_tb, tb in enumerate(TB_GRID):
        for csdl in csdl_grid:
            p, K, e = fit_pk(pt, k_tb, csdl, rows, d2f)
            grid.append({'Tb': tb, 'CSDL': csdl, 'PPSEN': p, 'K': K, 'rmse_train': e})
    grid = pd.DataFrame(grid)
    b = grid.loc[grid['rmse_train'].idxmin()]
    k_tb = int(np.argmin(np.abs(TB_GRID - b['Tb'])))
    C = pt.cum(k_tb, b['CSDL'], b['PPSEN'], rows)
    Kb = {}
    for br in gdd.target_breeds(tr):
        m = (tr['breed'] == br).values
        Kb[br] = fit_k(C[m], d2f[m], b['K'])
    return dict(Tb=b['Tb'], CSDL=b['CSDL'], PPSEN=b['PPSEN'], K=b['K'], k_tb=k_tb), Kb, grid


def predict_photothermal(pt, rows, te, prm, Kb):
    C = pt.cum(prm['k_tb'], prm['CSDL'], prm['PPSEN'], rows)
    common = cross(C, prm['K'])[0]
    Kvec = te['breed'].map(lambda b: Kb.get(b, prm['K'])).values
    return common, cross(C, Kvec)[0]


# ---------------------------------------------------------------- 比較対象
def stat_features(df):
    return np.column_stack([np.ones(len(df)), df['seed_doy'], df['lat'], df['tmean40']])


def fit_predict_linear(Xtr, ytr, Xte):
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    return Xte @ beta, beta


# ---------------------------------------------------------------- 評価
def metrics(df, by=None):
    rows = []
    groups = [('全体', df)] if by is None else list(df.groupby(by))
    for g, d in groups:
        for m in MODELS:
            e = d[f'pred_{m}'] - d['d2f']
            ok = e.notna()
            rows.append({'group_by': by or 'all', 'group': g, 'model': m, 'n': int(ok.sum()),
                         'n_unreached': int((~ok).sum()), 'RMSE': rmse(e[ok]),
                         'MAE': float(e[ok].abs().mean()), 'bias': float(e[ok].mean())})
    return pd.DataFrame(rows)


DIAG = gdd.DIAG


def residual_corr(cv):
    rows = []
    for m in MODELS:
        e = cv[f'pred_{m}'] - cv['d2f']
        ok = e.notna()
        for x, _ in DIAG:
            r, pr = stats.pearsonr(cv.loc[ok, x], e[ok])
            rho, ps = stats.spearmanr(cv.loc[ok, x], e[ok])
            rows.append({'model': m, 'x': x, 'pearson_r': r, 'pearson_p': pr,
                         'spearman_rho': rho, 'spearman_p': ps,
                         'slope': np.polyfit(cv.loc[ok, x], e[ok], 1)[0]})
    return pd.DataFrame(rows)


def edge_flags(row, csdl_grid):
    f = []
    if np.isclose(row['Tb'], TB_GRID[0]) or np.isclose(row['Tb'], TB_GRID[-1]):
        f.append('Tb')
    if np.isclose(row['CSDL'], csdl_grid[0]) or np.isclose(row['CSDL'], csdl_grid[-1]):
        f.append('CSDL')
    if row['PPSEN'] < PPSEN_BOUNDS[0] + 1e-3 or row['PPSEN'] > PPSEN_BOUNDS[1] - 1e-3:
        f.append('PPSEN')
    return ','.join(f)


# ---------------------------------------------------------------- 図
def plot_rmse_curve(grids, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, dl in enumerate(['sun', 'civil']):
        g = grids[grids['dl'] == dl]
        ax = axes[i, 0]
        gdd.style(ax)
        for k, (fold, d) in enumerate(g.groupby('fold', sort=False)):
            c = d.groupby('Tb')['rmse_train'].min()
            is_all = fold == 'all'
            ax.plot(c.index, c.values, color='#0b0b0b' if is_all else gdd.YEAR_COLORS[k],
                    lw=2.2 if is_all else 1.5, label='全データ' if is_all else f'{fold}年を除く')
        ax.set_title(f'{dl}: 学習 RMSE と Tb(CSDL 等は最良)', loc='left', fontsize=11)
        ax.set_xlabel('Tb [℃]')
        ax.set_ylabel('学習 RMSE [日]')
        ax.legend(frameon=False, fontsize=8)
        ax = axes[i, 1]
        gdd.style(ax)
        for k, (fold, d) in enumerate(g.groupby('fold', sort=False)):
            c = d.groupby('CSDL')['rmse_train'].min()
            is_all = fold == 'all'
            ax.plot(c.index, c.values, color='#0b0b0b' if is_all else gdd.YEAR_COLORS[k],
                    lw=2.2 if is_all else 1.5)
        ax.set_title(f'{dl}: 学習 RMSE と CSDL(Tb 等は最良)', loc='left', fontsize=11)
        ax.set_xlabel('CSDL [時間]')
        ax.set_ylabel('学習 RMSE [日]')
        ax = axes[i, 2]
        d = g[g['fold'] == 'all'].pivot(index='CSDL', columns='Tb', values='rmse_train')
        im = ax.imshow(d.values, origin='lower', aspect='auto', cmap='Blues_r',
                       extent=[TB_GRID[0] - 0.25, TB_GRID[-1] + 0.25,
                               d.index[0] - 0.125, d.index[-1] + 0.125])
        bi = np.unravel_index(np.nanargmin(d.values), d.shape)
        ax.plot(d.columns[bi[1]], d.index[bi[0]], marker='x', color='#e34948', ms=10, mew=2)
        ax.set_title(f'{dl}: 学習 RMSE(全データ、× が最小)', loc='left', fontsize=11)
        ax.set_xlabel('Tb [℃]')
        ax.set_ylabel('CSDL [時間]')
        fig.colorbar(im, ax=ax, label='学習 RMSE [日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_obs(cv, path):
    fig, axes = plt.subplots(1, len(PLOT_MODELS), figsize=(4.2 * len(PLOT_MODELS), 4.6),
                             sharex=True, sharey=True)
    lo, hi = cv['d2f'].min() - 5, cv['d2f'].max() + 5
    for ax, m in zip(axes, PLOT_MODELS):
        gdd.style(ax)
        e = cv[f'pred_{m}'] - cv['d2f']
        ax.scatter(cv['d2f'], cv[f'pred_{m}'], s=12, color=COLORS[m], alpha=0.55,
                   edgecolor=gdd.SURF, lw=0.4)
        ax.plot([lo, hi], [lo, hi], color=gdd.INK, lw=1, ls='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f'{LABEL[m]}\nRMSE {rmse(e.dropna()):.2f} 日, バイアス {e.mean():+.2f} 日',
                     loc='left', fontsize=10)
        ax.set_xlabel('実測 d2f [日]')
    axes[0].set_ylabel('予測 d2f(LOYO)[日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(cv, corr, path):
    fig, axes = plt.subplots(len(PLOT_MODELS), 3, figsize=(15, 3.6 * len(PLOT_MODELS)), sharey='row')
    for i, m in enumerate(PLOT_MODELS):
        e = cv[f'pred_{m}'] - cv['d2f']
        ok = e.notna()
        for j, (x, xl) in enumerate(DIAG):
            ax = axes[i, j]
            gdd.style(ax)
            ax.scatter(cv[x], e, s=10, color=COLORS[m], alpha=0.5, edgecolor=gdd.SURF, lw=0.4)
            ax.axhline(0, color=gdd.INK, lw=1, ls='--')
            b1, b0 = np.polyfit(cv.loc[ok, x], e[ok], 1)
            xs = np.linspace(cv[x].min(), cv[x].max(), 10)
            ax.plot(xs, b0 + b1 * xs, color='#0b0b0b', lw=1.5)
            c = corr[(corr['model'] == m) & (corr['x'] == x)].iloc[0]
            ax.set_title(f'{LABEL[m]}  r = {c.pearson_r:+.2f}(p = {c.pearson_p:.2g})',
                         loc='left', fontsize=10)
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel('残差(予測 − 実測)[日]')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 64 件の確認
def target_check(q, sm, w):
    """予測が必要な 64 件について、各モデルの入力がそろっているかを数える。"""
    _, tgt = gdd.no_flower_targets(q, sm)
    tgt = tgt.copy()
    tgt['sd'] = pd.to_datetime(tgt['seed_date'])
    tw = w.set_index(['field_id', 'date'])['TMP_mea']
    full = []
    for r in tgt.itertuples():
        if pd.isna(r.sd):
            full.append(False)
            continue
        days = pd.date_range(r.sd + pd.Timedelta(days=1), periods=120)
        full.append(bool(tw.reindex(pd.MultiIndex.from_product([[r.field_id], days])).notna().all()))
    tgt['weather_120d'] = full
    return tgt


def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    q, w, sm, dl = gdd.load()
    cases, T = gdd.build_cases(q, w, dl)
    cases['tmean40'] = T[:, :STAT_DAYS].mean(axis=1)
    Gs = gdd.cum_gdd(T)
    pts = {d: Photothermal(T, daylength_matrix(cases, f'day_length_{d}')) for d in ['sun', 'civil']}
    years = sorted(cases['year'].unique())

    params, grids, preds = [], [], []
    for fold in years + ['all']:
        tr_m = (cases['year'] != fold).values if fold != 'all' else np.ones(len(cases), bool)
        tr = cases[tr_m]
        te_m = ~tr_m
        te = cases[te_m].copy()
        # GDD (a)
        tb_a, K_a, _ = gdd.fit_common(Gs[:, tr_m], tr['d2f'].values)
        params.append({'fold': fold, 'model': 'gdd_a', 'breed': '(共通)', 'n_train': len(tr),
                       'Tb': tb_a, 'K': K_a})
        # 播種日回帰・統計モデル
        X1 = lambda d: np.column_stack([np.ones(len(d)), d['seed_doy']])  # noqa: E731
        p_seed, b_seed = fit_predict_linear(X1(tr), tr['d2f'].values, X1(te))
        p_stat, b_stat = fit_predict_linear(stat_features(tr), tr['d2f'].values, stat_features(te))
        params.append({'fold': fold, 'model': 'seed_reg', 'breed': '(共通)', 'n_train': len(tr),
                       'b0': b_seed[0], 'b_seed_doy': b_seed[1]})
        params.append({'fold': fold, 'model': 'stat', 'breed': '(共通)', 'n_train': len(tr),
                       'b0': b_stat[0], 'b_seed_doy': b_stat[1], 'b_lat': b_stat[2],
                       'b_tmean40': b_stat[3]})
        # 光熱モデル
        for d in ['sun', 'civil']:
            prm, Kb, grid = fit_photothermal(pts[d], tr_m, tr, CSDL_GRID[d])
            grids.append(grid.assign(fold=fold, dl=d))
            params.append({'fold': fold, 'model': f'pt_{d}', 'breed': '(共通)', 'n_train': len(tr),
                           **{k: prm[k] for k in ['Tb', 'CSDL', 'PPSEN', 'K']},
                           'rmse_train': grid['rmse_train'].min(),
                           'edge': edge_flags(prm, CSDL_GRID[d])})
            for br, K in Kb.items():
                params.append({'fold': fold, 'model': f'pt_{d}_kb', 'breed': br,
                               'n_train': int((tr['breed'] == br).sum()),
                               **{k: prm[k] for k in ['Tb', 'CSDL', 'PPSEN']}, 'K': K})
            if fold != 'all':
                te[f'pred_pt_{d}'], te[f'pred_pt_{d}_kb'] = predict_photothermal(pts[d], te_m, te, prm, Kb)
        if fold != 'all':
            te['pred_gdd_a'] = gdd.predict_d2f(Gs[int(np.argmin(np.abs(gdd.TB_GRID - tb_a)))][te_m], K_a)
            te['pred_seed_reg'] = p_seed
            te['pred_stat'] = p_stat
            preds.append(te)
        print(f'fold {fold} 完了({time.time() - t0:.0f} 秒)', flush=True)

    params = pd.DataFrame(params)
    grids = pd.concat(grids, ignore_index=True)
    cv = pd.concat(preds).sort_index()
    met = pd.concat([metrics(cv)] + [metrics(cv, by) for by in ['year', 'breed', 'region']],
                    ignore_index=True)
    corr = residual_corr(cv)

    params.to_csv(os.path.join(OUT_DIR, 'params.csv'), index=False, encoding='utf-8-sig')
    grids.to_csv(os.path.join(OUT_DIR, 'rmse_grid.csv'), index=False, encoding='utf-8-sig')
    cv[['field_id', 'place', 'region', 'year', 'breed', 'lat', 'seed_date', 'flower_date', 'd2f',
        'seed_doy', 'mean_daylength', 'tmean40'] + [f'pred_{m}' for m in MODELS]].to_csv(
        os.path.join(OUT_DIR, 'cv_predictions.csv'), index=False, encoding='utf-8-sig')
    met.to_csv(os.path.join(OUT_DIR, 'cv_metrics.csv'), index=False, encoding='utf-8-sig')
    plot_rmse_curve(grids, os.path.join(OUT_DIR, 'rmse_curve.png'))
    plot_pred_obs(cv, os.path.join(OUT_DIR, 'pred_vs_obs.png'))
    plot_residuals(cv, corr, os.path.join(OUT_DIR, 'residual_diagnostics.png'))

    # ---- 要約
    md = gdd.md_table
    lab = lambda s: s.map(LABEL)  # noqa: E731
    allm = met[met['group_by'] == 'all'][['model', 'n', 'n_unreached', 'RMSE', 'MAE', 'bias']]
    L = ['# 開花日予測 手順3:光熱モデル', '',
         '- スクリプト: `rocket/scripts/fit_flowering_photothermal.py`',
         f'- 対象: 開花日のある {len(cases)} 件。評価は leave-one-year-out(パラメータは各 fold の学習データだけで決めた)',
         '- 光熱モデル: r = max(TMP_mea − Tb, 0) × g(DL)、g = 1(DL ≦ CSDL)/ max(0, 1 − PPSEN(DL − CSDL))。'
         '播種翌日から積算し K に達した最初の日を予測開花日とする',
         f'- 探索: Tb {TB_GRID[0]:.0f}〜{TB_GRID[-1]:.0f}℃(0.5℃刻み)、CSDL sun {CSDL_GRID["sun"][0]:.0f}〜{CSDL_GRID["sun"][-1]:.0f} 時間 / '
         f'civil {CSDL_GRID["civil"][0]:.0f}〜{CSDL_GRID["civil"][-1]:.0f} 時間(0.25 時間刻み)はグリッド、PPSEN(0〜1)と K は Powell 法',
         '- K だけ品種別: 共通モデルの Tb・CSDL・PPSEN を使い、6 品種(fold 内 15 件以上)の K だけ決め直した',
         f'- 統計モデル: d2f ~ 播種日の通日 + 緯度 + 播種翌日から {STAT_DAYS} 日間の平均気温(線形回帰)', '',
         '## 1. CV の評価指標(全体)', '', md(allm.assign(model=lab(allm['model']))), '',
         '参考: 既往研究の誤差は約 1.4〜2.2 日(中野ら 2015)。', '',
         '## 2. 地域別(特に O, N, C)', '']
    for stat_name in ['RMSE', 'bias']:
        d = met[met['group_by'] == 'region']
        t = d.pivot_table(index='group', columns='model', values=stat_name)[MODELS]
        t.columns = [LABEL[c] for c in t.columns]
        t.insert(0, 'n', d[d['model'] == 'gdd_a'].set_index('group')['n'])
        t = t.reset_index().rename(columns={'group': '地域'})
        t['_f'] = ~t['地域'].isin(FOCUS_REGIONS)
        t = t.sort_values(['_f', 'n'], ascending=[True, False]).drop(columns='_f')
        L += [f'### {stat_name}', '', md(t), '']
    d = met[met['group_by'] == 'year']
    t = d.pivot_table(index='group', columns='model', values='RMSE')[MODELS]
    t.columns = [LABEL[c] for c in t.columns]
    L += ['### テスト年別の RMSE', '', md(t.reset_index().rename(columns={'group': '年'})), '']
    d = met[(met['group_by'] == 'breed')]
    t = d.pivot_table(index='group', columns='model', values='RMSE')[MODELS]
    t.columns = [LABEL[c] for c in t.columns]
    t.insert(0, 'n', d[d['model'] == 'gdd_a'].set_index('group')['n'])
    t = t[t['n'] >= MIN_N].sort_values('n', ascending=False)
    L += ['### 品種別の RMSE(15 件以上)', '', md(t.reset_index().rename(columns={'group': '品種'})), '']

    L += ['## 3. 推定されたパラメータ', '', '光熱モデル(共通):', '',
          md(params[params['model'].isin(['pt_sun', 'pt_civil'])]
             [['model', 'fold', 'n_train', 'Tb', 'CSDL', 'PPSEN', 'K', 'rmse_train', 'edge']]
             .fillna(''), '.3f'), '',
          '`edge` は探索範囲の端に張り付いたパラメータ。', '',
          'K だけ品種別(全データ):', '',
          md(params[(params['fold'] == 'all') & params['model'].isin(['pt_sun_kb', 'pt_civil_kb'])]
             [['model', 'breed', 'n_train', 'K']], '.1f'), '',
          '統計モデルの係数(fold ごと):', '',
          md(params[params['model'] == 'stat'][['fold', 'n_train', 'b0', 'b_seed_doy', 'b_lat', 'b_tmean40']], '.3f'), '',
          '播種日回帰の係数(fold ごと):', '',
          md(params[params['model'] == 'seed_reg'][['fold', 'b0', 'b_seed_doy']], '.3f'), '',
          'GDD (a)(fold ごと):', '', md(params[params['model'] == 'gdd_a'][['fold', 'Tb', 'K']], '.1f'), '']

    L += ['## 4. 学習 RMSE の曲面の平らさ(光熱モデル、全データ)', '',
          '学習 RMSE が最小値 + 0.1 日以内に入るグリッド点の範囲。図は `rmse_curve.png`。', '',
          '| DL | 最小 RMSE | +0.1 日以内の点数 | Tb の範囲 | CSDL の範囲 | PPSEN の範囲 |', '|---|---|---|---|---|---|']
    for dlk in ['sun', 'civil']:
        g = grids[(grids['dl'] == dlk) & (grids['fold'] == 'all')]
        n = g[g['rmse_train'] <= g['rmse_train'].min() + 0.1]
        L.append(f'| {dlk} | {g["rmse_train"].min():.2f} | {len(n)} / {len(g)} | {n["Tb"].min():.1f}〜{n["Tb"].max():.1f} | '
                 f'{n["CSDL"].min():.2f}〜{n["CSDL"].max():.2f} | {n["PPSEN"].min():.3f}〜{n["PPSEN"].max():.3f} |')
    L += ['']

    L += ['## 5. 残差の診断(残差 = 予測 − 実測)', '',
          md(corr.assign(model=lab(corr['model']),
                         pearson_p=corr['pearson_p'].map('{:.2g}'.format),
                         spearman_p=corr['spearman_p'].map('{:.2g}'.format)), '.3f'), '']

    tgt = target_check(q, sm, w)
    L += ['## 付録:開花日の予測が必要な 64 件の入力', '',
          f'- 件数 {len(tgt)}、播種日あり {int(tgt["seed_date"].notna().sum())}、緯度あり {int(tgt["lat"].notna().sum())}、'
          f'播種翌日から 120 日分の気温あり {int(tgt["weather_120d"].sum())}',
          f'- 品種が 6 品種(K 品種別の対象)に入る件: {int(tgt["breed"].isin(gdd.target_breeds(cases)).sum())}', '',
          md(tgt.groupby(['region', 'breed']).size().rename('n').reset_index()), '']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'保存先: {OUT_DIR}({time.time() - t0:.0f} 秒)')


if __name__ == '__main__':
    main()
