"""
段階39: N_post の情報は「反応の大きさ」と同じものか — 割り算を使わない切り分け(EXP-048、D-019)。

段階38で上昇量で割った NP_rise は偶然の水準に下がったが、小さな上昇量の後の極端な値によるノイズと区別できなかった。
今回は割り算に頼らずに、次の4入力(長さ59の1チャンネル)を段階38と同じ MiniRocket で比べる。
  NP              雨の直後の減少(段階38と同じ)
  RISE            雨の日の max(Δθ, 0)(反応の大きさだけ)
  NP_resid        NP − b_j × 上昇量(j = 雨の後の何日目か。b_j はラベルを使わず全件の原点を通る最小二乗)
  NP_rise_robust  上昇量 ≥ 0.01 のイベントの後だけ NP / 上昇量 を [−1, 0] に切り詰め、ほかは0
主は主ラベル120件。参考は段階38の主の集合108件(段階38の NP・NP_rise と同じ分割で比べる)。設定は docs/decisions.md の D-019。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage39\\run_stage39.py --part prep    (入力の作成と確認、数秒)
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage39\\run_stage39.py --part trial   (乱数2・置換20・ブートストラップ100、確認だけ)
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage39\\run_stage39.py --part main    (本番)
出力: rocket/outputs/stage39/(試運転は outputs/stage39/trial/)、入力は rocket/data/processed/stage39/
"""
import argparse
import os
import pickle
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage38'))
import run_stage38 as s38  # noqa: E402
from src.evaluation import make_splits, split_table  # noqa: E402

s6, s13, s27, s29 = s38.s6, s38.s13, s38.s27, s38.s29
K = s38.K
R_MIN = 0.01
D38 = s38.NEW_DATA
O38 = s38.OUT_MAIN
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage39')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage39')
MAIN = ['NP', 'RISE', 'NP_resid', 'NP_rise_robust']
REF = ['r108_RISE', 'r108_NP_resid', 'r108_NP_rise_robust']
NAMES = {'NP': 'N_post(基準)', 'RISE': '雨の日の上昇 max(Δθ, 0)', 'NP_resid': 'NP − b_j × 上昇量(残差)',
         'NP_rise_robust': 'NP ÷ 上昇量(上昇量 ≥ 0.01、[−1, 0] に切り詰め)',
         's38_NP': '段階38の NP(108件)', 's38_NP_rise': '段階38の NP_rise(108件)',
         'r108_RISE': '参考: RISE(108件)', 'r108_NP_resid': '参考: NP_resid(108件、b_j も108件で)',
         'r108_NP_rise_robust': '参考: NP_rise_robust(108件)'}
s38.NAMES.update(NAMES)
COL = {'NP': '#2a78d6', 'RISE': '#eb6834', 'NP_resid': '#1f9e74', 'NP_rise_robust': '#9b59b6', 's38_NP': '#7aa7e0',
       's38_NP_rise': '#f0a37f', 'r108_RISE': '#f0a37f', 'r108_NP_resid': '#7cc8a8', 'r108_NP_rise_robust': '#c39bd3'}
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- 入力
def day_table(log):
    """雨の直後の日の表(圃場 i、日 t、j、減少 d、上昇量 r、イベント番号)と、NP・RISE・meta。段階38の定義・関数を使う。"""
    TH, meta = s38.load_cases()
    th, ok = s38.theta_ext(TH, meta)
    log(f'  確認 段階38の theta_ext: θ(−1)・窓の60日が theta.npy と一致 {ok}')
    assert ok
    X = th[:, 3:][:, None, :]
    pre = th[:, 1:4] - th[:, 0:3]                         # Δθ(−2), Δθ(−1), Δθ(0)
    comps, rain, post = s29.components(X, pre, K)
    NP = comps['N_post'][:, 0, :]
    dext = np.concatenate([np.nan_to_num(pre, nan=0.0), np.diff(th[:, 3:], axis=1)], axis=1)   # t = −2 … 59
    rain_ext = dext > 0
    rows, evs = [], []
    for i in range(len(meta)):
        for s_, e_ in s38.s29_runs(rain_ext[i]):        # 段階38の build_inputs と同じ規則
            ts_, te_ = s_ - 2, e_ - 2
            days = []
            for t in range(te_ + 1, te_ + K + 1):
                if t > 59 or rain_ext[i, t + 2]:
                    break
                if t >= 1:
                    days.append(t)
            if not days:
                continue
            r = th[i, te_ + 3] - th[i, ts_ + 2]
            ev = len(evs)
            evs.append({'field_id': meta.field_id[i], 'year': meta.year[i], 't_start': ts_, 't_end': te_,
                        'n_post_days': len(days), 'rise': r})
            for t in days:
                rows.append({'i': i, 't': t, 'j': t - te_, 'd': NP[i, t - 1], 'r': r, 'ev': ev})
    D = pd.DataFrame(rows)
    E = pd.DataFrame(evs)
    # 段階38との一致
    e38 = pd.read_csv(os.path.join(D38, 'rain_events.csv'), encoding='utf-8-sig')
    ok_e = len(e38) == len(E) and all(np.array_equal(e38[c].values, E[c].values) for c in
                                      ['field_id', 'year', 't_start', 't_end', 'n_post_days']) and \
        np.allclose(e38.rise.values, E.rise.values, rtol=0, atol=1e-12)
    np38 = np.load(os.path.join(D38, 'NP_X.npy'))[:, 0, :]
    ok_np = np.array_equal(np38, NP)
    rebuilt = np.zeros_like(NP)
    rebuilt[D.i.values, D.t.values - 1] = D.d.values
    ok_days = len(D) == int(post.sum()) and np.array_equal(rebuilt, NP)
    RISE = np.maximum(np.diff(th[:, 3:], axis=1), 0.0)
    ok_rise = np.array_equal(RISE, np.load(os.path.join(D38, 'P_X.npy'))[:, 0, :]) and \
        np.array_equal(RISE > 0, rain)
    log(f'  確認 イベント(区間・日数・上昇量)が段階38の rain_events.csv と一致 {ok_e}、NP が段階38の NP_X.npy と一致 {ok_np}、'
        f'日の表から NP を組み立て直すと一致 {ok_days}、RISE が段階38の P と一致し0でない日 = 雨の日 {ok_rise}')
    assert ok_e and ok_np and ok_days and ok_rise
    m38 = pd.read_csv(os.path.join(D38, 'meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    assert np.array_equal(m38.field_id.values, meta.field_id.values) and np.array_equal(m38.year.values, meta.year.values)
    meta = m38
    meta['y'] = meta[s38.LABEL].astype(int)
    return D, E, NP, RISE, meta


def fit_b(D):
    """j ごとに原点を通る最小二乗 d ≈ b_j r。戻り値 {j: b_j}。"""
    return {j: float((g.d * g.r).sum() / (g.r ** 2).sum()) for j, g in D.groupby('j')}


def resid_input(D, b, n):
    X = np.zeros((n, 59))
    X[D.i.values, D.t.values - 1] = D.d.values - D.j.map(b).values * D.r.values
    return X


def robust_input(D, n):
    X = np.zeros((n, 59))
    ok = D.r >= R_MIN
    g = D[ok]
    X[g.i.values, g.t.values - 1] = np.clip(g.d.values / g.r.values, -1.0, 0.0)
    return X


def field_means(D, X, n):
    """雨の直後の日の平均(圃場ごと)。"""
    out = np.full(n, np.nan)
    s = pd.Series(X[D.i.values, D.t.values - 1]).groupby(D.i.values).mean()
    out[s.index.values] = s.values
    return out


def part_prep(log, out_dir):
    log('\n==== 入力の作成と確認 ====')
    D, E, NP, RISE, meta = day_table(log)
    n = len(meta)
    m108 = (meta.bd.notna() & meta.th30.notna()).values
    log(f'  主 {n} 件(多収 {int(meta.y.sum())} / 低収 {int((1 - meta.y).sum())})、参考 {int(m108.sum())} 件。'
        f'雨のイベント {len(E)} 個、雨の直後の日 {len(D)} 日(j = 1・2・3: '
        f'{[int((D.j == j).sum()) for j in (1, 2, 3)]})')
    # NP_resid
    fits = []
    Xs = {}
    for tag, sel in [('120', np.ones(n, bool)), ('108', m108)]:
        Ds = D[sel[D.i.values]]
        b = fit_b(Ds)
        Xr = resid_input(Ds, b, n)
        Xr[~sel] = np.nan
        Xs[tag] = Xr
        res = Xr[Ds.i.values, Ds.t.values - 1]
        for j in (1, 2, 3):
            g = Ds[Ds.j == j]
            rr = res[(Ds.j == j).values]
            fits.append({'set': tag, 'j': j, 'b_j': b[j], 'n_days': len(g), 'n_fields': g.i.nunique(),
                         'r2_through_origin': 1 - (rr ** 2).sum() / (g.d ** 2).sum(),
                         'pearson_d_r': stats.pearsonr(g.d, g.r)[0], 'pearson_resid_r': stats.pearsonr(rr, g.r)[0],
                         'spearman_resid_r': stats.spearmanr(rr, g.r)[0], 'mean_d': g.d.mean(), 'mean_resid': rr.mean()})
        log(f'  NP_resid({tag}件で b_j を推定): ' + '、'.join(f'b_{j} = {b[j]:+.4f}' for j in (1, 2, 3))
            + f'(回帰に使った日数 {len(Ds)})')
    fit = pd.DataFrame(fits)
    fit.to_csv(os.path.join(out_dir, 'resid_fit.csv'), index=False, encoding='utf-8-sig')
    log(fit.round(4).to_string(index=False))
    Xres = Xs['120']
    res_all = Xres[D.i.values, D.t.values - 1]
    log(f'  残差と上昇量の相関(日単位、全 {len(D)} 日): Pearson {stats.pearsonr(res_all, D.r)[0]:+.3f}、Spearman '
        f'{stats.spearmanr(res_all, D.r)[0]:+.3f}(減少 d と上昇量: Pearson {stats.pearsonr(D.d, D.r)[0]:+.3f}、Spearman '
        f'{stats.spearmanr(D.d, D.r)[0]:+.3f})')
    # NP_rise_robust
    Xrob = robust_input(D, n)
    small = D.r < R_MIN
    ok = D[~small]
    ratio = ok.d / ok.r
    log(f'  NP_rise_robust: 上昇量 < {R_MIN} のイベント {int((E.rise < R_MIN).sum())} / {len(E)} 個'
        f'({(E.rise < R_MIN).mean():.3f})、その後の日 {int(small.sum())} / {len(D)} 日({small.mean():.3f})を0にした。'
        f'比が −1 未満で −1 に切り詰めた値 {int((ratio < -1).sum())} 個({(ratio < -1).mean():.3f})、0を超える値 {int((ratio > 0).sum())} 個')
    np38r = np.load(os.path.join(D38, 'NP_rise_X.npy'))[:, 0, :]
    inputs = {'NP': NP, 'RISE': RISE, 'NP_resid': Xres, 'NP_rise_robust': Xrob, 'NP_resid_108': Xs['108']}
    os.makedirs(NEW_DATA, exist_ok=True)
    for k, v in inputs.items():
        np.save(os.path.join(NEW_DATA, f'{k}_X.npy'), v[:, None, :])
    meta.to_csv(os.path.join(NEW_DATA, 'meta.csv'), index=False, encoding='utf-8-sig')
    D.to_csv(os.path.join(NEW_DATA, 'post_days.csv'), index=False, encoding='utf-8-sig')
    # 圃場ごとの要約の相関
    rise_ev = E.groupby(['field_id', 'year']).rise.mean()
    f = pd.DataFrame({'NP': field_means(D, NP, n), 'NP_resid': field_means(D, Xres, n), 'NP_rise_robust': field_means(D, Xrob, n),
                      'NP_rise_s38': field_means(D, np38r, n), 'RISE': RISE.sum(1) / np.maximum((RISE > 0).sum(1), 1),
                      'rise_ev': rise_ev.reindex(list(zip(meta.field_id, meta.year))).values})
    f.insert(0, 'year', meta.year.values)
    f.insert(0, 'field_id', meta.field_id.values)
    f.to_csv(os.path.join(NEW_DATA, 'field_summaries.csv'), index=False, encoding='utf-8-sig')
    cols = ['NP', 'RISE', 'NP_resid', 'NP_rise_robust', 'NP_rise_s38', 'rise_ev']
    rho = f[cols].corr(method='spearman')
    u = f[cols].groupby(meta.muniyear.values).transform(lambda c: (c.rank() - 0.5) / c.notna().sum())
    rho_w = u.corr()
    out = pd.concat([rho.assign(kind='Spearman(120件)'), rho_w.assign(kind='層内の正規化順位の Pearson(120件)')])
    out.index.name = 'feature'
    out.to_csv(os.path.join(out_dir, 'feature_corr.csv'), encoding='utf-8-sig')
    log('  [圃場ごとの平均(雨の直後の日の平均。RISE は雨の日の平均、rise_ev はイベントの上昇量の平均)の Spearman、120件]')
    log(rho.round(3).to_string())
    log(f'  上昇量との相関(Spearman / 層内): NP {rho.loc["NP", "rise_ev"]:+.3f} / {rho_w.loc["NP", "rise_ev"]:+.3f}、NP_resid '
        f'{rho.loc["NP_resid", "rise_ev"]:+.3f} / {rho_w.loc["NP_resid", "rise_ev"]:+.3f}、段階38の NP_rise '
        f'{rho.loc["NP_rise_s38", "rise_ev"]:+.3f} / {rho_w.loc["NP_rise_s38", "rise_ev"]:+.3f}、NP_rise_robust '
        f'{rho.loc["NP_rise_robust", "rise_ev"]:+.3f} / {rho_w.loc["NP_rise_robust", "rise_ev"]:+.3f}')
    for k, v in inputs.items():
        fin = np.isfinite(v).all(1)
        log(f'  {k}: 有限な件 {int(fin.sum())}、0でない日の割合 {np.mean(v[fin] != 0):.3f}、範囲 {np.nanmin(v):+.4f}〜{np.nanmax(v):+.4f}')
    plot_scatter(D, fit[fit.set == '120'], out_dir)


def plot_scatter(D, fit, out_dir):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, j in zip(axs, (1, 2, 3)):
        g = D[D.j == j]
        b = fit.set_index('j').b_j[j]
        ax.scatter(g.r, g.d, s=7, alpha=0.35, color='#2a78d6', lw=0)
        xx = np.linspace(0, g.r.max(), 50)
        ax.plot(xx, b * xx, color=s6.C_RED, lw=1.8, label=f'd = {b:+.3f} × r')
        ax.axvline(R_MIN, color=MUTED, lw=0.8, ls=':')
        ax.set_title(f'雨の後 {j} 日目({len(g)} 日)', loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('イベントの上昇量 r = θ(e) − θ(s−1)')
        ax.legend(frameon=False, fontsize=9, loc='lower left')
        s6.style(ax)
    axs[0].set_ylabel('減少 d = min(Δθ, 0)')
    fig.suptitle(f'段階39: 雨の直後の減少と上昇量(120件、原点を通る回帰。点線 = 上昇量 {R_MIN})', fontsize=12, color=INK, x=0.01,
                 ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out_dir, 'fig_scatter_resid.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 評価
def load38(name, seeds, X, meta, sp, log):
    """段階38の保存済みの結果(108件)。乱数0の先頭10 fold を計算し直して一致を確認。"""
    with open(os.path.join(O38, f'partial_{name}.pkl'), 'rb') as fh:
        saved = pickle.load(fh)
    assert saved['tag']['seeds'] == list(range(10)) and saved['tag']['P'] == 200 and saved['tag']['n'] == len(meta)
    out = saved['out']
    f0 = out['folds'][0]
    re = s6.eval_cached(s6.fold_features('minirocket', 0, X, sp[:10]), meta.y.values, sp[:10])
    ok = np.allclose(re.auc.values, f0.auc.values[:10], rtol=0, atol=1e-12) and \
        np.allclose(re.bacc.values, f0.bacc.values[:10], rtol=0, atol=1e-12)
    log(f'  段階38の {name}: 保存済みの結果を使う。乱数0の先頭10 fold の計算し直しが一致 {ok}')
    assert ok
    out = dict(out)
    out['S'] = out['S'][:len(seeds)]
    return out


def summarize(res, names, null_dirs):
    rows = []
    for nm in names:
        r = {x['metric']: x for x in res[nm]['summary']}
        a, b = r['all_auc'], r['all_bacc']
        rows.append({'config': nm, 'name': NAMES[nm], 'n': a['n'], 'n_seeds': a['n_seeds'], 'n_perm': a['n_perm'],
                     'auc': a['mean'], 'seed_min': a['seed_min'], 'seed_max': a['seed_max'], 'null_mean': a['null_mean'],
                     'auc_null_q95': a['null_q95'], 'p_auc': a['p'], 'bacc': b['mean'], 'bacc_null_q95': b['null_q95'],
                     'p_bacc': b['p'], 'alpha_at_max': a.get('alpha_at_max'), 'null_file': null_dirs[nm]})
    return pd.DataFrame(rows)


def diffs(res, pairs, meta, sp, n_boot, tag):
    names = sorted({x for p in pairs for x in p})
    Cs, N0 = {}, None
    for nm in names:
        C, Nn = s27.pair_sums(res[nm]['S'], meta, sp)
        N0 = Nn if N0 is None else N0
        assert np.array_equal(Nn, N0)
        Cs[nm] = C
    boot, pooled = s27.boot_diff(Cs, N0, meta, n_boot)
    fm = {nm: np.mean([x['all_auc'] for x in res[nm]['seed']]) for nm in names}
    rows = []
    for a, b in pairs:
        d = boot[a] - boot[b]
        rows.append({'set': tag, 'comparison': f'{a} − {b}', 'point_fold_mean': fm[a] - fm[b],
                     'point_pooled_pairs': pooled[a] - pooled[b], 'boot_q025': d.quantile(.025), 'boot_q975': d.quantile(.975),
                     'boot_frac_pos': (d > 0).mean(), 'n_boot': n_boot})
    return pd.DataFrame(rows), boot.assign(set=tag)


def _fold_resid(seed, D, n, tr, te):
    b = fit_b(D[np.isin(D.i.values, tr)])
    X = resid_input(D, b, n)[:, None, :]
    Ftr, Fte = s6._one_fold('minirocket', seed, X, tr, te)
    return (Ftr, Fte), b


def sens_resid(meta, sp, seeds, log):
    D = pd.read_csv(os.path.join(NEW_DATA, 'post_days.csv'), encoding='utf-8-sig')
    y = meta.y.values
    per, bs = [], None
    for seed in seeds:
        out = Parallel(n_jobs=s13.N_JOBS)(delayed(_fold_resid)(seed, D, len(meta), tr, te) for r, f, tr, te in sp)
        f = s6.eval_cached([o[0] for o in out], y, sp)
        per.append({'seed': seed, 'all_auc': f.auc.mean(), 'all_bacc': f.bacc.mean(),
                    'alpha_at_max': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[-1]).mean()})
        if bs is None:
            bs = pd.DataFrame([o[1] for o in out])
    per = pd.DataFrame(per)
    log(f'  感度分析 NP_resid(b_j を各 fold の学習データで推定、置換なし): 全体 AUC {per.all_auc.mean():.3f}(乱数 '
        f'{per.all_auc.min():.3f}〜{per.all_auc.max():.3f})、bacc {per.all_bacc.mean():.3f}、alpha 上限 {per.alpha_at_max.mean():.2f}。'
        f'fold ごとの b_j の範囲: ' + '、'.join(f'b_{j} {bs[j].min():+.4f}〜{bs[j].max():+.4f}' for j in (1, 2, 3)))
    return per


def two_channel(X2, meta, sp, seeds, log):
    y = meta.y.values
    per = []
    for seed in seeds:
        f = s6.eval_cached(s27.fold_features('PN', seed, X2, sp), y, sp)
        per.append({'seed': seed, 'all_auc': f.auc.mean(), 'all_bacc': f.bacc.mean(),
                    'alpha_at_max': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[-1]).mean()})
    per = pd.DataFrame(per)
    log(f'  参考 NP + RISE の2チャンネル(MiniRocketMultivariate、置換なし): 全体 AUC {per.all_auc.mean():.3f}(乱数 '
        f'{per.all_auc.min():.3f}〜{per.all_auc.max():.3f})、bacc {per.all_bacc.mean():.3f}、alpha 上限 {per.alpha_at_max.mean():.2f}')
    return per


def part_eval(log, out_dir, seeds, P, n_boot):
    log(f'\n==== 評価(乱数 {seeds}、置換 {P} 回、ブートストラップ {n_boot} 回) ====')
    s6.N_JOBS = s13.N_JOBS
    s38.s12.N_JOBS = s13.N_JOBS
    meta = pd.read_csv(os.path.join(NEW_DATA, 'meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    X = {k: np.load(os.path.join(NEW_DATA, f'{k}_X.npy')) for k in MAIN + ['NP_resid_108']}
    # ---- 主: 120件
    sp = make_splits(meta.y.values, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    s38.check_splits(meta, sp, log, '主 120件')
    s13.prechecks('median', X['NP'], meta, sp, log)
    res, nulls = {}, {}
    for nm in MAIN:
        res[nm] = s38.run_mr(nm, X[nm], meta, sp, seeds, P, log, out_dir)
        nulls[nm] = os.path.join(out_dir, f'null_{nm}.npy')
    sm = summarize(res, MAIN, nulls)
    sm['p_holm'] = s27.holm(sm.p_auc.values)
    sm['set'] = '主 120件'
    d_main, b_main = diffs(res, [('RISE', 'NP'), ('NP_resid', 'NP'), ('NP_rise_robust', 'NP'), ('NP', 'RISE')], meta, sp, n_boot,
                           '主 120件')
    log('\n[記述: 主の120件]')
    sens = sens_resid(meta, sp, seeds, log)
    X2 = np.concatenate([X['NP'], X['RISE']], axis=1)
    two = two_channel(X2, meta, sp, seeds, log)
    # ---- 参考: 108件(段階38と同じ集合・分割)
    m108 = (meta.bd.notna() & meta.th30.notna()).values
    idx = np.flatnonzero(m108)
    mm = meta.iloc[idx].reset_index(drop=True)
    sp8 = make_splits(mm.y.values, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    log(f'\n  参考: 段階38の主の集合 {len(mm)} 件')
    s38.check_splits(mm, sp8, log, '参考 108件')
    res8, nulls8 = {}, {}
    for nm, src in [('s38_NP', 'NP'), ('s38_NP_rise', 'NP_rise')]:
        x38 = np.load(os.path.join(D38, f'{src}_X.npy'))[idx]
        res8[nm] = load38(src, seeds, x38, mm, sp8, log)
        nulls8[nm] = os.path.join(O38, f'null_{src}.npy')
    assert np.array_equal(np.load(os.path.join(D38, 'NP_X.npy'))[idx], X['NP'][idx])
    for nm, src in [('r108_RISE', 'RISE'), ('r108_NP_resid', 'NP_resid_108'), ('r108_NP_rise_robust', 'NP_rise_robust')]:
        xx = X[src][idx]
        assert np.isfinite(xx).all()
        res8[nm] = s38.run_mr(nm, xx, mm, sp8, seeds, P, log, out_dir)
        nulls8[nm] = os.path.join(out_dir, f'null_{nm}.npy')
    sm8 = summarize(res8, ['s38_NP', 's38_NP_rise'] + REF, nulls8)
    sm8['set'] = '参考 108件'
    d_ref, b_ref = diffs(res8, [('r108_NP_rise_robust', 's38_NP_rise'), ('r108_RISE', 's38_NP'), ('r108_NP_resid', 's38_NP'),
                                ('r108_NP_rise_robust', 's38_NP'), ('s38_NP_rise', 's38_NP')], mm, sp8, n_boot, '参考 108件')
    # ---- 保存
    allsm = pd.concat([sm, sm8], ignore_index=True)
    extra = pd.DataFrame([
        {'set': '主 120件', 'config': 'NP_resid_foldfit', 'name': '感度: NP_resid(b_j を fold の学習データで推定、置換なし)',
         'n': len(meta), 'n_seeds': len(sens), 'auc': sens.all_auc.mean(), 'seed_min': sens.all_auc.min(),
         'seed_max': sens.all_auc.max(), 'bacc': sens.all_bacc.mean(), 'alpha_at_max': sens.alpha_at_max.mean()},
        {'set': '主 120件', 'config': 'NP+RISE_2ch', 'name': '参考: NP と RISE の2チャンネル(置換なし)', 'n': len(meta),
         'n_seeds': len(two), 'auc': two.all_auc.mean(), 'seed_min': two.all_auc.min(), 'seed_max': two.all_auc.max(),
         'bacc': two.all_bacc.mean(), 'alpha_at_max': two.alpha_at_max.mean()}])
    allsm = pd.concat([allsm, extra], ignore_index=True)
    allsm['above_q95'] = allsm.auc > allsm.auc_null_q95
    cols = ['set', 'config', 'name', 'n', 'n_seeds', 'n_perm', 'auc', 'seed_min', 'seed_max', 'null_mean', 'auc_null_q95', 'p_auc',
            'p_holm', 'above_q95', 'bacc', 'bacc_null_q95', 'p_bacc', 'alpha_at_max']
    allsm[cols].to_csv(os.path.join(out_dir, 'auc_summary.csv'), index=False, encoding='utf-8-sig')
    dd = pd.concat([d_main, d_ref], ignore_index=True)
    dd.to_csv(os.path.join(out_dir, 'diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    pd.concat([b_main, b_ref]).to_csv(os.path.join(out_dir, 'bootstrap_auc.csv'), index=False, encoding='utf-8-sig')
    pd.concat([pd.DataFrame(r['seed']) for r in list(res.values()) + [res8[n] for n in REF]]).to_csv(
        os.path.join(out_dir, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat([sens.assign(config='NP_resid_foldfit'), two.assign(config='NP+RISE_2ch')]).to_csv(
        os.path.join(out_dir, 'per_seed_extra.csv'), index=False, encoding='utf-8-sig')
    log('\n[全体 AUC のまとめ]')
    log(allsm[cols].drop(columns=['name']).round(4).to_string(index=False))
    log(f'\n[差(市町村単位のクラスターブートストラップ {n_boot} 回)]')
    log(dd.round(4).to_string(index=False))
    plot_auc(allsm, out_dir)


def plot_auc(sm, out_dir):
    s = sm[sm.null_file.notna()].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 5.2))
    for i, r in s.iterrows():
        nl = np.load(r.null_file)[:, 1]
        vp = ax.violinplot([nl[np.isfinite(nl)]], positions=[i], widths=0.7, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor('#d8d6cf')
            b.set_alpha(0.8)
        ax.hlines(r.auc_null_q95, i - 0.3, i + 0.3, color=s6.C_RED, lw=2, ls='--')
        ax.scatter(i, r.auc, s=70, color=COL.get(r.config, '#9a9890'), zorder=3)
        lab = f'{r.auc:.3f}\np={r.p_auc:.3f}' + (f'\nHolm {r.p_holm:.3f}' if pd.notna(r.get('p_holm')) else '')
        ax.text(i + 0.12, r.auc + 0.01, lab, fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.axvline(len(MAIN) - 0.5, color=MUTED, lw=0.6)
    ax.set_ylim(0.28, 0.80)
    ax.set_xticks(range(len(s)), [c.replace('r108_', '108件\n').replace('s38_', '段階38\n') for c in s.config], fontsize=9)
    ax.set_ylabel('全体 AUC')
    ax.set_title('点 = 観測(乱数平均)、灰 = 置換の帰無分布、赤の破線 = 95%点。左: 主の120件 / 右: 参考の108件(段階38と同じ分割)',
                 loc='left', fontsize=10, color=INK)
    s6.style(ax)
    fig.suptitle('段階39: 割り算を使わない N_post の切り分け(全体 AUC と帰無分布)', fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, 'fig_auc_null.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='+', default=['prep'], choices=['prep', 'trial', 'main'])
    args = ap.parse_args()
    trial = 'trial' in args.part
    out_dir = os.path.join(OUT_MAIN, 'trial') if trial else OUT_MAIN
    os.makedirs(out_dir, exist_ok=True)
    append = 'main' in args.part and 'prep' not in args.part
    logf = open(os.path.join(out_dir, 'run_log.txt'), 'a' if append else 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'段階39({" ".join(args.part)})  並列数 {s13.N_JOBS}  設定は D-019')
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 EXP-048 が docs/experiments.md に無い: {"## EXP-048" not in docs}')
    if 'prep' in args.part:
        part_prep(log, OUT_MAIN)
    if trial:
        part_eval(log, out_dir, [0, 1], 20, 100)
    if 'main' in args.part:
        part_eval(log, out_dir, list(range(10)), 200, 2000)
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {out_dir}')
    logf.close()


if __name__ == '__main__':
    main()
