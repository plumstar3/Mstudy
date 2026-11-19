"""
段階21: 年をまたいだ確認(R_all の地域内 W、EXP-030)。

段階19・20では、市町村単位のランダムな分割で、学習とテストに同じ年の圃場が入っていた。段階21では、ある年の圃場を、
ほかの年の圃場だけで学習したモデルで評価する(leave-one-year-out、LOYO)。テストの年は2016年と2017年
(2018年は W のペアが6組だけなので学習にだけ使う)。仮説は段階19の結果を見てから立てたものの確認。

設定(データ・目的変数・モデルは段階19の R_all と同じ、カーネルの乱数0〜9)
  R_all_LOYO   主。学習 = 学習に使える235件のうち年が Y でない全件。評価 = 年 Y の median の評価の件
  R_all_LOYRO  参考。LOYO の学習から、さらにテストの年の評価の件が属する市町村の件を除く(年と地域をまたぐ)
  M_diff_LOYO  参考。同じ年の分け方で分類(RidgeClassifierCV、学習はほかの年の median の評価の件)
指標: W = 2016年と2017年のテスト件の、同じ市町村×年の中の多収 × 低収のペアを合わせた AUC(乱数ごとに計算し10通りで平均)。
  全体の AUC は年ごと(年の中の評価の件のすべてのペア)。「全体(2年の平均)」は年ごとの AUC の平均。
置換検定(R_all_LOYO 1000回・M_diff_LOYO 200回、置換 i の乱数は [2, i]): R_all は段階19と同じく、集団の中で収量を
  並べ替えて y_rel と median のラベルを作り直す。M_diff は市町村×年の中でラベルを並べ替える(段階13と同じ)。
判定(事前に固定、主は1つだけ): R_all_LOYO の W(2年合わせ、乱数10通りの平均)が W の帰無分布の95%点を上回るか。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage21\\run_stage21.py
  試運転: ... run_stage21.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage21/
"""
import argparse
import os
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage19'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage20'))
import run_stage19 as s19  # noqa: E402
import run_stage20 as s20  # noqa: E402
from run_stage19 import BASE_ALPHAS, N_JOBS, RidgeRegNTrain, s6, s14  # noqa: E402
from src.evaluation import permute_within  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT19 = os.path.join(ROCKET_DIR, 'outputs', 'stage19')
OUT20 = os.path.join(ROCKET_DIR, 'outputs', 'stage20')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage21')
TEST_YEARS = [2016, 2017]
PERM_SEED = 2
# 想定(年ごと): 評価の件, 市町村, W のペア, 学習(ほかの年), うちテストの年の評価の件の市町村の件
EXPECT = {2016: (79, 18, 92, 112, 47), 2017: (37, 8, 49, 155, 39), 2018: (7, 2, 6, 203, 10)}
CFGS = ['R_all_LOYO', 'R_all_LOYRO', 'M_diff_LOYO']
CFG_LABEL = {'R_all_LOYO': 'R_all LOYO(主)', 'R_all_LOYRO': 'R_all LOYRO 年と地域をまたぐ(参考)',
             'M_diff_LOYO': 'M_diff LOYO(参考)'}
CFG_COLOR = {'R_all_LOYO': '#2a78d6', 'R_all_LOYRO': '#8fb8e8', 'M_diff_LOYO': '#9a9890'}
PT = ['W', 'B', 'X', 'ALL']

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- 分け方
def build_year_folds(meta, i123, meta123, log):
    """{config: [(year, tr, pr, te123, ev_in_pr)]}。pr = テストの年の学習に使える全件(順位相関・置換のラベル用)。"""
    yr, muni = meta.year.values, meta.muniCd.to_numpy(dtype=str)
    is123 = np.zeros(len(meta), bool)
    is123[i123] = True
    lab = meta.label_muniyear_median.values
    my = meta.muniyear.to_numpy(dtype=str)
    log(f'  確認1 同じ field_id が別の年に出てくる件(学習に使える235件): {int(meta.field_id.duplicated().sum())}')
    assert not meta.field_id.duplicated().any()
    ok_all, ok_leak = True, True
    folds = {c: [] for c in CFGS}
    for Y, exp in EXPECT.items():
        ev_idx = np.flatnonzero(is123 & (yr == Y))
        ev_munis = np.unique(muni[ev_idx])
        w = sum(int((lab[ev_idx][my[ev_idx] == g] == 1).sum() * (lab[ev_idx][my[ev_idx] == g] == 0).sum())
                for g in np.unique(my[ev_idx]))
        tr = np.flatnonzero(yr != Y)
        same = int(np.isin(muni[tr], ev_munis).sum())
        got = (len(ev_idx), len(ev_munis), w, len(tr), same)
        ok = got == exp
        ok_all &= ok
        log(f'    {Y}年: 評価の件 {got[0]}、市町村 {got[1]}、W のペア {got[2]}、学習(ほかの年){got[3]}、うち同じ市町村 '
            f'{got[4]} → LOYRO の学習 {got[3] - got[4]}。想定と一致 {ok}')
        if Y not in TEST_YEARS:
            continue
        pr = np.flatnonzero(yr == Y)
        te = np.flatnonzero(meta123.year.values == Y)  # 123件の中の位置(段階13の並び)
        assert np.array_equal(np.sort(i123[te]), ev_idx)
        ev = np.array([np.flatnonzero(pr == i)[0] for i in i123[te]])
        tr_ro = tr[~np.isin(muni[tr], ev_munis)]
        tr_m = np.flatnonzero(meta123.year.values != Y)  # 123件の中の位置
        for c, t in [('R_all_LOYO', tr), ('R_all_LOYRO', tr_ro), ('M_diff_LOYO', i123[tr_m])]:
            ok_leak &= not (yr[t] == Y).any()
            if c == 'R_all_LOYRO':
                ok_leak &= not np.isin(muni[t], ev_munis).any()
            folds[c].append((Y, t, pr, te, ev))
        folds.setdefault('_m_tr', []).append(tr_m)
    log(f'  確認1 年ごとの件数が想定と一致: {ok_all}')
    log(f'  確認2 学習側にテストの年の件が無い(全設定)、LOYRO ではテストの年の評価の件の市町村の件も無い: {ok_leak}')
    assert ok_all and ok_leak
    return folds


# ---------------------------------------------------------------- 集計
def year_counts(preds, Lpr, meta, fl):
    """年ごと(テストの年)の、ペアの種類ごとの (一致の数, ペア数)((P,) ずつ)と bacc (P,)。"""
    out = []
    for (Y, tr, pr, te, ev), d, L in zip(fl, preds, Lpr):
        m = s19.masks_for(meta, pr)
        m['ALL'] = np.ones((len(pr), len(pr)), bool)
        out.append((s19.pair_counts_lab(d, L, m), s19.bacc_lab(d, L, 0.0)))
    return out


def metrics(yc):
    """year_counts の結果から指標 ((P,) ずつ)。W・B は2年合わせ、ほかは年ごと。"""
    res = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for t in ['W', 'B']:
            C = sum(c[t][0] for c, _ in yc)
            N = sum(c[t][1] for c, _ in yc)
            res[t] = np.where(N > 0, C / N, np.nan)
        for (c, b), Y in zip(yc, TEST_YEARS):
            for t in ['W', 'B', 'ALL']:
                res[f'{t if t != "ALL" else "all_auc"}_{Y}'] = np.where(c[t][1] > 0, c[t][0] / c[t][1], np.nan)
            res[f'all_bacc_{Y}'] = b
        res['all_auc'] = np.mean([res[f'all_auc_{Y}'] for Y in TEST_YEARS], axis=0)
        res['all_bacc'] = np.mean([res[f'all_bacc_{Y}'] for Y in TEST_YEARS], axis=0)
    return res


def check_obs(tag, yc, preds, fl, y123, log):
    """確認6: W+B+X = 年ごとの全体の AUC(X は0組)。年ごとの AUC・bacc が sklearn と一致。"""
    ok1 = ok2 = ok3 = True
    nx = []
    for (c, b), (Y, tr, pr, te, ev), d in zip(yc, fl, preds):
        n = sum(c[t][1][0] for t in ['W', 'B', 'X'])
        pooled = sum(c[t][0][0] for t in ['W', 'B', 'X']) / n
        auc = roc_auc_score(y123[te], d[0][ev])
        ok1 &= np.isclose(pooled, c['ALL'][0][0] / c['ALL'][1][0], rtol=0, atol=1e-12) and n == c['ALL'][1][0]
        ok2 &= np.isclose(auc, c['ALL'][0][0] / c['ALL'][1][0], rtol=0, atol=1e-12)
        ok3 &= np.isclose(b[0], balanced_accuracy_score(y123[te], (d[0][ev] > 0).astype(int)), rtol=0, atol=1e-12)
        nx.append(int(c['X'][1][0]))
    log(f'    確認6 {tag}: W+B+X(ペア数で重み付け)= 年ごとの全体の AUC {ok1}(X のペア {nx})、AUC・bacc が sklearn と一致 '
        f'{ok2}・{ok3}')
    assert ok1 and ok2 and ok3 and sum(nx) == 0


# ---------------------------------------------------------------- 確認3
def precheck_pipeline(Xd, yrel, fl, y123, log):
    ok = True
    for Y, tr, pr, te, ev in fl:
        pipe = make_pipeline(s6.Panel('minirocket', random_state=0), StandardScaler(with_mean=False),
                             RidgeRegNTrain()).fit(Xd[tr], yrel[tr])
        a = pipe.predict(Xd[pr])
        Ftr, Fpr = s19._feats(0, Xd, tr, pr)
        b = RidgeRegNTrain().fit(Ftr, yrel[tr]).predict(Fpr)
        ok &= np.allclose(a, b, rtol=0, atol=1e-12) and roc_auc_score(y123[te], a[ev]) == roc_auc_score(y123[te], b[ev])
    log(f'  確認3 Pipeline をそのまま回した結果 = 特徴量を使い回した結果(R_all LOYO、乱数0、2016年・2017年): {ok}')
    assert ok


def precheck_batched(Xd, fl, Ytr, Lpr, meta, log, n_perm=20):
    """確認4: まとめて解いた alpha・予測値 = sklearn RidgeCV を1回ずつ(乱数0)。"""
    res = [s19._fold_check(0, Xd, tr, pr, Y[:n_perm]) for (Yr, tr, pr, te, ev), Y in zip(fl, Ytr)]
    n_bad = n_unsure = n_tot = 0
    pb_all, ps_all = [], []
    for a, p, u, ra, rp in res:
        n_tot += len(a)
        n_unsure += int(u.sum())
        n_bad += int((~(np.isclose(a, ra) & np.isclose(p, rp, rtol=1e-7, atol=1e-10).all(axis=1)) & ~u).sum())
        pb = p.copy()
        pb[u] = rp[u]
        pb_all.append(pb)
        ps_all.append(rp)
    alpha_same = all(np.array_equal(a[~u], ra[~u]) for a, p, u, ra, rp in res)
    ma = metrics(year_counts(pb_all, [L[:n_perm] for L in Lpr], meta, fl))
    mb = metrics(year_counts(ps_all, [L[:n_perm] for L in Lpr], meta, fl))
    ok = all(np.array_equal(ma[k], mb[k], equal_nan=True) for k in ma)
    log(f'  確認4 置換{n_perm}回 × 2年(乱数0): まとめて解いた alpha・予測値 = sklearn RidgeCV を1回ずつ(あいまいでない '
        f'{n_tot - n_unsure} 組のうち不一致 {n_bad}、alpha の一致 {alpha_same}、あいまいな組 {n_unsure})。'
        f'学習し直した後の帰無分布(全指標)が完全に一致 {ok}')
    assert n_bad == 0 and alpha_same and ok


# ---------------------------------------------------------------- 図
def plot_overview(summary, ref):
    mets = [('W', 'W 2年合わせ\n(主の判定)'), ('W_2016', 'W 2016年'), ('W_2017', 'W 2017年'),
            ('all_auc_2016', '全体 2016年'), ('all_auc_2017', '全体 2017年')]
    sm = summary.set_index(['config', 'metric'])
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.26
    for k, c in enumerate(CFGS):
        xs = np.arange(len(mets)) + (k - 1) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.9, color=CFG_COLOR[c], alpha=0.9, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95'] if 'null_q95' in sm.columns else np.nan
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    x0 = 0 - w
    for v, mk, lb in [(ref['stage19'], 'o', '段階19 R_all の W(ランダムな分割)'),
                      (ref['stage20'], 's', '段階20 R_all の W(ランダムな分割)')]:
        ax.plot(x0 + (0.12 if mk == 's' else -0.12), v, marker=mk, ms=7, mfc='none', mec=INK, mew=1.3, ls='',
                zorder=4, label=lb)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点(LOYO)')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    lo = min(0.35, summary['mean'].min() - 0.03)
    ax.set_ylim(lo, max(0.75, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=2)
    ax.set_title('段階21: 年をまたいだ確認(テストの年 2016・2017、label_muniyear_median の評価の件)', fontsize=12,
                 color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stage21_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-perm-m', type=int, default=200, help='M_diff の置換の回数(記述用)')
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    P = args.n_perm
    Pm = min(args.n_perm_m, P)
    s6.N_JOBS = N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  M_diff の置換 {Pm}  並列数 {N_JOBS}')
    lab, X, meta = s20.load_data(log)
    X123, meta123 = s14.load()
    y123 = meta123.y.values

    log('\n[起動時の確認]')
    i123, in123 = s19.check_counts(lab, meta, X, meta123, X123, log)
    folds = build_year_folds(meta, i123, meta123, log)
    tr_m = folds.pop('_m_tr')

    # ---- 置換
    relabel = s19.Relabel(lab)
    yld, my_all = lab['yield'].values, lab.muniyear.to_numpy(dtype=str)
    Yp = np.array([permute_within(yld, my_all, np.random.default_rng([PERM_SEED, i])) for i in range(P)]).reshape(P, -1)
    s19.check_relabel(lab, relabel, meta, i123, meta123, Yp, log)  # 確認5
    gm, lpos = lab.group_mean_yield.values, meta.lab_pos.values
    Yrel_p = np.log(Yp[:, lpos] / gm[lpos])
    Lab_p = np.array([relabel(v) for v in Yp])[:, lpos]
    L_obs = relabel(yld)[lpos]
    yrel = meta.y_rel.values
    dummy = np.zeros(len(meta))
    yr = meta.year.values
    ntc = pd.DataFrame({f'n_test_{Y}': [int((~np.isnan(L) & (yr == Y)).sum()) for L in Lab_p] for Y in TEST_YEARS})
    ntc['n_test'] = ntc.sum(axis=1)
    ntc.rename_axis('perm').to_csv(os.path.join(OUT_DIR, 'null_test_counts.csv'), encoding='utf-8-sig')
    log('  置換ごとのテスト件の数(2016+2017): ' + '、'.join(f'{k}件 {v}回' for k, v in ntc.n_test.value_counts()
                                                    .sort_index().items())
        + '(年ごと: ' + '、'.join(f'{Y}年 {ntc[f"n_test_{Y}"].min()}〜{ntc[f"n_test_{Y}"].max()}件' for Y in TEST_YEARS) + ')')

    fl = folds['R_all_LOYO']
    Ytr = {'R_all_LOYO': [np.vstack([yrel[tr], dummy[tr], Yrel_p[:, tr]]) for _, tr, *_ in fl],
           'R_all_LOYRO': [np.vstack([yrel[tr], dummy[tr]]) for _, tr, *_ in folds['R_all_LOYRO']]}
    Lpr = {'R_all_LOYO': [np.vstack([L_obs[pr], Lab_p[:, pr]]) for _, _, pr, *_ in fl],
           'R_all_LOYRO': [L_obs[pr][None] for _, _, pr, *_ in folds['R_all_LOYRO']]}
    for (Y, tr, pr, te, ev), L in zip(fl, Lpr['R_all_LOYO']):
        assert np.array_equal(np.flatnonzero(~np.isnan(L[0])), np.sort(ev)) and np.array_equal(L[0][ev], y123[te])
    Ym = np.array([permute_within(y123, meta123.muniyear.to_numpy(dtype=str), np.random.default_rng([PERM_SEED, i]))
                   for i in range(Pm)]).reshape(Pm, -1)
    Ym_all = np.vstack([y123, Ym])
    Ytr['M_diff_LOYO'] = [Ym_all[:, t] for t in tr_m]
    Lpr['M_diff_LOYO'] = []
    for Y, tr, pr, te, ev in folds['M_diff_LOYO']:
        L = np.full((1 + Pm, len(pr)), np.nan)
        L[:, ev] = Ym_all[:, te]
        Lpr['M_diff_LOYO'].append(L)

    Xd = first_diff(X)
    precheck_pipeline(Xd, yrel, fl, y123, log)
    precheck_batched(Xd, fl, [Y[2:] for Y in Ytr['R_all_LOYO']], [L[1:] for L in Lpr['R_all_LOYO']], meta, log,
                     n_perm=min(20, P))

    # ---- 本番(乱数 × 年 × 設定をまとめて並列に)
    jobs = []
    for seed in seeds:
        for c in CFGS:
            for k, (Y, tr, pr, te, ev) in enumerate(folds[c]):
                jobs.append((seed, c, k))

    def run(seed, c, k):
        Y, tr, pr, te, ev = folds[c][k]
        if c == 'M_diff_LOYO':
            a, dec, ok, nr = s20._fold_m(seed, Xd, tr, pr, Ytr[c][k])
            return a, dec, ok, nr
        a, p, ok, nr = s19._fold_all(seed, Xd, tr, pr, Ytr[c][k])
        return a[0], np.vstack([p[:1], p[2:]]), ok, nr
    t1 = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(run)(*j) for j in jobs)
    assert all(r[2] for r in res), '観測の目的変数で、まとめて解いた結果が sklearn と一致しない'
    R = {j: r for j, r in zip(jobs, res)}
    log(f'  学習と置換 [{time.time() - t1:.0f}s]、sklearn で学習し直した組 '
        + '、'.join(f'{c} {sum(R[(s, c, k)][3] for s in seeds for k in range(2))}' for c in CFGS))

    per_seed, nulls, scores, spear = [], {c: [] for c in CFGS}, [], []
    test_mask = np.isin(yr, TEST_YEARS)
    for seed in seeds:
        for c in CFGS:
            preds = [R[(seed, c, k)][1] for k in range(2)]
            yc = year_counts(preds, Lpr[c], meta, folds[c])
            check_obs(f'{c} 乱数{seed}', yc, preds, folds[c], y123, log)
            mt = metrics(yc)
            if c != 'R_all_LOYRO':
                nulls[c].append({k: v[1:] for k, v in mt.items()})
            row = {'config': c, 'seed': seed, **{k: v[0] for k, v in mt.items()}}
            ntr = [len(f[1]) for f in folds[c]]
            al = np.array([R[(seed, c, k)][0] for k in range(2)]) / np.array(ntr)
            for Y, n, a in zip(TEST_YEARS, ntr, al):
                row[f'n_train_{Y}'] = n
                row[f'alpha_over_n_{Y}'] = a
            row['alpha_at_max'] = np.isclose(al, BASE_ALPHAS[-1]).mean()
            row['alpha_at_min'] = np.isclose(al, BASE_ALPHAS[0]).mean()
            yc0 = year_counts([p[:1] for p in preds], [L[:1] for L in Lpr[c]], meta, folds[c])
            for (cc, _), Y in zip(yc0, TEST_YEARS):
                for t in ['W', 'B', 'ALL']:
                    row[f'n_pairs_{t}_{Y}'] = cc[t][1][0]
            sc = np.full(len(meta), np.nan)
            for (Y, tr, pr, te, ev), p in zip(folds[c], preds):
                sc[pr] = p[0]
                mm = meta123.iloc[te]
                scores.append(pd.DataFrame({'config': c, 'seed': seed, 'test_year': Y, 'field_id': mm.field_id.values,
                                            'year': mm.year.values, 'muniCd': mm.muniCd.values,
                                            'muniyear': mm.muniyear.values, 'y': mm.y.values,
                                            'y_rel': yrel[i123[te]], 'score': p[0][ev]}))
            sr = pd.DataFrame(s19.within_spearman(sc, meta, test_mask), columns=['muniyear', 'n', 'rho'])
            sr = sr.assign(config=c, seed=seed, year=sr.muniyear.str[-4:].astype(int))
            spear.append(sr)
            d = sr.dropna()
            row['spearman_w'] = np.average(d.rho, weights=d.n)
            for Y in TEST_YEARS:
                dy = d[d.year == Y]
                row[f'spearman_w_{Y}'] = np.average(dy.rho, weights=dy.n)
            per_seed.append(row)
            log(f'    {c:12s} 乱数{seed}: W {row["W"]:.3f}(2016 {row["W_2016"]:.3f}・2017 {row["W_2017"]:.3f})  全体 '
                f'2016 {row["all_auc_2016"]:.3f}・2017 {row["all_auc_2017"]:.3f}  B {row["B"]:.3f}  順位相関 '
                f'{row["spearman_w"]:+.3f}  alpha/n {al[0]:.3g}・{al[1]:.3g}')
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat(scores).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'), index=False, encoding='utf-8-sig')

    # pair_auc(乱数ごと、W・B・全体を年ごとと2年合わせ)
    pa = []
    for _, r in per_seed.iterrows():
        for Y in TEST_YEARS:
            for t, col in [('W', f'W_{Y}'), ('B', f'B_{Y}'), ('ALL', f'all_auc_{Y}')]:
                pa.append({'config': r.config, 'seed': r.seed, 'test_year': Y, 'type': t, 'auc': r[col],
                           'n_pairs': r[f'n_pairs_{t}_{Y}']})
        for t in ['W', 'B']:
            pa.append({'config': r.config, 'seed': r.seed, 'test_year': '2016+2017', 'type': t, 'auc': r[t],
                       'n_pairs': sum(r[f'n_pairs_{t}_{Y}'] for Y in TEST_YEARS)})
    pd.DataFrame(pa).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    # 帰無分布
    nm = {c: {k: np.mean(np.stack([n[k] for n in nulls[c]]), axis=0) for k in nulls[c][0]} for c in nulls if nulls[c]}
    np.save(os.path.join(OUT_DIR, 'null_R_all_LOYO_W.npy'), nm['R_all_LOYO']['W'])
    np.save(os.path.join(OUT_DIR, 'null_R_all_LOYO_all.npy'),
            np.c_[nm['R_all_LOYO']['all_bacc'], nm['R_all_LOYO']['all_auc']])
    np.save(os.path.join(OUT_DIR, 'null_M_diff_LOYO_W.npy'), nm['M_diff_LOYO']['W'])
    np.save(os.path.join(OUT_DIR, 'null_M_diff_LOYO_all.npy'),
            np.c_[nm['M_diff_LOYO']['all_bacc'], nm['M_diff_LOYO']['all_auc']])

    mets = ['W', 'W_2016', 'W_2017', 'all_auc', 'all_auc_2016', 'all_auc_2017', 'all_bacc', 'B', 'B_2016', 'B_2017',
            'spearman_w', 'spearman_w_2016', 'spearman_w_2017']
    summary = []
    for c in CFGS:
        ps = per_seed[per_seed.config == c]
        base = {'config': c, 'n_seeds': len(ps), 'n_perm': {'R_all_LOYO': P, 'M_diff_LOYO': Pm}.get(c, 0),
                'alpha_at_max': ps.alpha_at_max.mean(), 'alpha_at_min': ps.alpha_at_min.mean(),
                **{f'n_train_{Y}': int(ps[f'n_train_{Y}'].iloc[0]) for Y in TEST_YEARS}}
        for met in mets:
            v = ps[met]
            r = {**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max()}
            t = met.split('_')[0] if met.startswith(('W', 'B')) else ('ALL' if met.startswith('all_auc') else None)
            if t and met[-4:].isdigit():
                r['n_pairs'] = ps[f'n_pairs_{t}_{met[-4:]}'].iloc[0]
            elif t in ('W', 'B'):
                r['n_pairs'] = sum(ps[f'n_pairs_{t}_{Y}'].iloc[0] for Y in TEST_YEARS)
            if c in nm and met in nm[c]:
                nl = nm[c][met]
                np_ = base['n_perm']
                r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                          'p': (1 + np.sum(nl >= v.mean())) / (1 + np_)})
            summary.append(r)
    summary = pd.DataFrame(summary)
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')

    sp = pd.concat(spear)
    sg = sp.groupby(['config', 'muniyear']).agg(year=('year', 'first'), n=('n', 'first'), rho_mean=('rho', 'mean'),
                                               rho_sd=('rho', 'std')).reset_index()
    tot = summary[summary.metric.str.startswith('spearman_w')][['config', 'metric', 'mean', 'seed_min', 'seed_max']]
    tot = tot.rename(columns={'metric': 'muniyear', 'mean': 'rho_mean', 'seed_min': 'rho_seed_min',
                              'seed_max': 'rho_seed_max'})
    tot['muniyear'] = tot.muniyear.map({'spearman_w': '(全体: 件数で重み付け)', 'spearman_w_2016': '(2016年)',
                                        'spearman_w_2017': '(2017年)'})
    pd.concat([tot, sg]).to_csv(os.path.join(OUT_DIR, 'within_group_spearman.csv'), index=False, encoding='utf-8-sig')
    ng = sg[sg.config == 'R_all_LOYO']
    log(f'\n[集団の中の順位相関] テストの年の窓のある全件のうち3件以上の集団: '
        + '、'.join(f'{Y}年 {int((ng.year == Y).sum())}集団・{int(ng[ng.year == Y].n.sum())}件' for Y in TEST_YEARS))

    # ランダムな分割(段階19・20)との比較
    s19s = pd.read_csv(os.path.join(OUT19, 'summary.csv'), encoding='utf-8-sig')
    s20s = pd.read_csv(os.path.join(OUT20, 'summary.csv'), encoding='utf-8-sig')
    cr = []
    for c_rand, c21 in [('R_all', 'R_all_LOYO'), ('R_all', 'R_all_LOYRO'), ('M_diff', 'M_diff_LOYO')]:
        for met in ['W', 'all_auc', 'B', 'spearman_w', 'all_bacc']:
            a = s19s[(s19s.config == c_rand) & (s19s.metric == met)].iloc[0]
            b = s20s[(s20s.config == c_rand) & (s20s.metric == met)].iloc[0]
            d = summary[(summary.config == c21) & (summary.metric == met)].iloc[0]
            cr.append({'config_random': c_rand, 'config_stage21': c21, 'metric': met,
                       'stage19_mean': a['mean'], 'stage19_null_q95': a.get('null_q95'), 'stage19_p': a.get('p'),
                       'stage20_mean': b['mean'], 'stage20_null_q95': b.get('null_q95'), 'stage20_p': b.get('p'),
                       'stage21_mean': d['mean'], 'stage21_null_q95': d.get('null_q95'), 'stage21_p': d.get('p'),
                       'note': '段階21の all_auc・all_bacc は年ごとの値の平均、W・B は2年合わせ' if met != 'spearman_w'
                       else '段階21はテストの年(2016・2017)の集団'})
    cr = pd.DataFrame(cr)
    cr.to_csv(os.path.join(OUT_DIR, 'compare_random_split.csv'), index=False, encoding='utf-8-sig')
    log('\n[ランダムな分割(段階19・20)との比較]')
    log(cr.drop(columns='note').round(3).to_string(index=False))

    w19 = s19s[(s19s.config == 'R_all') & (s19s.metric == 'W')]['mean'].iloc[0]
    w20 = s20s[(s20s.config == 'R_all') & (s20s.metric == 'W')]['mean'].iloc[0]
    plot_overview(summary, {'stage19': w19, 'stage20': w20})
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95', 'n_pairs',
            'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.config == 'R_all_LOYO') & (summary.metric == 'W')].iloc[0]
    hit = bool(r['mean'] > r.null_q95)
    log(f'\n[主の判定] R_all LOYO の W(2016+2017、{int(r.n_pairs)}組){r["mean"]:.3f}(乱数 {r.seed_min:.3f}〜'
        f'{r.seed_max:.3f})、W の帰無分布の95%点 {r.null_q95:.3f}、p={r.p:.3f} → ' + (
            '上回る: 別の年のデータだけで学習しても、相対収量の回帰モデルは、同じ地域・年の中の多収・低収を偶然以上に見分けた'
            if hit else
            '上回らない: 別の年への当てはまりは確かめられなかった。段階19・20の W の結果は、同じ年の圃場を学習に含む条件で'
            'のものにとどまる'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
