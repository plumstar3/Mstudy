"""
段階22: 期間の重要度 — 開花後のどの期間の水分の動きが効くか(EXP-031)。

主の設定 M_diff(label_muniyear_median の123件、開花起点 0〜+59 日の1次差分 → MiniRocket → RidgeClassifierCV)で、
60日の窓の中の短い部分窓(各20日)だけで学習と評価をやり直し、「その期間だけでどれだけ当てられるか」を比べる。
仮説の材料: 大豆は莢伸長期〜子実肥大期(R3〜R6)の水分ストレスが収量に効きやすい(調査資料。一次文献は未確認)。

部分窓(事前に固定): P1 0〜19、P2 10〜29、P3 20〜39、P4 30〜49、P5 40〜59(開花後の日)。生の VWC 20点を切り出してから
1次差分(19点)を取る。モデル・分割・置換は段階13の median・M と同じ(乱数0〜9、置換1000回、市町村×年の中で並べ替え)。
比較の基準: 60日全体(M_diff、段階13の結果)。
判定(事前に固定)
  問い1(主): 5つの部分窓の全体 AUC の置換検定の p を Holm 法で補正し、補正後 p < 0.05 の窓があるか
  問い2: P5 − P1(同じ分割・同じ乱数)の繰り返し平均の 2.5〜97.5%点が0を含まないか
参考(記述のみ): 60日の1次差分(59点)の10日の区間(0〜9, …, 50〜58)の1つを0にして学習し直し、60日全体との差を見る(置換なし)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage22\\run_stage22.py
  試運転: ... run_stage22.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage22/
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

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13', 'stage14', 'stage16']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage13 as s13  # noqa: E402
import run_stage14 as s14  # noqa: E402
import run_stage16 as s16  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY = 'median'
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage22')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
WINDOWS = {'P1': (0, 19), 'P2': (10, 29), 'P3': (20, 39), 'P4': (30, 49), 'P5': (40, 59)}  # 開花後の日(両端を含む)
OCC = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 58)]  # 60日の1次差分(59点)の区間(添字、両端を含む)
TYPES = ['W', 'B', 'X']

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


def sub_diff(X, a, b):
    """開花後 a〜b 日(両端を含む)の生の VWC を切り出してから1次差分。"""
    return first_diff(X[:, :, a:b + 1])


def holm(p):
    """Holm 法で補正した p(単調にし、1で頭打ち)。"""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


# ---------------------------------------------------------------- 起動時の確認
def check_slice(X, meta, log):
    """確認2: 1件について、元の VWC 時系列から開花後 20〜39 日を手で切り出した値と P3 の値が一致する。"""
    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig', parse_dates=['date'])
    i = len(meta) // 2
    r = meta.iloc[i]
    fl = pd.Timestamp(r.flower_date)
    s = ts[(ts.field_id == r.field_id) & (ts.year == r.year)].set_index('date').VWC
    hand = s.reindex(pd.date_range(fl + pd.Timedelta(days=20), fl + pd.Timedelta(days=39))).values
    a, b = WINDOWS['P3']
    ok = np.array_equal(hand, X[i, 0, a:b + 1]) and np.allclose(np.diff(hand), sub_diff(X, a, b)[i, 0], rtol=0, atol=0)
    log(f'  確認2 部分窓の切り出し: 圃場 {r.field_id}({r.year}、開花日 {fl.date()})の {(fl + pd.Timedelta(days=20)).date()}〜'
        f'{(fl + pd.Timedelta(days=39)).date()} を元の時系列から手で切り出した20点 = P3 の生の値、その1次差分 = P3 の入力: {ok}')
    assert ok


def minirocket_info(Xd, log):
    """短い系列での MiniRocket の特徴量の数と dilation(乱数0、123件で fit)。"""
    from sktime.transformations.panel.rocket import MiniRocket
    t = MiniRocket(num_kernels=s6.N_KERNELS, random_state=0).fit(Xd)
    d, nf, _ = t.parameters
    nfeat = t.transform(Xd).shape[1]
    return nfeat, list(map(int, d)), list(map(int, nf))


def check_pipeline(name, Xd, y, sp, log):
    """確認3: Pipeline をそのまま回した結果 = 特徴量を使い回した結果(乱数0、先頭10 fold)。"""
    a = evaluate(s6.pipeline('minirocket', 0), Xd, y, sp[:10])
    b = s6.eval_cached(s6.fold_features('minirocket', 0, Xd, sp[:10]), y, sp[:10])
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'  確認3 Pipeline をそのまま回した結果 = 特徴量を使い回した結果({name}、乱数0、先頭10 fold): {ok}')
    assert ok


# ---------------------------------------------------------------- 参考: 区間を消す
def occlusion(Xd_full, y, sp, seeds, mfolds, log):
    rows, fold_rows = [], []
    for a, b in OCC:
        Xo = Xd_full.copy()
        Xo[:, :, a:b + 1] = 0.0
        t = time.time()
        fs = []
        for seed in seeds:
            f = s6.eval_cached(s6.fold_features('minirocket', seed, Xo, sp), y, sp).assign(seed=seed)
            fs.append(f)
        f = pd.concat(fs)
        name = f'消す {a}〜{b}'
        fold_rows.append(f.assign(config=name))
        fM = mfolds[mfolds.seed.isin(seeds)][['seed', 'repeat', 'fold', 'auc', 'bacc']]
        ps = s7.paired_summary(s7.paired(f[['seed', 'repeat', 'fold', 'auc', 'bacc']], fM, f'{name} − 60日全体'))
        d = ps[ps.metric == 'auc'].iloc[0]
        per_seed = f.groupby('seed').auc.mean()
        rows.append({'segment': name, 'day_start': a, 'day_end': b + 1, 'auc': per_seed.mean(),
                     'auc_seed_min': per_seed.min(), 'auc_seed_max': per_seed.max(), 'bacc': f.groupby('seed').bacc.mean().mean(),
                     'diff_auc': d.mean_diff, 'diff_q025': d.rep_q025, 'diff_q975': d.rep_q975,
                     'diff_frac_neg': d.rep_frac_neg, 'alpha_at_max': np.isclose(f.alpha_over_n, s6.BASE_ALPHAS[-1]).mean()})
        log(f'    {name}(1次差分の添字。開花後 {a}〜{b + 1} 日の変化): AUC {per_seed.mean():.3f}、60日全体との差 '
            f'{d.mean_diff:+.3f} [{d.rep_q025:+.3f}, {d.rep_q975:+.3f}]  [{time.time() - t:.0f}s]')
    return pd.DataFrame(rows), pd.concat(fold_rows)


# ---------------------------------------------------------------- 図
def plot(summary, occ, null_full_q95):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [3, 2]})
    ax = axs[0]
    sm = summary.set_index('config')
    full = sm.loc['60日全体']
    for k, (nm, (a, b)) in enumerate(WINDOWS.items()):
        r = sm.loc[nm]
        c = (a + b) / 2
        ax.errorbar(c, r.auc, yerr=[[r.auc - r.auc_seed_min], [r.auc_seed_max - r.auc]], fmt='o', ms=8,
                    color='#2a78d6', ecolor='#2a78d6', capsize=3, zorder=3)
        ax.hlines(r.null_q95, c - 3, c + 3, color=s6.C_RED, lw=2, ls='--')
        ax.hlines(0.47 + 0.008 * k, a, b, color=MUTED, lw=3, alpha=0.6)
        ax.text(c, r.auc + 0.012, f'{nm}\n{r.auc:.3f}\n(Holm p={r.p_holm:.2f})', ha='center', fontsize=8.5, color=INK)
    ax.axhline(full.auc, color='#9a9890', lw=1.5, label=f'60日全体(M_diff){full.auc:.3f}')
    ax.axhline(null_full_q95, color='#9a9890', lw=1.2, ls='--', label=f'60日全体の95%点 {null_full_q95:.3f}')
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.plot([], [], 'o', color='#2a78d6', label='部分窓(20日)の全体 AUC(線 = 乱数の幅)')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='部分窓の帰無分布の95%点')
    ax.set_xlim(-2, 61)
    ax.set_ylim(0.46, max(0.70, summary.auc.max() + 0.06))
    ax.set_ylabel('全体の AUC', color=INK)
    ax.set_title('段階22: 開花後の部分窓(20日)だけで学習・評価した AUC(下の灰色の帯 = 部分窓の範囲)', fontsize=11,
                 color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    s6.style(ax)
    ax = axs[1]
    for _, r in occ.iterrows():
        c = (r.day_start + r.day_end) / 2
        ax.bar(c, r.diff_auc, width=8.5, color='#eb6834', alpha=0.75)
        ax.errorbar(c, r.diff_auc, yerr=[[r.diff_auc - r.diff_q025], [r.diff_q975 - r.diff_auc]], fmt='none',
                    ecolor=INK, capsize=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-2, 61)
    ax.set_xlabel('開花後の日', color=INK)
    ax.set_ylabel('AUC の差(消した − 60日全体)', color=INK)
    ax.set_title('参考: 1次差分の10日の区間を0にして学習し直したときの AUC の差(線 = 繰り返し平均の2.5〜97.5%点)',
                 fontsize=10, color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'period_importance.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    P = args.n_perm
    s6.N_JOBS = N_JOBS
    s16.OUT_DIR = OUT_DIR  # run_m が帰無分布の .npy を書く場所を段階22の出力にする
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {N_JOBS}')
    X, meta = s14.load()
    y = meta.y.values
    log('\n[起動時の確認]')
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    st = split_table(sp, y, meta.muniCd.values)
    ref = pd.read_csv(os.path.join(S13_DIR, 'splits.csv'), encoding='utf-8-sig')
    ref = ref[ref.label == KEY].reset_index(drop=True)
    cols = ['repeat', 'fold', 'n_train', 'n_test', 'n_test_pos', 'n_test_neg', 'n_test_groups']
    ok1 = len(meta) == 123 and X.shape == (123, 1, 60) and st[cols].reset_index(drop=True).equals(ref[cols])
    log(f'  確認1 件数 {len(meta)}(多収 {int(y.sum())} / 低収 {int((1 - y).sum())})、X {X.shape}、分割が段階13の splits.csv と一致: {ok1}')
    assert ok1
    check_slice(X, meta, log)

    D = {nm: sub_diff(X, a, b) for nm, (a, b) in WINDOWS.items()}
    Xd_full = first_diff(X)
    info = {}
    for nm, Xd in list(D.items()) + [('60日全体', Xd_full)]:
        nfeat, dil, nf = minirocket_info(Xd, log)
        info[nm] = (nfeat, dil)
        log(f'  MiniRocket {nm}: 入力 {Xd.shape[-1]} 点、特徴量 {nfeat}、dilation {dil}(dilation ごとの特徴量の組 {nf})')
        assert nfeat >= 0.9 * 9996, f'{nm}: 特徴量の数が大きく減った。止めて報告する'
    check_pipeline('P1', D['P1'], y, sp, log)
    check_pipeline('P5', D['P5'], y, sp, log)
    log('  P3 で段階13の確認(Pipeline、置換5回で分割が変わらない、まとめて解いた置換20回 = sklearn):')
    s13.prechecks(KEY, D['P3'], meta, sp, log)  # 確認4(まとめて解いた置換)と確認5(置換で分割が変わらない)
    mfolds, S_m, seeds13, summ_m, pa_m = s16.m_diff_from_stage13(meta, sp, Xd_full, log)
    ps = s13.perm_sets(KEY, meta, P)

    # ---- 部分窓
    res = {}
    for nm in WINDOWS:
        log(f'\n[{nm}: 開花後 {WINDOWS[nm][0]}〜{WINDOWS[nm][1]} 日]')
        res[nm] = s16.run_m(nm, D[nm], meta, sp, ps, seeds, P, log)

    # ---- まとめ
    rows = []
    for nm in list(WINDOWS) + ['60日全体']:
        if nm == '60日全体':
            sm = pd.DataFrame(summ_m).set_index('metric')
            r = {'config': nm, 'day_start': 0, 'day_end': 59, 'n_seeds': int(sm.n_seeds.iloc[0]), 'source': '段階13'}
            alpha = sm.alpha_at_max.iloc[0]
        else:
            sm = pd.DataFrame(res[nm]['summary']).set_index('metric')
            r = {'config': nm, 'day_start': WINDOWS[nm][0], 'day_end': WINDOWS[nm][1], 'n_seeds': len(seeds), 'source': '段階22'}
            alpha = sm.alpha_at_max.iloc[0]
        r.update({'n_features': info[nm][0], 'dilations': ' '.join(map(str, info[nm][1])), 'alpha_at_max': alpha,
                  'auc': sm.loc['all_auc', 'mean'], 'auc_seed_min': sm.loc['all_auc', 'seed_min'],
                  'auc_seed_max': sm.loc['all_auc', 'seed_max'], 'null_mean': sm.loc['all_auc', 'null_mean'],
                  'null_q95': sm.loc['all_auc', 'null_q95'], 'p': sm.loc['all_auc', 'p'],
                  'bacc': sm.loc['all_bacc', 'mean'], 'bacc_null_q95': sm.loc['all_bacc', 'null_q95'],
                  'bacc_p': sm.loc['all_bacc', 'p'], 'n_perm': P if nm != '60日全体' else int(sm.n_perm.iloc[0])})
        for t in TYPES:
            r[t] = sm.loc[t, 'mean']
            if t != 'X':
                r[f'{t}_null_q95'] = sm.loc[t, 'null_q95']
                r[f'{t}_p'] = sm.loc[t, 'p']
        rows.append(r)
    summary = pd.DataFrame(rows)
    win = summary.config.isin(list(WINDOWS))
    summary.loc[win, 'p_holm'] = holm(summary.loc[win, 'p'].values)
    summary['holm_significant'] = (summary.p_holm < 0.05).astype(object).where(win, None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')

    folds = pd.concat([f for nm in WINDOWS for f in res[nm]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for nm in WINDOWS for r in res[nm]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                       index=False, encoding='utf-8-sig')
    pd.concat([s for nm in WINDOWS for s in res[nm]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                       index=False, encoding='utf-8-sig')
    pd.concat([pa_m.assign(config='60日全体')] + [p for nm in WINDOWS for p in res[nm]['pair']]).to_csv(
        os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    # ---- 対ごとの差
    cols = ['seed', 'repeat', 'fold', 'auc', 'bacc']
    fw = {nm: folds[folds.config == nm][cols] for nm in WINDOWS}
    fM = mfolds[mfolds.seed.isin(seeds)][cols]
    names = list(WINDOWS)
    pairs = [s7.paired(fw['P5'], fw['P1'], 'P5 − P1')]
    pairs += [s7.paired(fw[nm], fM, f'{nm} − 60日全体') for nm in names]
    pairs += [s7.paired(fw[b], fw[a], f'{b} − {a}') for a, b in zip(names[:-1], names[1:])]
    pds = s7.paired_summary(pd.concat(pairs))
    pds['ci_excludes_0'] = (pds.rep_q025 > 0) | (pds.rep_q975 < 0)
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差(同じ分割・同じ乱数、繰り返し平均の2.5〜97.5%点)]')
    for _, r in pds[pds.metric == 'auc'].iterrows():
        log(f'  {r.comparison}: AUC {r.mean_diff:+.3f} [{r.rep_q025:+.3f}, {r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}')

    # ---- 参考: 区間を消す
    log('\n[参考: 60日の1次差分の10日の区間を0にして学習し直す(置換なし)]')
    occ, occ_folds = occlusion(Xd_full, y, sp, seeds, mfolds, log)
    occ.to_csv(os.path.join(OUT_DIR, 'occlusion_summary.csv'), index=False, encoding='utf-8-sig')

    plot(summary, occ, summary.set_index('config').loc['60日全体', 'null_q95'])
    show = ['config', 'day_start', 'day_end', 'auc', 'auc_seed_min', 'auc_seed_max', 'null_q95', 'p', 'p_holm', 'bacc',
            'bacc_p', 'W', 'W_p', 'B', 'B_p', 'X', 'n_features', 'dilations', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    sig = summary[win & (summary.p_holm < 0.05)]
    log('\n[問い1(主)] Holm 補正後 p < 0.05 の部分窓: '
        + ('、'.join(f'{r.config}(開花後 {r.day_start}〜{r.day_end} 日、AUC {r.auc:.3f}、p {r.p:.3f} → Holm {r.p_holm:.3f})'
                    for _, r in sig.iterrows()) if len(sig) else 'なし')
        + '。補正前: ' + '、'.join(f'{r.config} p={r.p:.3f}' for _, r in summary[win].iterrows()))
    d = pds[(pds.comparison == 'P5 − P1') & (pds.metric == 'auc')].iloc[0]
    log(f'[問い2] P5 − P1: AUC {d.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{d.rep_q025:+.3f}, {d.rep_q975:+.3f}] → '
        + ('0を含まない' if d.ci_excludes_0 else '0を含む(差を検出できない)'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
