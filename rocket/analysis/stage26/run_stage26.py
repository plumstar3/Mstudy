"""
段階26: 1次差分の分布の情報だけで分類できるか(ソートした1次差分、EXP-035)。

段階13の median(123件、開花起点 0〜+59 日)の1次差分(59点)を、系列ごとに小さい順に並べ替えた系列
(ソートした1次差分)を入力にする。時間の順番の情報は消え、変化量の分布の情報は残る。
  S_sort(主)  ソートした1次差分 → D と同じ MiniRocket(10000、乱数0〜9)→ StandardScaler(with_mean=False)
               → RidgeClassifierCV(logspace(−3, 3, 10) × n_train)。置換1000回
  L_sort(記述) ソートした59個の値 → StandardScaler → LogisticRegression(C=1)。置換200回。位置ごとの係数
  Q(記述)      1次差分の 5・10・25・50・75・90・95%点、歪度、尖度 → StandardScaler → LR(C=1)。置換200回
  D(比べる相手) 段階13の median・M(保存済みの結果。乱数0の先頭10 fold を計算し直して一致を確かめる)
分割: make_splits(y, muniCd, 5, 20, 0)。置換: 市町村×年の中で並べ替え(置換 i の乱数は [0, i])。
判定(事前に固定、1つ): S_sort の全体の AUC(乱数10通りの平均)が帰無分布の95%点を上回るか。
副(記述): D − S_sort(繰り返し平均の 2.5〜97.5%点)、(S_sort − 0.5) / (D − 0.5)、W・B、段階25のシャッフルとの比較、
alpha の張り付き。ソートした系列への MiniRocket の適用は、時間の並びを見る本来の使い方ではなく、
ROCKET が使っている情報を切り分けるための使い方。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage26\\run_stage26.py
  試運転: ... run_stage26.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage26/
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
from scipy.stats import kurtosis, skew
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage13'))
import run_stage13 as s13  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

s6, s12, s5 = s13.s6, s13.s12, s13.s5
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage26')
DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13', 'label_muniyear_median')
S13 = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
S25 = os.path.join(ROCKET_DIR, 'outputs', 'stage25')
KEY = 'median'
N_EXPECT = 123
QS = [5, 10, 25, 50, 75, 90, 95]
TYPES = ['W', 'B', 'X']
POWER_Z = s13.POWER_Z
INK, MUTED = s6.INK, s6.MUTED
COL = {'D': '#52514e', 'S_sort': '#2a78d6', 'L_sort': '#eb6834', 'Q': '#3a9a5b'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def q_features(Xd):
    d = Xd[:, 0, :]
    return np.c_[np.percentile(d, QS, axis=1).T, skew(d, axis=1), kurtosis(d, axis=1)]


# ---------------------------------------------------------------- 1つの設定
def run_config(config, kind, X, meta, sp, seeds, P, log):
    """kind: 'M'(MiniRocket、X は (n, 1, L))か 'LR'(X は (n, p))。段階13の run_label と同じ集計。"""
    y = meta.y.values
    ps = s13.perm_sets(KEY, meta, P)
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': [], 'summary': []}
    nulls = {'all': [], 'W': [], 'B': []}
    t0 = time.time()
    for seed in (seeds if kind == 'M' else [None]):
        if kind == 'M':
            feats = s6.fold_features('minirocket', seed, X, sp)
            if seed == seeds[0]:
                F0 = np.vstack([feats[0][0], feats[0][1]])
                allf = np.vstack([np.vstack(f) for f in feats[:10]])
                log(f'    特徴量(乱数{seed}、fold 0 の学習 + テスト {F0.shape[0]} 件 × {F0.shape[1]} 列): '
                    f'全件で同じ値の列 {int((F0.std(axis=0) == 0).sum())}、'
                    f'先頭10 fold の NaN {int(np.isnan(allf).sum())}・inf {int(np.isinf(allf).sum())}')
                assert np.isfinite(allf).all()
            ref = s6.eval_cached(feats, y, sp)
            decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
            del feats
        else:
            decs, n_redo = s12.lr_decisions(X, y, sp, ps[2]), 0
            ref = evaluate(s5.lr(), X, y, sp).assign(n_train=[len(tr) for _, _, tr, _ in sp], alpha_over_n=np.nan)
        obs, null = s13.analyse(decs, meta, sp, ps)
        s13.check_obs(f'{config} 乱数{seed}', obs, ref, log)
        out['folds'].append(ref.assign(config=config, seed=seed))
        for k in nulls:
            nulls[k].append(null[k])
        for (r, f, tr, te), dec in zip(sp, decs):
            m = meta.iloc[te]
            out['scores'].append(pd.DataFrame({'config': config, 'seed': seed, 'repeat': r, 'fold': f,
                                               'field_id': m.field_id.values, 'year': m.year.values,
                                               'muniyear': m.muniyear.values, 'y': m.y.values, 'score': dec[0]}))
        row = {'config': config, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
               'alpha_at_max': np.isclose(ref.alpha_over_n, s6.BASE_ALPHAS[-1]).mean() if kind == 'M' else np.nan}
        for t in TYPES:
            a, n = obs[t]
            row[t] = np.nanmean(a)
            out['pair'].append(pd.DataFrame({'config': config, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
        out['seed'].append(row)
        if kind == 'M':
            log(f'    {config} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
                + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES) + f'(学習し直した組 {n_redo})')
    per_seed = pd.DataFrame(out['seed'])
    pa = pd.concat(out['pair'])
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    np.save(os.path.join(OUT_DIR, f'null_{config}.npy'), null_all)
    base = {'config': config, 'n': len(y), 'n_seeds': len(per_seed), 'n_perm': P}
    for met, j in [('all_auc', 1), ('all_bacc', 0)]:
        v, nl = per_seed[met], null_all[:, j]
        out['summary'].append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                               'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                               'p': (1 + np.sum(nl >= v.mean())) / (1 + P), 'alpha_at_max': per_seed.alpha_at_max.mean()})
    for t in TYPES:
        v = per_seed[t]
        r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
             'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean(),
             'alpha_at_max': per_seed.alpha_at_max.mean()}
        if t in ('W', 'B'):
            nl = np.mean(np.stack(nulls[t]), axis=0)
            np.save(os.path.join(OUT_DIR, f'null_{config}_{t}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                      'p': (1 + np.sum(nl >= v.mean())) / (1 + P), 'auc_needed_80pct': 0.5 + POWER_Z * np.nanstd(nl)})
        out['summary'].append(r)
    s = pd.DataFrame(out['summary'])
    log(f'  {config:6s} ' + '  '.join(
        f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95')) else '')
        for _, r in s.iterrows()) + f'  [{time.time() - t0:.0f}s]')
    return out


def load_d():
    """段階13の median・M の保存済みの結果を、この段階の表の形に合わせる。"""
    sm = pd.read_csv(os.path.join(S13, 'summary.csv'), encoding='utf-8-sig')
    sm = sm[(sm.label == KEY) & (sm.config == 'M') & sm.metric.isin(['all_auc', 'all_bacc'] + TYPES)].copy()
    sm['config'] = 'D'
    sm = sm.drop(columns=['label'])
    folds = pd.read_csv(os.path.join(S13, 'folds.csv'), encoding='utf-8-sig')
    folds = folds[(folds.label == KEY) & (folds.config == 'M')].copy()
    pair = pd.read_csv(os.path.join(S13, 'pair_auc.csv'), encoding='utf-8-sig')
    pair = pair[(pair.label == KEY) & (pair.config == 'M')].copy()
    folds['config'] = pair['config'] = 'D'
    return sm, folds, pair


# ---------------------------------------------------------------- 図
def plot_profiles(Xs, y, uv, coef):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    pos = np.arange(1, Xs.shape[-1] + 1)
    ax = axs[0]
    for lab, c, nm in [(0, '#eb6834', '低収'), (1, '#2a78d6', '多収')]:
        d = Xs[y == lab, 0, :]
        m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
        ax.plot(pos, m, color=c, lw=1.8, label=f'{nm}({len(d)}件)')
        ax.fill_between(pos, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.15)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xlabel('小さい方から何番目か(1〜59)')
    ax.set_ylabel('1次差分(VWC の日ごとの変化)')
    ax.set_title('ソートした1次差分の平均(帯 = 平均の95%区間)', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ax.bar(pos, uv.auc - 0.5, bottom=0.5, color=np.where(uv.auc >= 0.5, '#2a78d6', '#eb6834'), alpha=0.7)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xlabel('小さい方から何番目か')
    ax.set_ylabel('単変量 AUC(多収が大きいほど > 0.5)')
    ax.set_title('位置ごとの単変量 AUC(全123件、学習なし)', loc='left', fontsize=11, color=INK)
    ax = axs[2]
    ax.errorbar(pos, coef.coef_mean, yerr=coef.coef_sd, fmt='o', ms=3, color='#eb6834', ecolor='#f2b48f', lw=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('小さい方から何番目か')
    ax.set_ylabel('係数(標準化後、+ は多収寄り)')
    ax.set_title('L_sort の係数(100 fold の平均 ± SD)', loc='left', fontsize=11, color=INK)
    for a in axs:
        s6.style(a)
    fig.suptitle('段階26: ソートした1次差分(変化量の分布)の多収・低収の違い(記述用)', fontsize=12.5, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'sorted_profiles.png'), dpi=150)
    plt.close(fig)


def plot_overview(summary, shuffle_means):
    cfgs = ['D', 'S_sort', 'L_sort', 'Q']
    mets = [('all_auc', '全体 AUC'), ('W', '地域内 W'), ('B', '地域間 B')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    sm = summary.set_index(['config', 'metric'])
    for ax, (met, nm) in zip(axs, mets):
        for i, c in enumerate(cfgs):
            if (c, met) not in sm.index:
                continue
            r = sm.loc[(c, met)]
            ax.bar(i, r['mean'], color=COL[c], alpha=0.75)
            ax.text(i, r['mean'] + 0.008, f'{r["mean"]:.3f}', ha='center', fontsize=9, color=INK)
            if pd.notna(r.get('null_q95')):
                ax.hlines(r.null_q95, i - 0.4, i + 0.4, color=s6.C_RED, lw=2, ls='--')
        key = {'all_auc': 'auc', 'W': 'W', 'B': 'B'}[met]
        for v, mk in [('S2a', 'v'), ('S2b', '^')]:
            ax.plot(0.5, shuffle_means.loc[v, key], marker=mk, color=INK, ms=7, ls='',
                    label=f'段階25 {v} の平均 {shuffle_means.loc[v, key]:.3f}')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks(range(len(cfgs)), cfgs)
        ax.set_title(nm, loc='left', fontsize=11, color=INK)
        ax.set_ylim(0.35, 0.72)
        s6.style(ax)
    axs[0].set_ylabel('AUC')
    axs[0].legend(frameon=False, fontsize=8, loc='lower left')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=s6.C_RED, lw=2, ls='--')], ['帰無分布の95%点'], loc='upper right',
               frameon=False, fontsize=9)
    fig.suptitle('段階26: D・S_sort(ソートした1次差分 + MiniRocket)・L_sort・Q の比較', fontsize=12.5, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'stage26_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000, help='S_sort の置換の数')
    ap.add_argument('--n-perm-lr', type=int, default=200, help='L_sort・Q の置換の数')
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    P, P_lr = args.n_perm, min(args.n_perm_lr, args.n_perm)
    s6.N_JOBS = s13.N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {s13.N_JOBS}')

    # ---- データ・分割・確認1〜4
    X = np.load(DATA + '_X.npy')
    meta = pd.read_csv(DATA + '_meta.csv', encoding='utf-8-sig', dtype={'muniCd': str})
    y = meta.y.values
    Xd = first_diff(X)
    Xs = np.sort(Xd, axis=-1)
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認1 段階26・EXP-035 が未使用(docs・CLAUDE.md に EXP-035 なし、outputs/stage26 は今回作成): '
        f'{"EXP-035" not in docs}')
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    ref = pd.read_csv(os.path.join(S13, 'splits.csv'), encoding='utf-8-sig')
    ref = ref[ref.label == KEY].drop(columns='label').reset_index(drop=True)
    mine = split_table(sp, y, meta.muniCd.values)
    ok2 = len(meta) == N_EXPECT and mine.shape == ref.shape and np.allclose(mine.values.astype(float),
                                                                             ref.values.astype(float))
    log(f'確認2 件数 {len(meta)}(多収 {int(y.sum())} / 低収 {int((1 - y).sum())})・分割が段階13の median と同じ: {ok2}')
    ok3 = np.array_equal(np.sort(Xd, axis=-1), Xs) and bool((np.diff(Xs, axis=-1) >= 0).all())
    log(f'確認3 ソートの後も各系列の値の集まりが同じで、小さい順に並んでいる: {ok3}'
        f'(1系列目: 最小 {Xs[0, 0, 0]:.4f}、中央 {Xs[0, 0, 29]:.4f}、最大 {Xs[0, 0, -1]:.4f})')
    f13 = pd.read_csv(os.path.join(S13, 'folds.csv'), encoding='utf-8-sig')
    f13 = f13[(f13.label == KEY) & (f13.config == 'M') & (f13.seed == 0)].reset_index(drop=True)
    d10 = s6.eval_cached(s6.fold_features('minirocket', 0, Xd, sp[:10]), y, sp[:10])
    ok4 = np.allclose(d10.auc, f13.auc[:10], rtol=0, atol=1e-12, equal_nan=True) and \
        np.allclose(d10.bacc, f13.bacc[:10], rtol=0, atol=1e-12)
    log(f'確認4 D の再計算(乱数0、先頭10 fold)が段階13と一致: {ok4}')
    assert ok2 and ok3 and ok4
    log('確認5〜7(S_sort):')
    s13.prechecks(KEY, Xs, meta, sp, log)   # 2 = Pipeline と使い回し、4 = 置換で分割が変わらない、3 = まとめて解いた置換

    # ---- 設定
    res = {}
    log('\n[S_sort(主): ソートした1次差分 + MiniRocket]')
    res['S_sort'] = run_config('S_sort', 'M', Xs, meta, sp, seeds, P, log)
    log('\n[L_sort(記述): ソートした59個の値 + LR]')
    res['L_sort'] = run_config('L_sort', 'LR', Xs[:, 0, :], meta, sp, seeds, P_lr, log)
    log('\n[Q(記述): 分位点7つ + 歪度 + 尖度 + LR]')
    Fq = q_features(Xd)
    res['Q'] = run_config('Q', 'LR', Fq, meta, sp, seeds, P_lr, log)

    # ---- D と合わせた表
    d_sm, d_folds, d_pair = load_d()
    if len(seeds) < 10:
        d_folds = d_folds[d_folds.seed.isin(seeds)]
        d_pair = d_pair[d_pair.seed.isin(seeds)]
    summary = pd.concat([d_sm.reindex(columns=pd.DataFrame(res['S_sort']['summary']).columns.union(d_sm.columns)),
                         *[pd.DataFrame(res[c]['summary']) for c in res]], ignore_index=True)
    d_auc = summary[(summary.config == 'D') & (summary.metric == 'all_auc')]['mean'].iloc[0]
    m = summary.metric == 'all_auc'
    summary.loc[m, 'share_of_D_above_chance'] = (summary.loc[m, 'mean'] - 0.5) / (d_auc - 0.5)
    has = summary.null_q95.notna()
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(has, None)
    keep = ['config', 'metric', 'n', 'n_seeds', 'n_perm', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95',
            'p', 'above_q95', 'n_pairs_per_repeat', 'auc_needed_80pct', 'alpha_at_max', 'share_of_D_above_chance']
    summary = summary.reindex(columns=keep)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat([d_folds.drop(columns=['label'], errors='ignore')] + [f for c in res for f in res[c]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for c in res for r in res[c]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False,
                                                                  encoding='utf-8-sig')
    pd.concat([s for c in res for s in res[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([d_pair.drop(columns=['label'], errors='ignore')] + [p for c in res for p in res[c]['pair']]) \
        .to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    # ---- D − S_sort(同じ分割・同じ乱数)
    a = pd.concat(res['S_sort']['folds'])[['seed', 'repeat', 'fold', 'auc', 'bacc']]
    b = d_folds[['seed', 'repeat', 'fold', 'auc', 'bacc']]
    j = b.merge(a, on=['seed', 'repeat', 'fold'], suffixes=('_D', '_S'))
    assert len(j) == len(a)
    rows = []
    for met in ['auc', 'bacc']:
        diff = (j[f'{met}_D'] - j[f'{met}_S'])
        rep = j.assign(d=diff).groupby('repeat').d.mean()
        rows.append({'comparison': 'D − S_sort', 'metric': met, 'mean': diff.mean(), 'repeat_q025': rep.quantile(.025),
                     'repeat_q975': rep.quantile(.975), 'share_repeat_positive': (rep > 0).mean(),
                     'n_repeats': len(rep), 'n_seeds': j.seed.nunique()})
    pdiff = pd.DataFrame(rows)
    pdiff.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    # ---- L_sort の係数、位置ごとの単変量 AUC
    F = Xs[:, 0, :]
    coefs = np.array([clone(s5.lr()).fit(F[tr], y[tr])[-1].coef_[0] for r, f, tr, te in sp])
    coef = pd.DataFrame({'position': np.arange(1, F.shape[1] + 1), 'coef_mean': coefs.mean(0), 'coef_sd': coefs.std(0),
                         'share_positive': (coefs > 0).mean(0)})
    coef.to_csv(os.path.join(OUT_DIR, 'lsort_coefficients.csv'), index=False, encoding='utf-8-sig')
    uv = pd.DataFrame({'position': np.arange(1, F.shape[1] + 1), 'auc': [roc_auc_score(y, F[:, k]) for k in range(F.shape[1])],
                       'mean_low': F[y == 0].mean(0), 'mean_high': F[y == 1].mean(0)})
    uv.to_csv(os.path.join(OUT_DIR, 'quantile_univariate_auc.csv'), index=False, encoding='utf-8-sig')
    plot_profiles(Xs, y, uv, coef)

    # ---- 段階25のシャッフルの平均
    sh = pd.read_csv(os.path.join(S25, 'shuffle_scores.csv'), encoding='utf-8-sig')
    shuffle_means = sh[sh.kernel_seed == 0].groupby('variant')[['auc', 'bacc', 'W', 'B']].mean()
    plot_overview(summary, shuffle_means)

    # ---- ログ
    log('\n[まとめ]')
    log(summary.round(3).to_string(index=False))
    r = summary[(summary.config == 'S_sort') & (summary.metric == 'all_auc')].iloc[0]
    log(f'\n[主の判定] S_sort の全体 AUC {r["mean"]:.3f}、帰無分布の95%点 {r.null_q95:.3f}、p={r.p:.3f} → '
        + ('上回る: 変化量の分布の情報だけで、多収・低収を偶然以上に分類できた' if r['mean'] > r.null_q95
           else '上回らない: 分布の情報だけでは、偶然以上に分類できるとは確かめられなかった')
        + ('' if len(seeds) == 10 and P == 1000 else '(試運転)'))
    log('\n[D − S_sort]')
    log(pdiff.round(4).to_string(index=False))
    log('\n[段階25のシャッフルの平均(k = 0)]')
    log(shuffle_means.round(3).to_string())
    log('\n[L_sort の係数(位置ごと、100 fold)]')
    log(coef.round(3).to_string(index=False))
    log('\n[位置ごとの単変量 AUC]')
    log(uv.round(4).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
