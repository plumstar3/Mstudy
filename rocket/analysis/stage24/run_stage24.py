"""
段階24: スコアの中身の確認(地域ごとの当たり方、B の内訳、ラベル間のスコアの相関、EXP-033)。

保存済みの OOF スコア(段階12・13)だけを使う記述用の解析。学習し直さない。主の設定は変えない。
  解析A  median の M の集団ごとの地域内 W。件数で重み付けした W と集団を同じ重みにした W、
         集団の大きさとの相関、集団ごとのばらつきが偶然を超えるか(スコア固定・集団の中でラベルを並べ替え、
         1000回、置換 i の乱数は [3, i])
  解析B  median の M の地域間 B を、多収側・低収側の集団の W(> 0.5 / ≦ 0.5)で4種類に分ける。
         集団ごとのスコアのずれと信号の強さ
  解析C  ラベル間(median / label_year / label_year_median)の件ごとの平均 OOF スコアのスピアマン相関
         (全体と集団の中)

ペアと AUC の数え方は段階12と同じ: 同じ (乱数, 繰り返し, fold) のテスト件の多収 × 低収の組、同点 0.5。
種類ごとの AUC は繰り返しごとに fold をまたいで合算し、20回で平均、乱数10通りで平均。
全体の AUC は fold ごとの AUC の平均(乱数ごとに平均して、乱数で平均)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage24\\run_stage24.py
  試運転: ... run_stage24.py --n-perm 20
出力: rocket/outputs/stage24/
"""
import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
from src.evaluation import permute_within  # noqa: E402

OUT = os.path.join(ROCKET_DIR, 'outputs')
OUT_DIR = os.path.join(OUT, 'stage24')
TOL = 1e-9

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = '#2b2a28', '#8a8884'
C_HI, C_LO, C_ACC = '#2a78d6', '#eb6834', '#3a9a5b'


# ---------------------------------------------------------------- 読み込み
def load_scores():
    s13 = pd.read_csv(os.path.join(OUT, 'stage13', 'oof_scores.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    s12 = pd.read_csv(os.path.join(OUT, 'stage12', 'oof_scores.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    s12.insert(0, 'label', 'year')
    d = pd.concat([s13, s12], ignore_index=True)
    return {k: g[g.config == 'M'].reset_index(drop=True) for k, g in d.groupby('label')}


def to_arrays(df):
    """件の表(items)と、スコア S (乱数, 繰り返し, 件)・fold F (乱数, 繰り返し, 件)。"""
    items = df.drop_duplicates(['field_id', 'year'])[['field_id', 'year', 'muniCd', 'muniyear', 'y']] \
        .sort_values(['muniyear', 'field_id']).reset_index(drop=True)
    idx = pd.Series(range(len(items)), index=pd.MultiIndex.from_frame(items[['field_id', 'year']]))
    seeds, reps = sorted(df.seed.unique()), sorted(df.repeat.unique())
    S = np.full((len(seeds), len(reps), len(items)), np.nan)
    F = np.full(S.shape, -1, int)
    ii = idx.loc[pd.MultiIndex.from_frame(df[['field_id', 'year']])].values
    si = df.seed.map({s: k for k, s in enumerate(seeds)}).values
    ri = df.repeat.map({r: k for k, r in enumerate(reps)}).values
    S[si, ri, ii] = df.score.values
    F[si, ri, ii] = df.fold.values
    assert np.isfinite(S).all() and (F >= 0).all(), 'どの (乱数, 繰り返し) でも全件がちょうど1回テストされていない'
    assert len(df) == S.size, '同じ件が同じ (乱数, 繰り返し) で2回以上テストされている'
    return items, S, F


# ---------------------------------------------------------------- ペアの集計
def pair_auc(items, S, F, cat=None):
    """種類ごとの AUC(段階12の数え方)と全体の fold の AUC。cat: (n, n) の種類ラベル(B の内訳用)。
    戻り値: dict 種類 → (AUC, 1回の繰り返しあたりのペア数)。"""
    y = items.y.values
    my = items.muniyear.to_numpy(dtype=str)
    yr = items.year.values
    pn = (y[:, None] == 1) & (y[None, :] == 0)
    same_my = my[:, None] == my[None, :]
    same_yr = yr[:, None] == yr[None, :]
    types = {'W': same_my, 'B': same_yr & ~same_my, 'X': ~same_yr}
    if cat is not None:
        for k in np.unique(cat[cat != '']):
            types[k] = types['B'] & (cat == k)
    ns, nr, n = S.shape
    conc = {t: np.zeros((ns, nr)) for t in types}
    npair = {t: np.zeros((ns, nr)) for t in types}
    fold_auc = [[] for _ in range(ns)]
    for s in range(ns):
        for r in range(nr):
            sc, fo = S[s, r], F[s, r]
            same_f = fo[:, None] == fo[None, :]
            cmp = (sc[:, None] > sc[None, :]) + 0.5 * (sc[:, None] == sc[None, :])
            base = pn & same_f
            for t, T in types.items():
                m = base & T
                conc[t][s, r] = cmp[m].sum()
                npair[t][s, r] = m.sum()
            for f in np.unique(fo):
                m = base & (fo[:, None] == f)
                fold_auc[s].append(cmp[m].mean() if m.any() else np.nan)
    out = {}
    for t in types:
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.where(npair[t] > 0, conc[t] / npair[t], np.nan)
        out[t] = (np.nanmean(np.nanmean(a, axis=1)), npair[t].mean(axis=1).mean(), npair[t])
    out['all'] = (np.mean([np.nanmean(v) for v in fold_auc]), np.nan, None)
    return out


def check_against_saved(data, log):
    s13 = pd.read_csv(os.path.join(OUT, 'stage13', 'summary.csv'), encoding='utf-8-sig')
    s12 = pd.read_csv(os.path.join(OUT, 'stage12', 'summary.csv'), encoding='utf-8-sig')
    s11 = pd.read_csv(os.path.join(OUT, 'stage11', 'summary.csv'), encoding='utf-8-sig')
    ok_all, rows = True, []
    for key, (items, S, F) in data.items():
        res = pair_auc(items, S, F)
        for t in ['all', 'W', 'B', 'X']:
            if key == 'year':
                ref = s11[(s11.config == 'M') & (s11.dataset == 'year_main')].auc_mean.iloc[0] if t == 'all' else \
                    s12[(s12.config == 'M') & (s12.type == t)].auc_mean.iloc[0]
            else:
                met = 'all_auc' if t == 'all' else t
                ref = s13[(s13.label == key) & (s13.config == 'M') & (s13.metric == met)]['mean'].iloc[0]
            ok = abs(res[t][0] - ref) < TOL
            ok_all &= ok
            rows.append(f'{key}:{t} {res[t][0]:.6f}/{ref:.6f}{"" if ok else " ×"}')
    log('確認1 スコアから計算した全体・W・B・X の AUC = 段階11〜13の summary: ' + str(ok_all))
    log('  ' + '  '.join(rows))
    assert ok_all, '保存済みの値と一致しない'


# ---------------------------------------------------------------- 解析A
def group_concordance(items, S):
    """集団ごとに、件 a・b の一致の平均(全乱数・全繰り返し)。集団の件は同じ fold に入るので常に比べられる。"""
    out = {}
    my = items.muniyear.to_numpy(dtype=str)
    for g in pd.unique(my):
        idx = np.flatnonzero(my == g)
        s = S[:, :, idx]
        out[g] = (idx, ((s[:, :, :, None] > s[:, :, None, :]) + 0.5 * (s[:, :, :, None] == s[:, :, None, :]))
                  .mean(axis=(0, 1)))
    return out


def group_w(gc, y):
    """集団ごとの W と ペア数(両クラスがそろう集団だけ)。"""
    w, npair, names = [], [], []
    for g, (idx, P) in gc.items():
        yy = y[idx]
        hi, lo = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
        if len(hi) and len(lo):
            w.append(P[np.ix_(hi, lo)].mean())
            npair.append(len(hi) * len(lo))
            names.append(g)
    return np.array(names), np.array(w), np.array(npair, float)


def wvar(w, n):
    m = np.sum(n * w) / n.sum()
    return np.sum(n * (w - m) ** 2) / n.sum()


def analysis_a(items, S, P, w_ref, log):
    y = items.y.values
    gc = group_concordance(items, S)
    names, w, npair = group_w(gc, y)
    w_weighted = np.sum(npair * w) / npair.sum()
    ok = abs(w_weighted - w_ref) < TOL
    log(f'確認2 集団ごとの W をペア数で重み付けした平均 {w_weighted:.6f} = 段階13の W {w_ref:.6f}: {ok}')
    assert ok
    stat = wvar(w, npair)
    groups = items.muniyear.to_numpy(dtype=str)
    null = np.empty(P)
    for i in range(P):
        yp = permute_within(y, groups, np.random.default_rng([3, i]))
        _, wp, npp = group_w(gc, yp)
        null[i] = wvar(wp, npp)
    p = (1 + np.sum(null >= stat)) / (1 + P)
    rho, prho = spearmanr(npair, w)
    gi = items.groupby('muniyear').agg(n=('y', 'size'), n_high=('y', 'sum'), muniCd=('muniCd', 'first'))
    gi['year'] = [int(g.split('_')[1]) for g in gi.index]
    tab = pd.DataFrame({'muniyear': names, 'w': w, 'n_pairs': npair.astype(int)}).join(gi, on='muniyear')
    tab['n_low'] = tab.n - tab.n_high
    tab = tab[['muniyear', 'muniCd', 'year', 'n', 'n_high', 'n_low', 'n_pairs', 'w']].sort_values(
        ['n_pairs', 'w'], ascending=[False, False])
    one = gi[(gi.n_high == 0) | (gi.n_high == gi.n)]
    summ = pd.DataFrame([{
        'n_groups_with_w': len(w), 'n_groups_one_class': len(one), 'n_pairs_total': int(npair.sum()),
        'w_weighted': w_weighted, 'w_equal': w.mean(), 'w_median_group': np.median(w),
        'n_groups_w_gt_05': int((w > 0.5).sum()), 'n_groups_w_eq_05': int((w == 0.5).sum()),
        'n_groups_w_lt_05': int((w < 0.5).sum()),
        'spearman_npairs_w': rho, 'spearman_p': prho,
        'var_weighted_obs': stat, 'var_null_mean': null.mean(), 'var_null_q95': np.quantile(null, .95),
        'var_p': p, 'n_perm': P}])
    return tab, summ, null, dict(zip(names, w))


# ---------------------------------------------------------------- 解析B
def analysis_b(items, S, F, gw, log):
    my = items.muniyear.to_numpy(dtype=str)
    cls = np.array(['なし' if g not in gw else '高' if gw[g] > 0.5 else '低' for g in my], dtype=object)
    cat = (cls[:, None] + '×' + cls[None, :]).astype(str)  # 行 = 多収側の件の集団、列 = 低収側
    res = pair_auc(items, S, F, cat=cat)
    cats = [k for k in res if '×' in k]
    nb = res['B'][2]
    nsum = sum(res[k][2] for k in cats)
    ok = np.array_equal(nb, nsum)
    log(f'確認3 B の内訳のペア数の合計 = B のペア数(全乱数・全繰り返し): {ok}')
    assert ok
    order = ['高×高', '高×低', '低×高', '低×低']
    order += [k for k in cats if k not in order]
    rows = [{'type': 'B 全体', 'auc': res['B'][0], 'n_pairs_per_repeat': res['B'][1]}]
    for k in order:
        if k in res and res[k][1] > 0:
            rows.append({'type': k, 'auc': res[k][0], 'n_pairs_per_repeat': res[k][1],
                         'share': res[k][1] / res['B'][1]})
    bt = pd.DataFrame(rows)

    # 集団ごとのずれと信号の強さ(件ごとの平均 OOF スコア)
    m = S.mean(axis=(0, 1))
    d = items.assign(score=m)
    rows = []
    for g, x in d.groupby('muniyear'):
        hi, lo = x[x.y == 1].score, x[x.y == 0].score
        rows.append({'muniyear': g, 'n': len(x), 'n_high': len(hi), 'offset': x.score.mean(),
                     'signal': hi.mean() - lo.mean() if len(hi) and len(lo) else np.nan,
                     'within_var': x.score.var(ddof=0), 'w': gw.get(g, np.nan)})
    go = pd.DataFrame(rows)
    between = np.average((go.offset - np.average(go.offset, weights=go.n)) ** 2, weights=go.n)
    within = np.average(go.within_var, weights=go.n)
    # fold ごとのスコアでも同じ比(fold のテスト件の中で)
    ratios = []
    for s in range(S.shape[0]):
        for r in range(S.shape[1]):
            for f in np.unique(F[s, r]):
                k = F[s, r] == f
                t = pd.DataFrame({'g': my[k], 's': S[s, r, k]})
                gm = t.groupby('g').s.transform('mean')
                bv, wv = ((gm - t.s.mean()) ** 2).mean(), ((t.s - gm) ** 2).mean()
                if wv > 0:
                    ratios.append(bv / wv)
    var_row = {'between_var': between, 'within_var': within, 'ratio_between_within': between / within,
               'ratio_fold_median': np.median(ratios), 'ratio_fold_q25': np.quantile(ratios, .25),
               'ratio_fold_q75': np.quantile(ratios, .75), 'n_folds': len(ratios),
               'spearman_signal_w': spearmanr(go.signal, go.w, nan_policy='omit')[0]}
    return bt, go, var_row


# ---------------------------------------------------------------- 解析C
def analysis_c(data, log):
    mean_sc = {}
    for key, (items, S, F) in data.items():
        mean_sc[key] = items.assign(score=S.mean(axis=(0, 1)))
    combos = [('median', 'year'), ('median', 'year_median'), ('year', 'year_median')]
    rows, scat = [], {}
    for a, b in combos:
        j = mean_sc[a].merge(mean_sc[b][['field_id', 'year', 'score', 'muniyear', 'y']], on=['field_id', 'year'],
                             suffixes=('_a', '_b'))
        assert (j.muniyear_a == j.muniyear_b).all()
        rho = spearmanr(j.score_a, j.score_b)[0]
        rs, ns = [], []
        for g, x in j.groupby('muniyear_a'):
            if len(x) >= 3:
                r = spearmanr(x.score_a, x.score_b)[0]
                if np.isfinite(r):
                    rs.append(r)
                    ns.append(len(x))
        rows.append({'model_a': a, 'model_b': b, 'n_common': len(j), 'n_a': len(mean_sc[a]), 'n_b': len(mean_sc[b]),
                     'spearman_all': rho, 'spearman_within_weighted': np.average(rs, weights=ns),
                     'spearman_within_equal': np.mean(rs), 'n_groups_within': len(rs), 'n_cases_within': int(sum(ns)),
                     'same_label_rate': (j.y_a == j.y_b).mean()})
        log(f'  {a} × {b}: 共通 {len(j)} 件({a} {len(mean_sc[a])}・{b} {len(mean_sc[b])})、'
            f'集団の中の相関に使った集団 {len(rs)}({int(sum(ns))} 件)')
        scat[(a, b)] = j
    return pd.DataFrame(rows), scat


# ---------------------------------------------------------------- 図
def plot(ga, bt, scat, summ_a):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    ax = axs[0]
    rng = np.random.default_rng(0)
    xj = ga.n_pairs + rng.uniform(-0.25, 0.25, len(ga))
    ax.scatter(xj, ga.w, s=18 + 4 * ga.n, color=C_HI, alpha=0.6, edgecolor='none')
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.axhline(summ_a.w_weighted.iloc[0], color=INK, lw=1.2, label=f'件数で重み付け {summ_a.w_weighted.iloc[0]:.3f}')
    ax.axhline(summ_a.w_equal.iloc[0], color=C_LO, lw=1.2, ls='--', label=f'集団を同じ重み {summ_a.w_equal.iloc[0]:.3f}')
    ax.set_xlabel('集団の W のペア数')
    ax.set_ylabel('集団ごとの W(全乱数・全繰り返し)')
    ax.set_title('A: 集団ごとの地域内 W(median の M)', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    b = bt[bt.type != 'B 全体']
    cols = [C_ACC if t == '高×高' else C_HI if t in ('高×低', '低×高') else C_LO if t == '低×低' else MUTED
            for t in b.type]
    ax.bar(range(len(b)), b.auc, color=cols, alpha=0.75)
    for k, (_, r) in enumerate(b.iterrows()):
        ax.text(k, r.auc + 0.01, f'{r.auc:.3f}\n({r.n_pairs_per_repeat:.0f}組)', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.axhline(bt.auc.iloc[0], color=INK, lw=1.2, label=f'B 全体 {bt.auc.iloc[0]:.3f}')
    ax.set_xticks(range(len(b)), list(b.type), fontsize=9)
    ax.set_xlabel('多収側の集団 × 低収側の集団(なし = 片方のクラスだけの集団)', fontsize=9)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel('AUC')
    ax.set_title('B: 地域間 B の内訳(集団の W 高 > 0.5 / 低 ≦ 0.5)', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[2]
    j = scat[('median', 'year')]
    ax.scatter(j.score_a, j.score_b, s=16, c=np.where(j.y_a == 1, C_HI, C_LO), alpha=0.7, edgecolor='none')
    ax.set_xlabel('median の M(件ごとの平均 OOF スコア)')
    ax.set_ylabel('label_year の M(件ごとの平均 OOF スコア)')
    ax.set_title(f'C: スコアの散布図(共通 {len(j)} 件、色 = median のラベル)', loc='left', fontsize=11, color=INK)
    for a in axs:
        for sp in ['top', 'right']:
            a.spines[sp].set_visible(False)
    fig.suptitle('段階24: スコアの中身の確認(保存済みのスコアだけを使う記述用の解析)', fontsize=13, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'stage24_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    args = ap.parse_args()
    P = args.n_perm
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}')
    raw = load_scores()
    data = {k: to_arrays(raw[k]) for k in ['median', 'min2', 'year_median', 'year']}
    for k, (items, S, F) in data.items():
        log(f'  {k}: {len(items)} 件、乱数 {S.shape[0]}、繰り返し {S.shape[1]}、市町村×年 {items.muniyear.nunique()}')
    check_against_saved(data, log)

    s13 = pd.read_csv(os.path.join(OUT, 'stage13', 'summary.csv'), encoding='utf-8-sig')
    w_ref = s13[(s13.label == 'median') & (s13.config == 'M') & (s13.metric == 'W')]['mean'].iloc[0]
    items, S, F = data['median']

    log('\n[解析A] 集団ごとの地域内 W(median の M)')
    ga, summ_a, null_a, gw = analysis_a(items, S, P, w_ref, log)
    ga.to_csv(os.path.join(OUT_DIR, 'group_w.csv'), index=False, encoding='utf-8-sig')
    np.save(os.path.join(OUT_DIR, 'null_group_w_var.npy'), null_a)

    log('\n[解析B] 地域間 B の内訳(median の M)')
    bt, go, var_row = analysis_b(items, S, F, gw, log)
    bt.to_csv(os.path.join(OUT_DIR, 'b_breakdown.csv'), index=False, encoding='utf-8-sig')
    go.to_csv(os.path.join(OUT_DIR, 'group_offsets.csv'), index=False, encoding='utf-8-sig')
    summ_a = summ_a.assign(**var_row)
    summ_a.to_csv(os.path.join(OUT_DIR, 'group_w_summary.csv'), index=False, encoding='utf-8-sig')

    log('\n[解析C] ラベル間のスコアの相関(件ごとの平均 OOF スコア)')
    sc, scat = analysis_c(data, log)
    sc.to_csv(os.path.join(OUT_DIR, 'score_correlation.csv'), index=False, encoding='utf-8-sig')

    plot(ga, bt, scat, summ_a)

    log('\n[解析A の結果]')
    log(summ_a.T.round(4).to_string(header=False))
    log('\n[集団ごとの W(ペア数の多い順)]')
    log(ga.round(3).to_string(index=False))
    log('\n[解析B の結果]')
    log(bt.round(3).to_string(index=False))
    log('\n[集団ごとのずれと信号の強さ]')
    log(go.sort_values('w', ascending=False).round(3).to_string(index=False))
    log('\n[解析C の結果]')
    log(sc.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
