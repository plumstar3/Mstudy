"""
段階15: 品種が集団の中で同じ集団だけでの評価(EXP-024)。

段階14(EXP-023)の B3(品種 + 播種日)は label_muniyear_median で全体 AUC 0.575(p=0.047)と偶然水準ではなく、
M(0.607)との差も検出できなかった。品種が集団(市町村×年)の中で1つだけで欠測の無い集団に絞り、学習から評価までをやり直す。
この部分集合では品種は集団の中で同じなので、地域内のラベルの手がかりになりえない。

データ: data/processed/stage13/label_muniyear_median_*(123件)から部分集合を取る。
設定: M(1次差分 → MiniRocket 10000、乱数0〜9 → StandardScaler(with_mean=False) → RidgeClassifierCV)、
      B1(VWC の要約統計 + LR)、B3(品種の one-hot + 播種日の通日 + LR、段階14と同じ)。
分割: 部分集合で make_splits(y, muniCd, 5, 20, 0)。置換: 市町村×年の中で1000回(置換 i の乱数は [0, i])。
判定(事前に固定、1つだけ): 部分集合で学習し直した M の全体の AUC が、帰無分布の95%点を上回るか。
参考(判定に使わない): 段階13の median の oof スコア(全123件で学習)を、部分集合の件どうしだけで集計した値。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage15\\run_stage15.py
  試運転: ... run_stage15.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage15/
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13', 'stage14']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage11 as s11  # noqa: E402
import run_stage12 as s12  # noqa: E402
import run_stage13 as s13  # noqa: E402
import run_stage14 as s14  # noqa: E402
from run_stage6 import BASE_ALPHAS, s5  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY = 'median'
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage15')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
CONFIGS = ['M', 'B1', 'B3']
TYPES = ['W', 'B', 'X']
POWER_Z = s13.POWER_Z
# 事前の集計(段階14の報告の「20集団・80件前後」を、欠測を含む集団を除いて数え直したもの)
EXPECT = {'n_groups': 19, 'n': 76, 'n_high': 40, 'n_low': 36, 'n_muni': 16}
TOL = 0.15  # 件数がこれ以上想定から外れたら止める

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED
CFG_COLOR = {'M': '#2a78d6', 'B1': '#eb6834', 'B3': '#9a9890'}
CFG_LABEL = {'M': 'M MiniRocket・1次差分', 'B1': 'B1 VWC の要約統計', 'B3': 'B3 品種 + 播種日'}


# ---------------------------------------------------------------- データ
def classify_groups(meta):
    """市町村×年の集団ごとの件数・クラス構成・品種と、部分集合に含めるか。"""
    rows = []
    for g, d in meta.groupby('muniyear', sort=True):
        n, hi = len(d), int(d.y.sum())
        n_na = int(d.breed.isna().sum())
        br = sorted(d.breed.dropna().astype(str).unique())
        if n_na > 0:
            inc, why = False, f'品種の欠測を含む({n_na}件)'
        elif len(br) >= 2:
            inc, why = False, f'品種が{len(br)}つ'
        else:
            inc, why = True, '品種が1つ・欠測なし'
        rows.append({'muniyear': g, 'muniCd': d.muniCd.iloc[0], 'year': int(d.year.iloc[0]), 'n': n, 'n_high': hi,
                     'n_low': n - hi, 'balance': '同数' if hi == n - hi else ('片方だけ' if hi in (0, n) else '数が違う'),
                     'breeds': '/'.join(br + (['(欠測)'] if n_na else [])), 'n_breed_missing': n_na,
                     'included': inc, 'reason': why})
    return pd.DataFrame(rows)


def build_subset(log):
    X, meta = s14.load()
    groups = classify_groups(meta)
    groups.to_csv(os.path.join(OUT_DIR, 'subset_groups.csv'), index=False, encoding='utf-8-sig')
    keep = meta.muniyear.isin(groups[groups.included].muniyear).values
    Xs, ms = X[keep], meta[keep].reset_index(drop=True)
    # 確認1: 定義どおり(集団の中で品種が1つ・欠測なし)
    chk = ms.groupby('muniyear').breed.agg(lambda s: s.notna().all() and s.nunique() == 1).all()
    rest = meta[~keep].groupby('muniyear').breed.agg(lambda s: s.isna().any() or s.nunique() >= 2).all()
    log(f'確認1 部分集合の件はすべて「集団の中で品種が1つ・欠測なし」{chk}、除いた集団はすべてそれに当たらない {rest}')
    assert chk and rest
    ex = groups[~groups.included]
    log(f'\n[部分集合] 元 {len(meta)} 件・{len(groups)} 集団 → {len(ms)} 件・{groups.included.sum()} 集団')
    log(f'  除いた集団 {len(ex)}: 品種の欠測を含む {ex.reason.str.startswith("品種の欠測").sum()} 集団、'
        f'品種が2つ以上 {ex.reason.str.startswith("品種が").sum()} 集団')
    for _, r in ex.iterrows():
        log(f'    {r.muniyear}: {r.n}件(多{r.n_high}/低{r.n_low})、{r.breeds} → {r.reason}')
    inc = groups[groups.included]
    got = {'n_groups': len(inc), 'n': len(ms), 'n_high': int(ms.y.sum()), 'n_low': int((1 - ms.y).sum()),
           'n_muni': ms.muniCd.nunique()}
    log(f'  {got["n"]} 件(多収 {got["n_high"]} / 低収 {got["n_low"]})、集団 {got["n_groups"]}、市町村 {got["n_muni"]}、年別 '
        + ', '.join(f'{yr}年 {n}件' for yr, n in ms.year.value_counts().sort_index().items()))
    for b, s in inc.groupby('balance'):
        log(f'  クラス構成 {b}: {len(s)} 集団・{int(s.n.sum())} 件'
            + ('' if b == '同数' else '(' + '、'.join(f'{r.muniyear} 多{r.n_high}/低{r.n_low}' for _, r in s.iterrows()) + ')'))
    log('  品種の内訳(集団数・件数): ' + '、'.join(f'{b} {len(s)}集団・{int(s.n.sum())}件'
                                         for b, s in inc.groupby('breeds')))
    if got['n_muni'] < N_SPLITS:
        raise SystemExit(f'市町村が {got["n_muni"]} で {N_SPLITS} 分割できない。止めて報告する')
    if abs(got['n'] - EXPECT['n']) > TOL * EXPECT['n'] or got != EXPECT:
        raise SystemExit(f'件数などが想定 {EXPECT} と違う: {got}。止めて報告する')
    return Xs, ms, groups


# ---------------------------------------------------------------- 1つの設定
def run_config(config, X, Xd, meta, sp, ps, seeds, P, feats_b1, log):
    y = meta.y.values
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': []}
    nulls = {'all': [], 'W': [], 'B': []}
    t0 = time.time()
    for seed in (seeds if config == 'M' else [None]):
        n_redo = 0
        if config == 'M':
            feats = s6.fold_features('minirocket', seed, Xd, sp)
            ref = s6.eval_cached(feats, y, sp)
            decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
            del feats
        elif config == 'B1':
            decs = s12.lr_decisions(feats_b1, y, sp, ps[2])
            ref = evaluate(s5.lr(), feats_b1, y, sp)
        else:
            F = s14.b3_features(meta)
            Y = np.vstack([y, ps[2]])
            decs = Parallel(n_jobs=N_JOBS)(delayed(s14._b3_fold)(F, tr, te, Y) for r, f, tr, te in sp)
            ref = evaluate(s14.b3_pipeline(), F, y, sp)
        if config != 'M':
            ref = ref.assign(n_train=[len(tr) for _, _, tr, _ in sp], alpha_over_n=np.nan)
        obs, null = s13.analyse(decs, meta, sp, ps)
        s13.check_obs(f'{config} 乱数{seed}', obs, ref, log)  # 確認5
        out['folds'].append(ref.assign(config=config, seed=seed))
        for k in nulls:
            nulls[k].append(null[k])
        for (r, f, tr, te), dec in zip(sp, decs):
            m = meta.iloc[te]
            out['scores'].append(pd.DataFrame({'config': config, 'seed': seed, 'repeat': r, 'fold': f,
                                               'field_id': m.field_id.values, 'year': m.year.values,
                                               'muniCd': m.muniCd.values, 'muniyear': m.muniyear.values,
                                               'y': m.y.values, 'score': dec[0]}))
        row = {'config': config, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
               'alpha_at_max': np.isclose(ref.alpha_over_n, BASE_ALPHAS[-1]).mean() if config == 'M' else np.nan,
               'n_redo': n_redo}
        for t in TYPES:
            a, n = obs[t]
            row[t] = np.nanmean(a)
            out['pair'].append(pd.DataFrame({'config': config, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
        out['seed'].append(row)
        if config == 'M':
            log(f'    M 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
                + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES) + f'(学習し直した組 {n_redo})')
    per_seed = pd.DataFrame(out['seed'])
    pa = pd.concat(out['pair'])
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    np.save(os.path.join(OUT_DIR, f'null_{config}_all.npy'), null_all)
    base = {'config': config, 'n': len(y), 'n_high': int(y.sum()), 'n_low': int((1 - y).sum()),
            'n_seeds': len(per_seed), 'n_perm': P, 'alpha_at_max': per_seed.alpha_at_max.mean()}
    summ = []
    for met, j in [('all_auc', 1), ('all_bacc', 0)]:
        v, nl = per_seed[met], null_all[:, j]
        summ.append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                     'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                     'p': (1 + np.sum(nl >= v.mean())) / (1 + P), 'auc_needed_80pct': 0.5 + POWER_Z * np.nanstd(nl)})
    for t in TYPES:
        v = per_seed[t]
        r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
             'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean()}
        if t in ('W', 'B'):
            nl = np.mean(np.stack(nulls[t]), axis=0)
            np.save(os.path.join(OUT_DIR, f'null_{config}_{t}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                      'p': (1 + np.sum(nl >= v.mean())) / (1 + P), 'auc_needed_80pct': 0.5 + POWER_Z * np.nanstd(nl)})
        summ.append(r)
    s = pd.DataFrame(summ)
    log(f'  {config:2s} ' + '  '.join(
        f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95')) else '')
        for _, r in s.iterrows()) + f'  [{time.time() - t0:.0f}s]')
    out['summary'] = summ
    return out


# ---------------------------------------------------------------- 参考: 全123件で学習したモデル
def reference_stage13(meta_sub, log):
    """段階13の median・M の oof スコアを、部分集合の件どうしのペアだけで集計する(判定には使わない)。"""
    X, meta = s14.load()
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    S, seeds, _ = s14.load_stage13(meta, sp, log)
    sub = meta.muniyear.isin(meta_sub.muniyear.unique()).values
    assert sub.sum() == len(meta_sub)
    counts, fold_auc = [], []
    for r, f, tr, te in sp:
        st = sub[te]
        m = s12.pair_masks(meta, te)
        bb = st[:, None] & st[None, :]
        mm = {t: m[t] & bb for t in TYPES}
        mm['ALL'] = bb
        counts.append(s12.pair_counts(S[:, r, te], np.tile(y[te], (len(seeds), 1)), mm))
        ts = te[st]
        fold_auc.append([roc_auc_score(y[ts], S[k, r, ts]) if len(np.unique(y[ts])) == 2 else np.nan
                         for k in range(len(seeds))])
    fold_auc = np.array(fold_auc)  # (fold, 乱数)
    rows = [{'metric': 'fold_auc(部分集合のテスト件だけの fold の AUC)',
             'mean': np.nanmean(fold_auc, axis=0).mean(), 'seed_min': np.nanmean(fold_auc, axis=0).min(),
             'seed_max': np.nanmean(fold_auc, axis=0).max(), 'n_folds_valid': int(np.isfinite(fold_auc[:, 0]).sum())}]
    for t in ['ALL'] + TYPES:
        a, n = s12.repeat_auc(counts, sp, t)  # (繰り返し, 乱数)
        ps = np.nanmean(a, axis=0)
        rows.append({'metric': f'{t}(ペアを繰り返しごとに合算)', 'mean': ps.mean(), 'seed_min': ps.min(), 'seed_max': ps.max(),
                     'n_pairs_per_repeat': n[:, 0].mean()})
    t = pd.DataFrame(rows).assign(source='段階13 median・M(全123件で学習)', n_subset=int(sub.sum()))
    t.to_csv(os.path.join(OUT_DIR, 'reference_stage13_scores.csv'), index=False, encoding='utf-8-sig')
    log('\n[参考 段階13の全件モデルのスコアを部分集合のペアだけで集計(判定に使わない)]')
    for _, r in t.iterrows():
        log(f'  {r.metric}: {r["mean"]:.3f}(乱数 {r.seed_min:.3f}〜{r.seed_max:.3f})'
            + (f'、ペア/繰り返し {r.n_pairs_per_repeat:.0f}' if pd.notna(r.get('n_pairs_per_repeat')) else ''))
    return t


# ---------------------------------------------------------------- 図
def plot(summary, ref):
    fig, ax = plt.subplots(figsize=(10, 5))
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内'), ('B', 'B 地域間'), ('X', 'X 年をまたぐ(参考)')]
    sm = summary.set_index(['config', 'metric'])
    w = 0.26
    for k, c in enumerate(CONFIGS):
        xs = np.arange(len(mets)) + (k - 1) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.92, color=CFG_COLOR[c], alpha=0.85, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95'] if 'null_q95' in sm.columns else np.nan
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    rmap = {'all_auc': 'fold_auc', 'W': 'W(', 'B': 'B(', 'X': 'X('}
    for i, (m, _) in enumerate(mets):
        v = ref[ref.metric.str.startswith(rmap[m])]['mean'].iloc[0]
        ax.plot(i - 1.62 * w, v, marker='D', ms=6, mfc='white', mec=INK, mew=1.4, ls='', zorder=4,
                label='参考: 段階13の全件モデル(M)' if i == 0 else None)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    lo = min(0.35, summary['mean'].min() - 0.03)
    ax.set_ylim(lo, max(0.75, summary['mean'].max() + 0.05))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=2)
    n = int(summary.n.iloc[0])
    ax.set_title(f'段階15: 品種が集団の中で同じ集団だけ({n}件)で学習し直した結果', fontsize=12, color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stage15_overview.png'), dpi=150)
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
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {N_JOBS}')
    X, meta, groups = build_subset(log)
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    st = split_table(sp, y, meta.muniCd.values)
    st.to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    nte = np.array([len(te) for *_, te in sp])
    log(f'  分割: テスト件 {nte.min()}〜{nte.max()} 件、テスト側が1クラスだけの fold '
        f'{sum(len(np.unique(y[te])) < 2 for *_, te in sp)}')

    log('\n[起動時の確認]')
    s13.prechecks(KEY, Xd, meta, sp, log)  # 確認2・4・3
    ps = s13.perm_sets(KEY, meta, P)
    feats_b1 = s11.baseline_features(X, meta)['B1']

    res = {}
    for c in CONFIGS:
        log(f'\n[{c}]')
        res[c] = run_config(c, X, Xd, meta, sp, ps, seeds, P, feats_b1, log)

    summary = pd.DataFrame([r for c in res for r in res[c]['summary']])
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat([f for c in res for f in res[c]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for c in res for r in res[c]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([s for c in res for s in res[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([p for c in res for p in res[c]['pair']]).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'),
                                                               index=False, encoding='utf-8-sig')

    # 対ごとの差(段階14と同じ形式。基準線は乱数によらないので各乱数に同じ値を並べる)
    fM = folds[folds.config == 'M'][['seed', 'repeat', 'fold', 'auc', 'bacc']]
    pdiff = []
    for b in ['B3', 'B1']:
        fb = pd.concat([folds[folds.config == b].assign(seed=s) for s in seeds])
        pdiff.append(s7.paired(fM, fb[['seed', 'repeat', 'fold', 'auc', 'bacc']], f'M − {b}'))
    pds = s7.paired_summary(pd.concat(pdiff))
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差]')
    for _, r in pds.iterrows():
        log(f'  {r.comparison} {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返しの2.5〜97.5%点 [{r.rep_q025:+.3f}, '
            f'{r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}、差を検出 {r.detected}')

    ref = reference_stage13(meta, log)
    plot(summary, ref)

    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'n_pairs_per_repeat', 'auc_needed_80pct', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.config == 'M') & (summary.metric == 'all_auc')].iloc[0]
    log(f'\n[主の判定] 部分集合で学習し直した M の全体の AUC {r["mean"]:.3f}、帰無分布の95%点 {r.null_q95:.3f}、p={r.p:.3f} → '
        + ('上回る: 品種が集団の中で同じ条件でも、土壌水分の時系列は多収・低収を分類できる(M の結果は品種では説明できない)'
           if r['mean'] > r.null_q95 else '上回らない: この件数では、品種の影響を除いた効果を確かめられない'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
