"""
段階18: 段階17(MultiRocket のプーリングの比較、EXP-026)を label_year_median で行う(EXP-027)。

label_year_median は段階13から「参考」のラベル(年の中の中央値分割。地域差と交絡するので結論には使わない)。
今回も採用の判定はせず、比べるだけにする。

  MR4_diff  1次差分(59点)に MultiRocket(num_kernels=2520)、入力の系列 × 4種類だけ(10,080特徴量)
  MR_std    生の値(60点)に MultiRocket の既定値(元の系列 + 1次差分 × 4種類、49,728特徴量)。置換200回
  参考2     MR4_diff の特徴量の部分集合(PPV のみ / +MPV / +MIPV / +LSPV / 4種類)。置換なし
  M_diff    比べる相手。1次差分 → MiniRocket の PPV(段階13の M と同じ構成)を、238件で計算し直す

データ: data/processed/stage13/label_year_median_*(240件)から、圃場472・473(2018年。VWC が127日すべて同じ値、
docs/data_issues.md の3)を除いた238件。ラベルは付け直さない(どちらも低収)。分割は238件で
make_splits(y, muniCd, 5, 20, 0) を作り直す。段階13の240件とは分割が違い、段階13の M とは fold ごとに比べられない
ので、M_diff もこの238件・同じ分割・同じ置換で計算し直す。
置換(段階13の year_median と同じ): 全体と B は年の中、W は市町村×年の中で並べ替える。年の中の置換では分割が
変わるので、観測ラベルの分割を固定して使う。

部品は段階16・17の run_stage16.py・run_stage17.py から import する。run_m・run_cfg・implementation_check は保存先に
それぞれのモジュールの OUT_DIR を使うので、このスクリプトの中でだけ outputs/stage18/ に置き換える
(段階16・17のファイルは変更しない)。
並列数は os.cpu_count()(run_stage6_local.N_JOBS)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage18\\run_stage18.py
  試運転: ... run_stage18.py --n-seeds 1 --n-perm 20 --n-perm-std 20 --n-boot 50
出力: rocket/outputs/stage18/
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

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage17'))
import run_stage17 as s17  # noqa: E402

s6, s7, s12, s13, s16 = s17.s6, s17.s7, s17.s12, s17.s13, s17.s16
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY, LABEL = 'year_median', 'label_year_median'
DATA_DIR = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage18')
s17.OUT_DIR = OUT_DIR  # 段階16・17の部品の保存先を、このスクリプトの中でだけ置き換える
s16.OUT_DIR = OUT_DIR
DROP = [(472, 2018), (473, 2018)]  # VWC が127日すべて同じ値の2件(docs/data_issues.md の3)
N_SPLITS, N_REPEATS, SEED0 = s17.N_SPLITS, s17.N_REPEATS, s17.SEED0
N_JOBS = s17.N_JOBS
TYPES = s17.TYPES
SUBSETS = s17.SUBSETS
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- データ
def load(log):
    X = np.load(os.path.join(DATA_DIR, f'{LABEL}_X.npy'))
    meta = pd.read_csv(os.path.join(DATA_DIR, f'{LABEL}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    drop = meta[[k in DROP for k in zip(meta.field_id, meta.year)]]
    assert len(drop) == 2, drop
    assert np.array_equal(X[drop.index[0]], X[drop.index[1]]), '472・473 の窓が同じでない'
    keep = ~meta.index.isin(drop.index)
    log('除いた圃場: ' + '、'.join(f'{r.field_id}({r.year}、{r.muniyear}、ラベル {r.y})' for _, r in drop.iterrows())
        + '(窓の60点がすべて同じ)。ラベルは付け直さない')
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    u = np.unique(X.reshape(len(X), -1), axis=0)
    log(f'  残り {len(meta)} 件で、窓の系列が完全に同じ組: {len(meta) - len(u)}')
    assert len(u) == len(meta)
    return X, meta


# ---------------------------------------------------------------- 起動時の確認
def prechecks(Xd, meta, sp, log, n_perm=20):
    y = meta.y.values
    # 2. Pipeline と特徴量の使い回し
    a = evaluate(s17.mr_pipeline('MR4_diff', 0), Xd, y, sp[:10])
    res = Parallel(n_jobs=N_JOBS)(delayed(s17._fold)('MR4_diff', 0, Xd, tr, te, y, np.zeros((0, len(y))), False)
                                  for r, f, tr, te in sp[:10])
    b = pd.DataFrame([x[3] for x in res])
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'  確認2 Pipeline と特徴量の使い回しの一致(MR4_diff、乱数0、先頭10 fold): {ok}')
    assert ok
    # 4. year_median では年の中の置換で分割が変わるので、観測ラベルの分割を固定する(段階13と同じ)
    ps = s13.perm_sets(KEY, meta, n_perm)
    log(f'  確認4 置換の単位: 全体・B は{ps[5]}の中、W は市町村×年の中。分割は観測ラベルのものを固定(段階13の year_median と同じ)')
    # 3. まとめて解いた置換 = sklearn で1回ずつ
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(s17._fold_check)('MR4_diff', 0, Xd, tr, te, y, ps[2]) for r, f, tr, te in sp)
    _, na = s13.analyse([d for d, _, _ in res], meta, sp, ps)
    _, nb = s13.analyse([np.vstack([d[:1], rr]) for d, rr, _ in res], meta, sp, ps)
    ok = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'  確認3 置換{n_perm}回分の帰無分布(全体・W・B): まとめて解いた結果 = sklearn で1回ずつ: {ok}'
        f'(学習し直した組 {sum(k for *_, k in res)}、{time.time() - t:.1f}s)')
    assert ok


# ---------------------------------------------------------------- 図
def plot(summary, abl, n):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [3, 2]})
    ax = axs[0]
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内'), ('B', 'B 地域間')]
    cfgs = ['M_diff', 'MR4_diff', 'MR_std']
    sm = summary.set_index(['config', 'metric'])
    w = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        xs = np.arange(len(mets)) + (k - (len(cfgs) - 1) / 2) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.92, color=s17.CFG_COLOR[c], alpha=0.85, label=s17.CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95']
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点')
    ax.set_xticks(range(len(mets)), [n_ for _, n_ in mets])
    ax.set_ylim(0.4, max(0.75, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('設定ごとの全体・W・B', fontsize=11, color=INK, loc='left')
    s6.style(ax)
    ax = axs[1]
    names = list(SUBSETS)
    rep = abl.groupby(['subset', 'seed', 'repeat']).auc.mean().groupby(['subset', 'repeat']).mean()
    data = [rep.loc[n_].values for n_ in names]
    bp = ax.boxplot(data, positions=range(len(names)), widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.2), boxprops=dict(lw=0), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED), flierprops=dict(marker='o', ms=2.5, mfc=MUTED, mec='none'))
    for b in bp['boxes']:
        b.set_facecolor('#2a78d6')
        b.set_alpha(0.45)
    ax.plot(range(len(names)), [d.mean() for d in data], 'D', color=INK, ms=5)
    m = sm.loc[('M_diff', 'all_auc'), 'mean']
    ax.axhline(m, color=s17.CFG_COLOR['M_diff'], lw=1.5, ls='--', label=f'M_diff {m:.3f}')
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.set_xticks(range(len(names)), names, fontsize=9)
    ax.set_ylabel('全体の AUC(繰り返しごと)', color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title('参考2: MR4_diff のプーリングの内訳(記述のみ)', fontsize=11, color=INK, loc='left')
    s6.style(ax)
    fig.suptitle(f'段階18: MultiRocket のプーリング(label_year_median、{n}件、参考のラベル)', fontsize=13, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'stage18_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-perm-std', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-seeds-std', type=int, default=None, help='MR_std の乱数の数(既定は --n-seeds と同じ)')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    seeds_std = list(range(args.n_seeds_std if args.n_seeds_std is not None else args.n_seeds))
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
    X, meta = load(log)
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    split_table(sp, y, meta.muniCd.values).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    log(f'データ: {LABEL} {len(meta)} 件(多収 {y.sum()} / 低収 {(1 - y).sum()})、地域 {meta.muniCd.nunique()}、'
        f'市町村×年 {meta.muniyear.nunique()}。分割は {len(meta)} 件で作り直した')
    s13s = pd.read_csv(os.path.join(S13_DIR, 'summary.csv'), encoding='utf-8-sig')
    r13 = s13s[(s13s.label == KEY) & (s13s.config == 'M')].set_index('metric')['mean']
    log(f'  参考: 段階13の year_median・M(240件、別の分割): 全体 AUC {r13["all_auc"]:.3f}、W {r13["W"]:.3f}、B {r13["B"]:.3f}')

    log('\n[起動時の確認]')
    s17.implementation_check(Xd, log)  # 確認1(実装の確認。保存先は outputs/stage18/)
    log('  [M_diff(MiniRocket)]')
    s13.prechecks(KEY, Xd, meta, sp, log)  # M_diff の確認2・3(year_median なので確認4は無い)
    log('  [MR4_diff]')
    prechecks(Xd, meta, sp, log)  # 確認2・4・3
    ps = s13.perm_sets(KEY, meta, P)
    ps_std = s13.perm_sets(KEY, meta, args.n_perm_std)

    res = {}
    log('\n[M_diff(1次差分 → MiniRocket、この件数・分割で計算し直す)]')
    res['M_diff'] = s16.run_m('M_diff', Xd, meta, sp, ps, seeds, P, log)  # 確認5 は中で
    for r in res['M_diff']['summary']:
        r['n_features'] = 9996
    log('\n[MR4_diff]')
    res['MR4_diff'] = s17.run_cfg('MR4_diff', Xd, meta, sp, ps, seeds, P, log, with_subsets=True)  # 確認5 は中で
    log('\n[MR_std(参考1)]')
    log('  特徴量のキャッシュは持たず、fold ごとに worker の中で特徴量を作って決定関数だけを返す(メモリのため)')
    res['MR_std'] = s17.run_cfg('MR_std', X.astype(float), meta, sp, ps_std, seeds_std, args.n_perm_std, log)

    summary = pd.DataFrame([r for c in res for r in res[c]['summary']])
    summary['null_group'] = np.where(summary.metric == 'W', 'muniyear',
                                     np.where(summary.metric.isin(['all_auc', 'all_bacc', 'B']), 'year', ''))
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    folds = pd.concat([f for c in res for f in res[c]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for c in res for r in res[c]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([s for c in res for s in res[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([p for c in res for p in res[c]['pair']]).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'),
                                                                        index=False, encoding='utf-8-sig')

    # 対ごとの差(同じ分割・同じ乱数)
    cols = ['seed', 'repeat', 'fold', 'auc', 'bacc']
    fM = folds[folds.config == 'M_diff'][cols]
    g = lambda c: folds[folds.config == c][cols]  # noqa: E731
    pdiff = pd.concat([s7.paired(g('MR4_diff'), fM[fM.seed.isin(seeds)], 'MR4_diff − M_diff'),
                       s7.paired(g('MR_std'), fM[fM.seed.isin(seeds_std)], 'MR_std − M_diff')])
    pdiff.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    pds = s7.paired_summary(pdiff)
    pds['positive_ci'] = pds.rep_q025 > 0
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差(同じ分割・同じ乱数)]')
    for _, r in pds.iterrows():
        log(f'  {r.comparison} {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{r.rep_q025:+.3f}, '
            f'{r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}')

    # W の差のブートストラップ
    boot, pt = s16.w_diff_bootstrap(res['MR4_diff']['S'], res['M_diff']['S'], meta, sp, args.n_boot)
    boot = boot.rename(columns={'W_GC_date': 'W_MR4_diff'})
    boot.to_csv(os.path.join(OUT_DIR, 'w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    bq = boot.W_diff.quantile([.025, .975]).values
    log(f'\n[W の差 MR4_diff − M_diff(市町村のブートストラップ {args.n_boot}回)] 点推定 {pt["a"] - pt["b"]:+.3f}'
        f'(MR4_diff {pt["a"]:.3f}、M_diff {pt["b"]:.3f})、95%区間 [{bq[0]:+.3f}, {bq[1]:+.3f}]、'
        f'正の割合 {(boot.W_diff > 0).mean():.3f}')
    m = (summary.config == 'MR4_diff') & (summary.metric == 'W')
    summary.loc[m, 'w_diff_boot_low'], summary.loc[m, 'w_diff_boot_high'] = bq[0], bq[1]

    # 参考2: プーリングの内訳
    abl = pd.DataFrame(res['MR4_diff']['sub'])
    ab = []
    for sn in SUBSETS:
        d = abl[abl.subset == sn]
        per_seed = d.groupby('seed')[['auc', 'bacc']].mean()
        ab.append({'subset': sn, 'n_features': int(d.n_features.iloc[0]), 'auc': per_seed.auc.mean(),
                   'auc_seed_min': per_seed.auc.min(), 'auc_seed_max': per_seed.auc.max(), 'bacc': per_seed.bacc.mean(),
                   'alpha_at_max': np.isclose(d.alpha_over_n, s17.BASE_ALPHAS[-1]).mean()})
    base_ppv = abl[abl.subset == 'PPV のみ'][cols]
    pab = s7.paired_summary(pd.concat([s7.paired(abl[abl.subset == sn][cols], base_ppv, f'{sn} − PPV のみ')
                                       for sn in SUBSETS if sn != 'PPV のみ']))
    ab = pd.DataFrame(ab)
    pa_auc = pab[pab.metric == 'auc'].set_index('comparison')
    for c in ['mean_diff', 'rep_q025', 'rep_q975', 'rep_frac_pos']:
        ab[f'vs_ppv_{c}'] = [pa_auc.loc[f'{sn} − PPV のみ', c] if sn != 'PPV のみ' else np.nan for sn in ab.subset]
    ab.to_csv(os.path.join(OUT_DIR, 'pooling_ablation.csv'), index=False, encoding='utf-8-sig')
    log('\n[参考2 プーリングの内訳(MR4_diff の特徴量の部分集合、置換なし、記述のみ)]')
    for _, r in ab.iterrows():
        log(f'  {r.subset}({r.n_features} 特徴量): AUC {r.auc:.3f}(乱数 {r.auc_seed_min:.3f}〜{r.auc_seed_max:.3f})、'
            f'bacc {r.bacc:.3f}、alpha 上限 {r.alpha_at_max:.0%}'
            + ('' if pd.isna(r.vs_ppv_mean_diff) else
               f'、− PPV のみ {r.vs_ppv_mean_diff:+.3f} [{r.vs_ppv_rep_q025:+.3f}, {r.vs_ppv_rep_q975:+.3f}]'))

    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    plot(summary, abl, len(meta))
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'n_features', 'alpha_at_max']
    log('\n[まとめ(参考のラベルなので採用の判定はしない)]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
