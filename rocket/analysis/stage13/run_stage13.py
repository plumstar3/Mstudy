"""
段階13: 中央値分割のラベル3種での評価と、地域内/地域間への分解(EXP-022)。

段階11(評価)と段階12(ペアの分解)を、地域内のペアが多いラベルでまとめて行う。主な問いは
「土壌水分で、同じ市町村×年の中の多収・低収を当てられるか(W)」。
  min2         label_muniyear_median_min2  主
  median       label_muniyear_median       比較(4圃場以上の集団だけ。ラベルのノイズが少ない)
  year_median  label_year_median           参考(年の中のラベル。結論には使わない)

データは段階11と同じ作り方(field_year_labeled.csv の flower_date、開花起点 0〜+59 日、収穫日より後は使わない)で、
ラベルごとに窓を切り出す。入力は1次差分(59点)。
モデル: M(MiniRocket・1次差分、乱数0〜9)、B1(VWC の要約統計 + LR)、B2(緯度・経度・播種日の通日 + LR)。
分割: ラベルごとに make_splits(y, muniCd, 5, 20, 0)。
指標: 全体(fold ごとの AUC・bacc。乱数ごとの20回の繰り返し平均を10通りで平均)と、段階12と同じペアの分解
(W 同じ市町村×年 / B 同じ年・違う市町村 / X 違う年)。
置換検定(1000回、置換 i の乱数は [0, i]):
  min2・median   全体・W・B とも市町村×年の中で並べ替え。集団のクラス構成が保たれるので分割は変わらない
  year_median    全体と B は年の中、W は市町村×年の中。観測ラベルの分割を固定して使う
判定(事前に固定、補正なし): 主 = min2 で M の W の AUC が W の帰無分布の95%点を上回るか。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage13\\run_stage13.py
  試運転: ... run_stage13.py --n-seeds 1 --n-perm 20
出力: rocket/outputs/stage13/、入力: rocket/data/processed/stage13/
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
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage11'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage12'))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage11 as s11  # noqa: E402
import run_stage12 as s12  # noqa: E402
from run_stage6 import BASE_ALPHAS, s5  # noqa: E402
from src.evaluation import evaluate, make_splits, permute_within, split_table  # noqa: E402
from src.windows import extract_windows, first_diff, load_timeseries  # noqa: E402

PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DATA_DIR = os.path.join(PROC, 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
LABELS = {'min2': 'label_muniyear_median_min2', 'median': 'label_muniyear_median', 'year_median': 'label_year_median'}
ROLE = {'min2': '主', 'median': '比較', 'year_median': '参考'}
# 事前の集計: 件数, 多収, 低収, 地域数, 両クラスがそろう集団, W のペア
EXPECT = {'min2': (228, 117, 111, 51, 74, 196), 'median': (123, 64, 59, 21, 25, 147),
          'year_median': (240, 111, 129, 54, 32, 102)}
CONFIGS = ['M', 'B1', 'B2']
TYPES = ['W', 'B', 'X']
POWER_Z = 1.645 + 0.8416  # 片側5%・検出力80% の正規近似

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED
TYPE_COLOR = {'all': '#52514e', 'W': '#2a78d6', 'B': '#eb6834', 'X': '#9a9890'}
TYPE_LABEL = {'all': '全体(fold の AUC)', 'W': 'W 地域内(同じ市町村×年)', 'B': 'B 地域間(同じ年・違う市町村)',
              'X': 'X 年をまたぐ(参考)'}


# ---------------------------------------------------------------- データ
def load_cases(label):
    """label が付いた圃場×年。段階11の load_cases_year を任意のラベルにしたもの。group_n_all は元の表での集団の大きさ。"""
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    lab['muniyear'] = lab['muniCd'] + '_' + lab['year'].astype(str)
    group_n = lab.groupby('muniyear').size()
    d = lab[lab[label].notna()].copy()
    d['y'] = d[label].astype(int)
    d['group_n_all'] = d['muniyear'].map(group_n)
    for c in ['seed_date', 'harvest_date', 'flower_date']:
        d[c] = pd.to_datetime(d[c])
    d['seed_doy'] = d['seed_date'].dt.dayofyear
    d['harvest_dap'] = (d['harvest_date'] - d['seed_date']).dt.days
    d['flower_dap'] = (d['flower_date'] - d['seed_date']).dt.days
    assert d['flower_dap'].notna().all() and d['harvest_dap'].notna().all(), 'flower_dap / harvest_dap に欠測がある'
    d['flower_dap'] = d['flower_dap'].astype(int)
    return d.reset_index(drop=True)


def build_dataset(key, log):
    cases = load_cases(LABELS[key])
    ts = load_timeseries(cases)
    chk = ts.merge(cases[['field_id', 'year', 'seed_date']], on=['field_id', 'year'])
    assert ((chk['date'] - chk['seed_date']).dt.days == chk['dap']).all(), '観測日 − seed_date が dap と一致しない'
    X, meta, excluded = extract_windows(cases, ts, 'flower', 0, 59)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, f'{LABELS[key]}_X.npy'), X)
    meta.to_csv(os.path.join(DATA_DIR, f'{LABELS[key]}_meta.csv'), index=False, encoding='utf-8-sig')
    excluded.to_csv(os.path.join(DATA_DIR, f'{LABELS[key]}_excluded.csv'), index=False, encoding='utf-8-sig')
    log(f'\n[{key}: {LABELS[key]}({ROLE[key]})] ラベル付き {len(cases)} 件 → 窓を覆う {len(meta)} 件、除外 {len(excluded)}')

    y = meta.y.values
    g = meta.groupby('muniyear').y.agg(['size', 'sum'])
    both = g[(g['sum'] > 0) & (g['sum'] < g['size'])]
    one = g[(g['sum'] == 0) | (g['sum'] == g['size'])]
    w_pairs = int((both['sum'] * (both['size'] - both['sum'])).sum())
    got = (len(meta), int(y.sum()), int((1 - y).sum()), meta.muniCd.nunique(), len(both), w_pairs)
    log(f'  {got[0]} 件(多収 {got[1]} / 低収 {got[2]})、地域 {got[3]}、市町村×年 {len(g)}、'
        f'両クラスがそろう集団 {got[4]}({int(both["size"].sum())} 件)、W のペア {got[5]}')
    sz = g['size'].clip(upper=4).value_counts().sort_index()
    log('  窓で絞った後の集団の大きさ: ' + '、'.join(f'{"4以上" if k == 4 else k}圃場 {v}' for k, v in sz.items()))
    szb = both['size'].clip(upper=4).value_counts().sort_index()
    log('  うち両クラスがそろう集団: ' + '、'.join(f'{"4以上" if k == 4 else k}圃場 {v}' for k, v in szb.items()))
    gall = meta.groupby('muniyear').group_n_all.first()
    log('  元の表での集団の大きさ(ラベルを付けたときの集団): ' + '、'.join(
        f'{"4以上" if k == 4 else k}圃場 {v}' for k, v in gall.clip(upper=4).value_counts().sort_index().items()))
    log(f'  窓で絞って片方のクラスだけになった集団: {len(one)}({int(one["size"].sum())} 件)')
    log(f'  年別: ' + ', '.join(f'{yr}年 {n}件' for yr, n in meta.year.value_counts().sort_index().items())
        + f'、開花日 {meta.flower_source.value_counts().to_dict()}')
    assert got == EXPECT[key], f'{key}: 件数などが想定 {EXPECT[key]} と違う: {got}'
    return X, meta


# ---------------------------------------------------------------- 置換
def perm_sets(key, meta, P):
    """全体・B 用と W 用の置換ラベル。min2・median は同じもの(市町村×年の中)。"""
    y = meta.y.values

    def perms(groups):
        g = np.asarray(groups)
        return np.array([permute_within(y, g, np.random.default_rng([0, i])) for i in range(P)]).reshape(P, len(y))
    if key == 'year_median':
        Ym, Yw = perms(meta.year.values), perms(meta.muniyear.to_numpy(dtype=str))
        return Ym, Yw, np.vstack([Ym, Yw]), np.arange(1, P + 1), np.arange(P + 1, 2 * P + 1), 'year'
    Ym = perms(meta.muniyear.to_numpy(dtype=str))
    rows = np.arange(1, P + 1)
    return Ym, Ym, Ym, rows, rows, 'muniyear'


def bacc_rows(dec, Y):
    """dec, Y: (P, n)。予測 = dec > 0(RidgeClassifierCV・LR の predict と同じ)。片方のクラスが無ければ有る方の再現率。"""
    pred = np.atleast_2d(dec) > 0
    pos = np.atleast_2d(Y) == 1
    with np.errstate(invalid='ignore', divide='ignore'):
        tpr = (pred & pos).sum(1) / pos.sum(1)
        tnr = (~pred & ~pos).sum(1) / (~pos).sum(1)
    return np.nanmean(np.c_[tpr, tnr], axis=1)


# ---------------------------------------------------------------- 1つの設定・1つの乱数の集計
def analyse(decs, meta, splits, ps):
    """観測(fold ごとの AUC・bacc、ペアの繰り返しごとの AUC)と帰無分布(全体 (P, 2)、W、B)。"""
    Ym, Yw, _, rows_m, rows_w, _ = ps
    y = meta.y.values
    same = rows_m is rows_w
    obs_counts, cm, cw = [], [], []
    f_auc, f_bacc, n_auc, n_bacc = [], [], [], []
    for (r, f, tr, te), dec in zip(splits, decs):
        masks = s12.pair_masks(meta, te)
        all_mask = np.ones((len(te), len(te)), bool)
        oc = s12.pair_counts(dec[0], y[te], {**masks, 'ALL': all_mask})
        obs_counts.append(oc)
        f_auc.append(oc['ALL'][0][0] / oc['ALL'][1][0] if oc['ALL'][1][0] else np.nan)
        f_bacc.append(bacc_rows(dec[0], y[te])[0])
        mm = {'ALL': all_mask, 'B': masks['B']}
        if same:
            mm['W'] = masks['W']
        c = s12.pair_counts(dec[rows_m], Ym[:, te], mm)
        with np.errstate(invalid='ignore', divide='ignore'):
            n_auc.append(np.where(c['ALL'][1] > 0, c['ALL'][0] / c['ALL'][1], np.nan))
        n_bacc.append(bacc_rows(dec[rows_m], Ym[:, te]))
        cm.append(c)
        cw.append(c if same else s12.pair_counts(dec[rows_w], Yw[:, te], {'W': masks['W']}))
    obs = {'fold_auc': np.array(f_auc), 'fold_bacc': np.array(f_bacc), 'counts': obs_counts}
    for t in TYPES:
        a, n = s12.repeat_auc(obs_counts, splits, t)
        obs[t] = (a[:, 0], n[:, 0])
    null = {'all': np.c_[np.nanmean(np.array(n_bacc), axis=0), np.nanmean(np.array(n_auc), axis=0)]}
    for t, cc in [('W', cw), ('B', cm)]:
        a, _ = s12.repeat_auc(cc, splits, t)
        null[t] = np.nanmean(a, axis=0)
    return obs, null


def check_obs(tag, obs, ref_folds, log):
    """fold の AUC・bacc が sklearn の評価と一致し、W・B・X を合わせると fold の AUC になること。"""
    ok1 = np.allclose(obs['fold_auc'], ref_folds.auc.values, equal_nan=True, rtol=0, atol=1e-12)
    ok2 = np.allclose(obs['fold_bacc'], ref_folds.bacc.values, equal_nan=True, rtol=0, atol=1e-12)
    pooled = []
    for c in obs['counts']:
        n = sum(c[t][1][0] for t in TYPES)
        pooled.append(sum(c[t][0][0] for t in TYPES) / n if n else np.nan)
    ok3 = np.allclose(np.array(pooled), obs['fold_auc'], equal_nan=True, rtol=0, atol=1e-12)
    log(f'    確認 {tag}: fold の AUC・bacc が sklearn と一致 {ok1}・{ok2}、W+B+X = fold の AUC {ok3}')
    assert ok1 and ok2 and ok3, f'{tag}: 一致しない'


# ---------------------------------------------------------------- 起動時の確認
def prechecks(key, Xd, meta, sp, log, n_perm=20):
    y = meta.y.values
    # 2. Pipeline と特徴量の使い回し
    a = evaluate(s6.pipeline('minirocket', 0), Xd, y, sp[:10])
    b = s6.eval_cached(s6.fold_features('minirocket', 0, Xd, sp[:10]), y, sp[:10])
    ok = np.allclose(a.bacc, b.bacc) and np.allclose(a.auc, b.auc, equal_nan=True)
    log(f'  確認2 Pipeline と特徴量の使い回しの一致(M、乱数0、先頭10 fold): {ok}')
    assert ok
    # 4. 置換しても分割が変わらない(min2・median)
    ps = perm_sets(key, meta, n_perm)
    if key != 'year_median':
        same = True
        for i in range(5):
            sp2 = make_splits(ps[0][i], meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
            same &= all(np.array_equal(t1, t2) and np.array_equal(e1, e2)
                        for (_, _, t1, e1), (_, _, t2, e2) in zip(sp, sp2))
        log(f'  確認4 置換(市町村×年の中)5回分で分割が変わらない: {same}')
        assert same
    # 3. まとめて解いた置換 = sklearn で1回ずつ
    feats = s6.fold_features('minirocket', 0, Xd, sp)
    t = time.time()
    decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
    ta = time.time() - t
    t = time.time()
    ref = Parallel(n_jobs=N_JOBS)(delayed(s12._m_fold_sklearn)(Ftr, Fte, ps[2][:, tr])
                                  for (r, f, tr, te), (Ftr, Fte) in zip(sp, feats))
    tb = time.time() - t
    _, na = analyse(decs, meta, sp, ps)
    _, nb = analyse([np.vstack([d[:1], rr]) for d, rr in zip(decs, ref)], meta, sp, ps)
    ok = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'  確認3 置換{n_perm}回分の帰無分布(全体・W・B): まとめて解いた結果 = sklearn で1回ずつ: {ok}'
        f'(学習し直した組 {n_redo}、{ta:.1f}s / {tb:.1f}s)')
    assert ok


# ---------------------------------------------------------------- 記述用
def w_pairs_table(meta):
    """W のペア(i 多収、j 低収、同じ市町村×年)。集団は fold をまたがないので、どの繰り返しでも同じ fold に入る。"""
    rows = []
    my = meta.muniyear.to_numpy(dtype=str)
    for g in np.unique(my):
        idx = np.flatnonzero(my == g)
        hi, lo = idx[meta.y.values[idx] == 1], idx[meta.y.values[idx] == 0]
        for i in hi:
            for j in lo:
                rows.append({'i': i, 'j': j, 'muniyear': g, 'group_n_all': int(meta.group_n_all.iloc[i]),
                             'group_n_win': len(idx), 'yield_gap': meta['yield'].iloc[i] - meta['yield'].iloc[j]})
    return pd.DataFrame(rows)


def pair_concordance(S, pairs):
    """S: (乱数, 繰り返し, n) のスコア。戻り値 (乱数, 繰り返し, ペア) の一致(同点 0.5)。"""
    si, sj = S[:, :, pairs.i.values], S[:, :, pairs.j.values]
    return (si > sj) + 0.5 * (si == sj)


def by_category(C, cat):
    """カテゴリごとの AUC(乱数ごとに繰り返しで平均し、乱数で平均)。"""
    out = []
    for k in pd.unique(cat):
        m = (cat == k)
        per_seed = C[:, :, m].mean(axis=2).mean(axis=1)
        out.append({'category': k, 'n_pairs': int(m.sum()), 'auc_mean': per_seed.mean(),
                    'auc_seed_min': per_seed.min(), 'auc_seed_max': per_seed.max()})
    return pd.DataFrame(out)


def univariate_pairs(meta, score):
    """学習させない単変量の AUC を、全件のペア(fold を問わない)で W・B・X・全体に分けて。"""
    y = meta.y.values
    my = meta.muniyear.to_numpy(dtype=str)
    yr = meta.year.values
    pn = (y[:, None] == 1) & (y[None, :] == 0)
    cmp = (score[:, None] > score[None, :]) + 0.5 * (score[:, None] == score[None, :])
    tie = score[:, None] == score[None, :]
    masks = {'W': my[:, None] == my[None, :], 'B': (yr[:, None] == yr[None, :]) & (my[:, None] != my[None, :]),
             'X': yr[:, None] != yr[None, :]}
    masks['ALL'] = np.ones_like(pn)
    return [{'type': t, 'n_pairs': int((pn & m).sum()), 'auc': cmp[pn & m].mean(), 'tie_rate': tie[pn & m].mean()}
            for t, m in masks.items()]


# ---------------------------------------------------------------- 図
def plot(pair_auc, folds, summary):
    keys = list(LABELS)
    fig, axs = plt.subplots(len(keys), 1, figsize=(13, 4.2 * len(keys)), sharey=True)
    types = ['all', 'W', 'B', 'X']
    sm = summary.set_index(['label', 'config', 'metric'])
    for ax, key in zip(axs, keys):
        pos, ticks = 0, []
        for c in CONFIGS:
            for k, t in enumerate(types):
                x = pos + k
                if t == 'all':
                    d = folds[(folds.label == key) & (folds.config == c)].groupby('repeat').auc.mean().values
                    met = 'all_auc'
                else:
                    d = pair_auc[(pair_auc.label == key) & (pair_auc.config == c) & (pair_auc.type == t)] \
                        .groupby('repeat').auc.mean().dropna().values
                    met = t
                bp = ax.boxplot([d], positions=[x], widths=0.6, patch_artist=True,
                                medianprops=dict(color=INK, lw=1.2), boxprops=dict(lw=0),
                                whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                                flierprops=dict(marker='o', ms=2.5, mfc=MUTED, mec='none'))
                bp['boxes'][0].set_facecolor(TYPE_COLOR[t])
                bp['boxes'][0].set_alpha(0.55)
                r = sm.loc[(key, c, met)]
                ax.plot(x, r['mean'], marker='D', ms=4.5, color=INK, zorder=3)
                if np.isfinite(r.null_q95):
                    ax.hlines(r.null_q95, x - 0.38, x + 0.38, color=s6.C_RED, lw=2, ls='--')
            ticks.append((pos + 1.5, c))
            pos += 5
        for p in [4, 9]:
            ax.axvline(p, color=MUTED, lw=0.8)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks([p for p, _ in ticks], [n for _, n in ticks], fontsize=10)
        n = int(summary[(summary.label == key)].n.iloc[0])
        ax.set_title(f'{key}: {LABELS[key]}({ROLE[key]}、{n}件)', fontsize=11, color=INK, loc='left')
        ax.set_ylabel('AUC(繰り返しごと)', color=INK)
        s6.style(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=TYPE_COLOR[t], alpha=0.55) for t in types] +
               [Line2D([], [], marker='D', color=INK, ls='', ms=5), Line2D([], [], color=s6.C_RED, lw=2, ls='--')],
               [TYPE_LABEL[t] for t in types] + ['平均(M は乱数で平均)', '帰無分布の95%点'],
               loc='lower left', ncol=6, frameon=False, fontsize=9)
    fig.suptitle('段階13: ラベル × 設定 × ペアの種類の AUC(箱 = 20回の繰り返し)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(os.path.join(OUT_DIR, 'pair_auc.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 1つのラベル
def run_label(key, X, meta, seeds, P, log):
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    prechecks(key, Xd, meta, sp, log)
    ps = perm_sets(key, meta, P)
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': [], 'summary': [], 'S': None}
    feats_bl = s11.baseline_features(X, meta)
    for config in CONFIGS:
        t0 = time.time()
        nulls = {'all': [], 'W': [], 'B': []}
        S = []
        for seed in (seeds if config == 'M' else [None]):
            if config == 'M':
                feats = s6.fold_features('minirocket', seed, Xd, sp)
                ref = s6.eval_cached(feats, y, sp)
                decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
                del feats
            else:
                decs, n_redo = s12.lr_decisions(feats_bl[config], y, sp, ps[2]), 0
                ref = evaluate(s5.lr(), feats_bl[config], y, sp)
                ref = ref.assign(n_train=[len(tr) for _, _, tr, _ in sp], alpha_over_n=np.nan)
            obs, null = analyse(decs, meta, sp, ps)
            check_obs(f'{key} {config} 乱数{seed}', obs, ref, log)
            ref = ref.assign(label=key, config=config, seed=seed)
            out['folds'].append(ref)
            for k in nulls:
                nulls[k].append(null[k])
            sc = np.full((N_REPEATS, len(y)), np.nan)
            for (r, f, tr, te), dec in zip(sp, decs):
                sc[r, te] = dec[0]
                m = meta.iloc[te]
                out['scores'].append(pd.DataFrame({'label': key, 'config': config, 'seed': seed, 'repeat': r, 'fold': f,
                                                   'field_id': m.field_id.values, 'year': m.year.values,
                                                   'muniCd': m.muniCd.values, 'muniyear': m.muniyear.values,
                                                   'y': m.y.values, 'score': dec[0]}))
            S.append(sc)
            ps_row = {'label': key, 'config': config, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
                      'alpha_at_max': np.isclose(ref.alpha_over_n, BASE_ALPHAS[-1]).mean() if config == 'M' else np.nan}
            for t in TYPES:
                a, n = obs[t]
                ps_row[t] = np.nanmean(a)
                out['pair'].append(pd.DataFrame({'label': key, 'config': config, 'type': t, 'seed': seed,
                                                 'repeat': range(len(a)), 'auc': a, 'n_pairs': n}))
            out['seed'].append(ps_row)
            if config == 'M':
                log(f'    M 乱数{seed}: 全体 AUC {ps_row["all_auc"]:.3f} bacc {ps_row["all_bacc"]:.3f}  '
                    + '  '.join(f'{t} {ps_row[t]:.3f}' for t in TYPES) + f'(学習し直した組 {n_redo})')
        if config == 'M':
            out['S'] = np.stack(S)
        else:
            out.setdefault('S_bl', {})[config] = np.stack(S)
        # まとめ
        per_seed = pd.DataFrame([r for r in out['seed'] if r['config'] == config])
        pa = pd.concat([p for p in out['pair'] if p.config.iloc[0] == config])
        null_all = np.mean(np.stack(nulls['all']), axis=0)
        np.save(os.path.join(OUT_DIR, f'null_{key}_{config}_all.npy'), null_all)
        base = {'label': key, 'config': config, 'n': len(y), 'n_high': int(y.sum()), 'n_low': int((1 - y).sum()),
                'n_seeds': len(per_seed), 'n_perm': P}
        for met, j in [('all_auc', 1), ('all_bacc', 0)]:
            v = per_seed[met]
            nl = null_all[:, j]
            out['summary'].append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                                   'null_group': ps[5], 'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                                   'null_sd': np.nanstd(nl), 'p': (1 + np.sum(nl >= v.mean())) / (1 + P),
                                   'alpha_at_max': per_seed.alpha_at_max.mean()})
        for t in TYPES:
            v = per_seed[t]
            r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                 'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean(),
                 'alpha_at_max': per_seed.alpha_at_max.mean()}
            if t in ('W', 'B'):
                nl = np.mean(np.stack(nulls[t]), axis=0)
                np.save(os.path.join(OUT_DIR, f'null_{key}_{config}_{t}.npy'), nl)
                r.update({'null_group': 'muniyear' if t == 'W' else ps[5], 'null_mean': np.nanmean(nl),
                          'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                          'p': (1 + np.sum(nl >= v.mean())) / (1 + P),
                          'auc_needed_80pct': 0.5 + POWER_Z * np.nanstd(nl)})
            out['summary'].append(r)
        s = pd.DataFrame([r for r in out['summary'] if r['config'] == config])
        log(f'  {config:2s} ' + '  '.join(
            f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95')) else '')
            for _, r in s.iterrows()) + f'  [{time.time() - t0:.0f}s]')
    out['splits'] = split_table(sp, y, meta.muniCd.values).assign(label=key)
    return out


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
    data = {k: build_dataset(k, log) for k in LABELS}

    res = {}
    for key, (X, meta) in data.items():
        log(f'\n[{key}: {LABELS[key]}({ROLE[key]})]')
        res[key] = run_label(key, X, meta, seeds, P, log)

    summary = pd.DataFrame([r for k in res for r in res[k]['summary']])
    for met in ['all_auc', 'all_bacc', 'W', 'B']:
        best = summary[(summary.metric == met) & (summary.config != 'M')].groupby('label')['mean'].max()
        m = summary.metric == met
        summary.loc[m, 'best_baseline'] = summary.loc[m, 'label'].map(best)
    has_null = summary.null_q95.notna()
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(has_null, None)
    summary['hit'] = ((summary['mean'] > summary.null_q95) & (summary['mean'] > summary.best_baseline)) \
        .astype(object).where(has_null & (summary.config == 'M'), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat([f for k in res for f in res[k]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for k in res for r in res[k]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([res[k]['splits'] for k in res]).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False,
                                                      encoding='utf-8-sig')
    pd.concat([s for k in res for s in res[k]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pair_auc = pd.concat([p for k in res for p in res[k]['pair']])
    pair_auc.to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    # ---- 記述用(M の W)
    wg, gs, yg, uv = [], [], [], []
    for key, (X, meta) in data.items():
        pairs = w_pairs_table(meta)
        C = pair_concordance(res[key]['S'], pairs)  # (乱数, 繰り返し, ペア)
        w_ref = summary[(summary.label == key) & (summary.config == 'M') & (summary.metric == 'W')]['mean'].iloc[0]
        assert np.isclose(C.mean(axis=2).mean(axis=1).mean(), w_ref), f'{key}: W のペアの表と W の AUC が一致しない'
        for g, d in pairs.groupby('muniyear'):
            per = C[:, :, d.index.values].mean(axis=2)  # (乱数, 繰り返し)
            m = meta[meta.muniyear == g]
            wg.append({'label': key, 'muniyear': g, 'muniCd': m.muniCd.iloc[0], 'year': int(m.year.iloc[0]),
                       'n': len(m), 'n_high': int(m.y.sum()), 'group_n_all': int(m.group_n_all.iloc[0]),
                       'n_pairs': len(d), 'auc_mean': per.mean(), 'auc_sd': per.std()})
        if key in ('min2', 'median'):
            cat = np.where(pairs.group_n_all.values >= 4, '4圃場以上', '2〜3圃場')
            t = by_category(C, cat).assign(label=key)
            t['n_groups'] = [pairs[cat == k].muniyear.nunique() for k in t.category]
            gs.append(t)
        q = pd.qcut(pairs.yield_gap, 3, labels=['小', '中', '大'])
        t = by_category(C, q.astype(str).values).assign(label=key)
        rng = pairs.groupby(q, observed=True).yield_gap.agg(['min', 'max'])
        t['gap_min'] = t.category.map(rng['min'])
        t['gap_max'] = t.category.map(rng['max'])
        yg.append(t.set_index('category').loc[['小', '中', '大']].reset_index())
        for r in univariate_pairs(meta, meta.seed_doy.values.astype(float)):
            uv.append({'label': key, 'n': len(meta), **r})
    X11 = pd.read_csv(os.path.join(ROCKET_DIR, 'data', 'processed', 'stage11', 'year_main_meta.csv'),
                      encoding='utf-8-sig', dtype={'muniCd': str})
    for r in univariate_pairs(X11, X11.seed_doy.values.astype(float)):
        uv.append({'label': 'label_year(段階11・12)', 'n': len(X11), **r})
    wg = pd.DataFrame(wg).sort_values(['label', 'auc_mean'], ascending=[True, False])
    wg.to_csv(os.path.join(OUT_DIR, 'within_groups.csv'), index=False, encoding='utf-8-sig')
    gs = pd.concat(gs)
    gs.to_csv(os.path.join(OUT_DIR, 'within_by_groupsize.csv'), index=False, encoding='utf-8-sig')
    yg = pd.concat(yg)
    yg.to_csv(os.path.join(OUT_DIR, 'within_by_yieldgap.csv'), index=False, encoding='utf-8-sig')
    uv = pd.DataFrame(uv)
    uv.to_csv(os.path.join(OUT_DIR, 'seed_doy_univariate.csv'), index=False, encoding='utf-8-sig')

    plot(pair_auc, folds, summary)

    show = ['label', 'config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p',
            'best_baseline', 'above_q95', 'hit', 'n_pairs_per_repeat', 'auc_needed_80pct', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.label == 'min2') & (summary.config == 'M') & (summary.metric == 'W')].iloc[0]
    log(f'\n[主の判定] min2 の M の W: {r["mean"]:.3f}、W の帰無分布の95%点 {r.null_q95:.3f}、p={r.p:.3f} → '
        + ('上回る(地域内でも当たっている)' if r['mean'] > r.null_q95 else '上回らない'))
    log('\n[W を集団の大きさ別に(M)]')
    log(gs.round(3).to_string(index=False))
    log('\n[W を収量の差別に(M)]')
    log(yg.round(3).to_string(index=False))
    log('\n[播種日の通日の単変量 AUC(全件のペア、学習なし)]')
    log(uv.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
