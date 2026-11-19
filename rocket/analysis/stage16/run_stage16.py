"""
段階16: 暦の日付でそろえた集団内中心化(EXP-025)。

同じ市町村×年の圃場は同じ日に雨が降り、同じ天候の中で乾いていく。この共通の動きは集団の中の多収・低収の手がかりに
ならないので、取り除けば圃場ごとの違いが残り、地域内 W と全体の AUC が上がる、という仮説を確かめる。
段階5・6(実験 F2)の groupcenter は開花起点の窓の「何日目か」でそろえて平均を引いたので、同じ集団でも暦の日付が
ずれていた。今回は暦の日付でそろえて平均を引き、その後で窓を切り出す。

  GC_date   主。集団(label_muniyear_median の123件が属する28集団)の全圃場(窓に入らない件・ラベルなしの件も含む、
            収穫日より後は除く、系列が他の圃場と全日同じ / 全日一定の圃場は平均から除く)について、暦の日付ごとに
            「その圃場を除いたほかの圃場」の VWC の平均を引く → 開花起点 0〜+59 日を切り出す → 1次差分
  GC_index  参考。段階13の median の窓(生の値)に run_stage5.groupcenter(窓の何日目かでそろえ、自分を含む平均)→ 1次差分
  M_diff    比べる相手。段階13の median・M(outputs/stage13/ の folds.csv・oof_scores.csv・summary.csv・null)

モデル・分割・置換は段階13・14と同じ。判定(事前に固定): GC_date − M_diff の繰り返し平均の 2.5〜97.5%点が0より上、
かつ GC_date の全体の AUC が帰無分布の95%点を上回れば、GC_date を主の入力に採用する。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage16\\run_stage16.py
  試運転: ... run_stage16.py --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage16/、入力: rocket/data/processed/stage16/
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
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13', 'stage14']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage12 as s12  # noqa: E402
import run_stage13 as s13  # noqa: E402
import run_stage14 as s14  # noqa: E402
from run_stage6 import BASE_ALPHAS, s5  # noqa: E402
from src.evaluation import make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY = 'median'
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DATA_DIR = os.path.join(PROC, 'stage16')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage16')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
TYPES = ['W', 'B', 'X']
POWER_Z = s13.POWER_Z
WIN = 60
CFGS = ['M_diff', 'GC_date', 'GC_index']
CFG_LABEL = {'M_diff': 'M_diff 1次差分(段階13)', 'GC_date': 'GC_date 暦の日付で中心化', 'GC_index': 'GC_index 窓の日目で中心化(段階6の方式)'}
CFG_COLOR = {'M_diff': '#9a9890', 'GC_date': '#2a78d6', 'GC_index': '#eb6834'}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- 中心化(GC_date)
def load_members(meta, log):
    """123件が属する集団の全圃場×年の VWC(暦の日付 × 圃場)。収穫日より後は除く。"""
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str},
                      parse_dates=['seed_date', 'harvest_date', 'flower_date'])
    lab['muniyear'] = lab['muniCd'] + '_' + lab['year'].astype(str)
    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig', parse_dates=['date'])
    # 全日で同じ値の系列(他の圃場と全日一致 / 全日一定)を、データ全体で探す
    key = {k: (tuple(s.date.dt.strftime('%Y%m%d')), tuple(np.round(s.VWC.values, 10)))
           for k, s in ts.sort_values('date').groupby(['field_id', 'year'])}
    by_series = {}
    for k, v in key.items():
        by_series.setdefault(v, []).append(k)
    dup = {k for ks in by_series.values() if len(ks) > 1 for k in ks}
    const = {k for k, s in ts.groupby(['field_id', 'year']) if s.VWC.nunique() == 1}
    groups = sorted(meta.muniyear.unique())
    mem = lab[lab.muniyear.isin(groups)][['field_id', 'year', 'muniyear', 'muniCd', 'seed_date', 'harvest_date',
                                         'flower_date', 'label_muniyear_median']].copy()
    ts = ts.merge(mem[['field_id', 'year', 'muniyear', 'harvest_date']], on=['field_id', 'year'])
    n0 = len(ts)
    ts = ts[ts.date <= ts.harvest_date]
    has = ts.groupby(['field_id', 'year']).size()
    mem = mem.set_index(['field_id', 'year'])
    mem['n_obs'] = has.reindex(mem.index).fillna(0).astype(int)
    mem['in_123'] = mem.index.isin(list(zip(meta.field_id, meta.year)))
    mem['excluded_from_mean'] = [('他の圃場と全日同じ' if k in dup else '全日一定' if k in const else '')
                                 for k in mem.index]
    mem = mem.reset_index()
    log(f'\n[GC_date の平均に使う圃場] 28集団の全圃場×年 {len(mem)}(うち123件 {mem.in_123.sum()}、窓に入らない件・'
        f'ラベルなしの件 {(~mem.in_123).sum()})。収穫日より後の観測 {n0 - len(ts)} 行を除いた')
    log(f'  データ全体で、系列が他の圃場と全日同じ: {sorted(dup)}、全日一定: {sorted(const)}')
    ex = mem[mem.excluded_from_mean != '']
    log('  このうち28集団に入り、平均から除く圃場: '
        + ('なし' if ex.empty else '、'.join(f'{r.field_id}({r.year}、{r.muniyear}、{r.excluded_from_mean})'
                                           for _, r in ex.iterrows())))
    if (mem.n_obs == 0).any():
        log(f'  観測の無い圃場×年: {mem[mem.n_obs == 0][["field_id", "year"]].values.tolist()}')
    return mem, ts


def loo_center(mem, ts):
    """集団ごとに、暦の日付 × 圃場の表を作り、その圃場を除いたほかの圃場の平均を引く。

    戻り値: {(field_id, year): DataFrame(date, VWC, loo_mean, n_others, centered)}
    """
    out = {}
    use = set(map(tuple, mem[mem.excluded_from_mean == ''][['field_id', 'year']].values))
    for g, d in ts.groupby('muniyear'):
        P = d.pivot_table(index='date', columns=['field_id', 'year'], values='VWC', aggfunc='first').sort_index()
        inc = np.array([c in use for c in P.columns])
        V = P.values
        obs = ~np.isnan(V)
        Vi = np.where(obs & inc[None, :], V, 0.0)
        S, C = Vi.sum(axis=1), (obs & inc[None, :]).sum(axis=1)
        for j, c in enumerate(P.columns):
            own = obs[:, j] & inc[j]
            s_o = S - np.where(own, V[:, j], 0.0)
            n_o = C - own.astype(int)
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_o = np.where(n_o > 0, s_o / n_o, np.nan)
            out[c] = pd.DataFrame({'date': P.index, 'VWC': V[:, j], 'loo_mean': mean_o, 'n_others': n_o,
                                   'centered': V[:, j] - mean_o}).dropna(subset=['VWC']).reset_index(drop=True)
    return out


def hand_check(cent, mem, ts, log):
    """確認6: 1集団・1日について、平均と引いた値を素直なループで計算し直して照合する。"""
    g = mem.groupby('muniyear').size().idxmax()
    m = mem[mem.muniyear == g]
    k0 = tuple(m[m.in_123].iloc[0][['field_id', 'year']])
    c0 = cent[k0]
    day = c0.date.iloc[len(c0) // 2]
    vals = []
    for _, r in m.iterrows():
        k = (r.field_id, r.year)
        if k == k0 or r.excluded_from_mean:
            continue
        v = ts[(ts.field_id == k[0]) & (ts.year == k[1]) & (ts.date == day)].VWC
        if len(v):
            vals.append(float(v.iloc[0]))
    own = float(ts[(ts.field_id == k0[0]) & (ts.year == k0[1]) & (ts.date == day)].VWC.iloc[0])
    mean = sum(vals) / len(vals)
    row = c0[c0.date == day].iloc[0]
    ok = np.isclose(mean, row.loo_mean, rtol=0, atol=1e-12) and np.isclose(own - mean, row.centered, rtol=0, atol=1e-12)
    log(f'  確認6 手計算: 集団 {g}、圃場 {k0[0]}({k0[1]})、{day.date()}: ほかの圃場 {len(vals)} 件の平均 {mean:.6f}、'
        f'自分 {own:.6f}、引いた値 {own - mean:+.6f} / スクリプト {row.loo_mean:.6f}・{row.centered:+.6f} → 一致 {ok}')
    assert ok


def build_gc_date(X13, meta, log):
    mem, ts = load_members(meta, log)
    cent = loo_center(mem, ts)
    hand_check(cent, mem, ts, log)
    X, keep, excl, max_raw = [], [], [], 0.0
    for i, r in meta.iterrows():
        c = cent.get((r.field_id, r.year))
        dates = pd.date_range(r.window_start_date, r.window_end_date)
        assert len(dates) == WIN
        w = c.set_index('date').reindex(dates)
        max_raw = max(max_raw, np.nanmax(np.abs(w.VWC.values - X13[i, 0])))
        if w.centered.isna().any():
            nn = int(w.centered.isna().sum())
            excl.append({'field_id': r.field_id, 'year': r.year, 'muniyear': r.muniyear,
                         'reason': f'窓の中で、ほかの圃場に観測が無い日が {nn} 日ある'})
            continue
        X.append(w.centered.values)
        keep.append(i)
    log(f'  窓の生の値が段階13の X と一致(最大差 {max_raw:.1e})')
    assert max_raw < 1e-12
    Xg = np.array(X)[:, None, :]
    mg = meta.loc[keep].reset_index(drop=True)
    n_oth = np.array([cent[(r.field_id, r.year)].set_index('date').reindex(
        pd.date_range(r.window_start_date, r.window_end_date)).n_others.values for _, r in mg.iterrows()])
    mg['n_others_min'] = n_oth.min(axis=1)
    mg['n_others_mean'] = n_oth.mean(axis=1)
    exd = pd.DataFrame(excl, columns=['field_id', 'year', 'muniyear', 'reason'])
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'GC_date_X.npy'), Xg)
    mg.to_csv(os.path.join(DATA_DIR, 'GC_date_meta.csv'), index=False, encoding='utf-8-sig')
    exd.to_csv(os.path.join(DATA_DIR, 'GC_date_excluded.csv'), index=False, encoding='utf-8-sig')
    log(f'  確認1 GC_date の件数 {len(mg)}(段階13は {len(meta)})、除外 {len(exd)}'
        + ('' if exd.empty else ': ' + '、'.join(f'{r.field_id}({r.year}) {r.reason}' for _, r in exd.iterrows())))
    log(f'  窓の各日で平均に使ったほかの圃場の数: 最小 {int(n_oth.min())}、平均 {n_oth.mean():.2f}、'
        f'1件だけの日を含む件 {(n_oth.min(axis=1) == 1).sum()}')
    return Xg, mg, exd, mem, ts, cent


# ---------------------------------------------------------------- 処理の確認(記述用)
def gc_check(meta, inputs, log):
    rows = []
    fl = pd.to_datetime(meta.flower_date)
    rng = fl.groupby(meta.muniyear.values).agg(lambda s: (s.max() - s.min()).days)
    for g, v in rng.items():
        rows.append({'item': 'flower_range_days', 'muniyear': g, 'config': '', 'value': v,
                     'n': int((meta.muniyear == g).sum())})
    q = rng.describe()
    log(f'\n[処理の確認] 集団の中の開花日の範囲(最大 − 最小、123件、28集団): 中央値 {q["50%"]:.0f} 日、'
        f'平均 {q["mean"]:.1f}、最小 {q["min"]:.0f}、最大 {q["max"]:.0f}、0日の集団 {(rng == 0).sum()}、'
        f'7日以上の集団 {(rng >= 7).sum()}')
    my = meta.muniyear.to_numpy(dtype=str)
    for name, D in inputs.items():
        per = []
        for g in np.unique(my):
            idx = np.flatnonzero(my == g)
            if len(idx) < 2:
                continue
            Cm = np.corrcoef(D[idx, 0])
            iu = np.triu_indices(len(idx), 1)
            v = np.nanmean(Cm[iu])
            per.append((g, v, len(iu[0])))
            rows.append({'item': 'mean_corr_diff', 'muniyear': g, 'config': name, 'value': v, 'n': len(idx)})
        vals = np.array([v for _, v, _ in per])
        w = np.array([n for *_, n in per])
        rows.append({'item': 'mean_corr_diff_all', 'muniyear': '(全体: 集団の平均)', 'config': name,
                     'value': vals.mean(), 'n': len(per)})
        rows.append({'item': 'mean_corr_diff_all_pairs', 'muniyear': '(全体: ペアで重み付け)', 'config': name,
                     'value': (vals * w).sum() / w.sum(), 'n': int(w.sum())})
        log(f'  同じ集団の圃場どうしの、窓内の1次差分の相関の平均 {name}: 集団の平均 {vals.mean():+.3f}、'
            f'ペアで重み付け {(vals * w).sum() / w.sum():+.3f}(集団ごとの最小 {vals.min():+.3f}・最大 {vals.max():+.3f})')
    return pd.DataFrame(rows)


def plot_examples(meta, mem, cent, groups):
    fig, axs = plt.subplots(2, len(groups), figsize=(5.2 * len(groups), 7), sharey='row')
    for j, g in enumerate(groups):
        m = mem[mem.muniyear == g]
        for _, r in m.iterrows():
            c = cent[(r.field_id, r.year)]
            lab = r.label_muniyear_median
            col = '#2a78d6' if lab == 1 else ('#eb6834' if lab == 0 else MUTED)
            ls = '-' if r.in_123 else ':'
            axs[0, j].plot(c.date, c.VWC, color=col, lw=1.1, ls=ls)
            axs[1, j].plot(c.date, c.centered, color=col, lw=1.1, ls=ls)
            if r.in_123:
                mr = meta[(meta.field_id == r.field_id) & (meta.year == r.year)].iloc[0]
                axs[1, j].axvspan(pd.Timestamp(mr.window_start_date), pd.Timestamp(mr.window_end_date),
                                  color=col, alpha=0.03)
        fl = pd.to_datetime(meta[meta.muniyear == g].flower_date)
        axs[0, j].set_title(f'{g}({len(m)}圃場、開花日の範囲 {(fl.max() - fl.min()).days}日)', fontsize=10, color=INK,
                            loc='left')
        axs[1, j].axhline(0, color=MUTED, lw=0.8)
        for a in axs[:, j]:
            a.tick_params(axis='x', labelrotation=30, labelsize=8)
            s6.style(a)
    axs[0, 0].set_ylabel('VWC(中心化の前)', color=INK)
    axs[1, 0].set_ylabel('VWC − ほかの圃場の平均(GC_date)', color=INK)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color='#2a78d6'), Line2D([], [], color='#eb6834'), Line2D([], [], color=MUTED, ls=':')],
               ['多収', '低収', 'ラベルなし・窓に入らない(平均には使う)'], loc='lower center', ncol=3, frameon=False)
    fig.suptitle('段階16: 暦の日付でそろえた集団内中心化の例(下段の薄い帯は各圃場の窓)', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'gc_examples.png'), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- モデル
def run_m(name, Xd, meta, sp, ps, seeds, P, log):
    y = meta.y.values
    out = {'folds': [], 'pair': [], 'scores': [], 'seed': []}
    nulls = {'all': [], 'W': [], 'B': []}
    S = np.full((len(seeds), N_REPEATS, len(y)), np.nan)
    t0 = time.time()
    for k, seed in enumerate(seeds):
        feats = s6.fold_features('minirocket', seed, Xd, sp)
        ref = s6.eval_cached(feats, y, sp)
        decs, n_redo = s12.m_decisions(feats, y, sp, ps[2])
        del feats
        obs, null = s13.analyse(decs, meta, sp, ps)
        s13.check_obs(f'{name} 乱数{seed}', obs, ref, log)  # 確認5
        out['folds'].append(ref.assign(config=name, seed=seed))
        for kk in nulls:
            nulls[kk].append(null[kk])
        for (r, f, tr, te), dec in zip(sp, decs):
            S[k, r, te] = dec[0]
            m = meta.iloc[te]
            out['scores'].append(pd.DataFrame({'config': name, 'seed': seed, 'repeat': r, 'fold': f,
                                               'field_id': m.field_id.values, 'year': m.year.values,
                                               'muniCd': m.muniCd.values, 'muniyear': m.muniyear.values,
                                               'y': m.y.values, 'score': dec[0]}))
        row = {'config': name, 'seed': seed, 'all_auc': ref.auc.mean(), 'all_bacc': ref.bacc.mean(),
               'alpha_at_max': np.isclose(ref.alpha_over_n, BASE_ALPHAS[-1]).mean(), 'n_redo': n_redo}
        for t in TYPES:
            a, n = obs[t]
            row[t] = np.nanmean(a)
            out['pair'].append(pd.DataFrame({'config': name, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                             'auc': a, 'n_pairs': n}))
        out['seed'].append(row)
        log(f'    {name} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
            + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES) + f'(学習し直した組 {n_redo})')
    per_seed = pd.DataFrame(out['seed'])
    pa = pd.concat(out['pair'])
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    np.save(os.path.join(OUT_DIR, f'null_{name}_all.npy'), null_all)
    base = {'config': name, 'n': len(y), 'n_seeds': len(per_seed), 'n_perm': P,
            'alpha_at_max': per_seed.alpha_at_max.mean(), 'source': '段階16'}
    summ = []
    for met, j in [('all_auc', 1), ('all_bacc', 0)]:
        v, nl = per_seed[met], null_all[:, j]
        summ.append({**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
                     'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                     'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
    for t in TYPES:
        v = per_seed[t]
        r = {**base, 'metric': t, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max(),
             'n_pairs_per_repeat': pa[pa.type == t].groupby('repeat').n_pairs.first().mean()}
        if t in ('W', 'B'):
            nl = np.mean(np.stack(nulls[t]), axis=0)
            np.save(os.path.join(OUT_DIR, f'null_{name}_{t}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                      'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
        summ.append(r)
    s = pd.DataFrame(summ)
    log(f'  {name} ' + '  '.join(
        f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95')) else '')
        for _, r in s.iterrows()) + f'  [{time.time() - t0:.0f}s]')
    out['summary'], out['S'] = summ, S
    return out


def m_diff_from_stage13(meta, sp, Xd_raw, log):
    """段階13の median・M。起動時に乱数0の先頭10 fold を計算し直して folds.csv と一致を確かめる。"""
    folds = pd.read_csv(os.path.join(S13_DIR, 'folds.csv'), encoding='utf-8-sig')
    folds = folds[(folds.label == KEY) & (folds.config == 'M')].copy()
    f0 = folds[folds.seed == 0].set_index(['repeat', 'fold']).loc[[(r, f) for r, f, *_ in sp[:10]]]
    a = s6.eval_cached(s6.fold_features('minirocket', 0, Xd_raw, sp[:10]), meta.y.values, sp[:10])
    ok = np.allclose(a.auc.values, f0.auc.values, rtol=0, atol=1e-12) and np.allclose(a.bacc.values, f0.bacc.values)
    log(f'  M_diff: 段階13の folds.csv の乱数0・先頭10 fold を計算し直して一致 {ok}')
    assert ok, '段階13の M_diff と一致しない。止めて報告する'
    S, seeds13, _ = s14.load_stage13(meta, sp, log)
    s13s = pd.read_csv(os.path.join(S13_DIR, 'summary.csv'), encoding='utf-8-sig')
    s13s = s13s[(s13s.label == KEY) & (s13s.config == 'M')]
    summ = []
    for _, r in s13s.iterrows():
        summ.append({'config': 'M_diff', 'n': int(r.n), 'n_seeds': int(r.n_seeds), 'n_perm': int(r.n_perm),
                     'alpha_at_max': r.alpha_at_max, 'source': '段階13(outputs/stage13/)', 'metric': r.metric,
                     'mean': r['mean'], 'seed_min': r.seed_min, 'seed_max': r.seed_max, 'null_mean': r.null_mean,
                     'null_q95': r.null_q95, 'null_sd': r.null_sd, 'p': r.p,
                     'n_pairs_per_repeat': r.n_pairs_per_repeat})
    pa = pd.read_csv(os.path.join(S13_DIR, 'pair_auc.csv'), encoding='utf-8-sig')
    pa = pa[(pa.label == KEY) & (pa.config == 'M')].drop(columns='label').assign(config='M_diff')
    return folds.assign(config='M_diff'), S, seeds13, summ, pa


# ---------------------------------------------------------------- W の差のブートストラップ
def w_diff_bootstrap(Sa, Sb, meta, sp, n_boot):
    """W の差(a − b)を、市町村を単位にしたクラスターブートストラップで(段階14の A1 と同じ重み k_c)。"""
    masks, _ = s14.full_masks(meta)
    res = {}
    for nm, S in [('a', Sa), ('b', Sb)]:
        C, N = s14.a1_pair_sums(S, meta, sp)
        res[nm] = ((C * masks['W']).sum(axis=1), (N * masks['W']).sum(axis=1))
    units, cidx = np.unique(meta.muniCd.to_numpy(dtype=str), return_inverse=True)
    pt = {nm: res[nm][0].sum() / res[nm][1].sum() for nm in res}
    rng = np.random.default_rng(0)
    rows = []
    for b in range(n_boot):
        k = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = k[cidx].astype(float)
        wa = (w @ res['a'][0]) / (w @ res['a'][1]) if w @ res['a'][1] > 0 else np.nan
        wb = (w @ res['b'][0]) / (w @ res['b'][1]) if w @ res['b'][1] > 0 else np.nan
        rows.append({'boot': b, 'W_GC_date': wa, 'W_M_diff': wb, 'W_diff': wa - wb})
    return pd.DataFrame(rows), pt


# ---------------------------------------------------------------- 図
def plot_overview(summary):
    fig, ax = plt.subplots(figsize=(10, 5))
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内'), ('B', 'B 地域間')]
    sm = summary.set_index(['config', 'metric'])
    w = 0.26
    for k, c in enumerate(CFGS):
        xs = np.arange(len(mets)) + (k - 1) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.92, color=CFG_COLOR[c], alpha=0.85, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95']
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(min(0.35, summary['mean'].min() - 0.03), max(0.72, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title(f'段階16: 集団内中心化の比較(label_muniyear_median、{int(summary.n.iloc[0])}件)', fontsize=12,
                 color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stage16_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
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
    X13, meta13 = s14.load()
    Xg, meta, exd, mem, ts, cent = build_gc_date(X13, meta13, log)
    same = len(meta) == len(meta13) and (meta[['field_id', 'year']].values == meta13[['field_id', 'year']].values).all()
    if not same:
        raise SystemExit('GC_date の件が段階13の123件と違う。比較の方法を報告して確認を取る')
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    split_table(sp, y, meta.muniCd.values).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    D = {'M_diff': first_diff(X13), 'GC_date': first_diff(Xg), 'GC_index': first_diff(s5.groupcenter(X13, meta13))}

    chk = gc_check(meta, {'中心化の前(M_diff)': D['M_diff'], 'GC_date': D['GC_date'], 'GC_index': D['GC_index']}, log)
    chk.to_csv(os.path.join(OUT_DIR, 'gc_check.csv'), index=False, encoding='utf-8-sig')
    ex_groups = list(mem.groupby('muniyear').size().sort_values(ascending=False).index[:3])
    plot_examples(meta, mem, cent, ex_groups)

    log('\n[起動時の確認]')
    mfolds, S_m, seeds13, summ_m, pa_m = m_diff_from_stage13(meta, sp, D['M_diff'], log)
    s13.prechecks(KEY, D['GC_date'], meta, sp, log)  # 確認2・4・3(GC_date)
    ps = s13.perm_sets(KEY, meta, P)

    res = {}
    for c in ['GC_date', 'GC_index']:
        log(f'\n[{c}]')
        res[c] = run_m(c, D[c], meta, sp, ps, seeds, P, log)

    summary = pd.DataFrame(summ_m + [r for c in res for r in res[c]['summary']])
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds = pd.concat([f for c in res for f in res[c]['folds']])
    folds.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([r for c in res for r in res[c]['seed']]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([s for c in res for s in res[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                 index=False, encoding='utf-8-sig')
    pd.concat([pa_m] + [p for c in res for p in res[c]['pair']]).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'),
                                                                        index=False, encoding='utf-8-sig')

    # 対ごとの差(同じ分割・同じ乱数)
    fM = mfolds[mfolds.seed.isin(seeds)][['seed', 'repeat', 'fold', 'auc', 'bacc']]
    pdiff = pd.concat([s7.paired(folds[folds.config == c][['seed', 'repeat', 'fold', 'auc', 'bacc']], fM,
                                 f'{c} − M_diff') for c in ['GC_date', 'GC_index']])
    pdiff.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    pds = s7.paired_summary(pdiff)
    pds['positive_ci'] = pds.rep_q025 > 0
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差(同じ分割・同じ乱数)]')
    for _, r in pds.iterrows():
        log(f'  {r.comparison} {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{r.rep_q025:+.3f}, '
            f'{r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}')

    # W の差のブートストラップ
    k_m = [seeds13.index(s) for s in seeds]
    boot, pt = w_diff_bootstrap(res['GC_date']['S'], S_m[k_m], meta, sp, args.n_boot)
    boot.to_csv(os.path.join(OUT_DIR, 'w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    bq = boot.W_diff.quantile([.025, .975]).values
    log(f'\n[W の差 GC_date − M_diff(市町村のブートストラップ {args.n_boot}回)] 点推定 {pt["a"] - pt["b"]:+.3f}'
        f'(GC_date {pt["a"]:.3f}、M_diff {pt["b"]:.3f})、95%区間 [{bq[0]:+.3f}, {bq[1]:+.3f}]、'
        f'正の割合 {(boot.W_diff > 0).mean():.3f}')

    plot_overview(summary)
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'n_pairs_per_repeat', 'alpha_at_max']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.config == 'GC_date') & (summary.metric == 'all_auc')].iloc[0]
    d = pds[(pds.comparison == 'GC_date − M_diff') & (pds.metric == 'auc')].iloc[0]
    c1, c2 = bool(d.rep_q025 > 0), bool(r['mean'] > r.null_q95)
    log(f'\n[採用の判定] 1. GC_date − M_diff の繰り返し平均の2.5%点 {d.rep_q025:+.3f} > 0: {c1}  '
        f'2. GC_date の全体 AUC {r["mean"]:.3f} > 95%点 {r.null_q95:.3f}: {c2} → '
        + ('採用する(GC_date を主の入力にする)' if c1 and c2 else '採用しない(主の入力は1次差分のまま)'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
