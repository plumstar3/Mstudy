"""
段階34: D-012(段階33)の検定1〜3と追加の記述を、fixed 版のデータで2つのラベルについて回す(ラベルを変えた探索)(EXP-043)。
設定の記録は docs/decisions.md の D-014(データとラベル以外は D-012 と同じ)。

データ: data/processed/soilmoisture_timeseries_fixed.csv(D-013)。515 は除く。収穫日より後の観測は使わない。
  窓は field_year_labeled.csv の flower_date を0日目とした 0〜+59 日で、60日すべてに観測がある件だけ(src/windows.py と同じ規則)。
ラベル: label_muniyear(主の関心)と label_muniyear_median(比較用)。同じ設定で別々に回す。
rain_days: 段階29の定義(開花起点の日 0〜59 で Δθ > 0。日0は開花日の前日の θ、無ければ雨の日でない)を fixed 版から作る。
検定1〜3・開花日の幅14日以下の版・年ごと・全 p 値の表は D-012 と同じ。統計の関数は段階33(run_stage33.py)のものを使う。
全 p 値の表: ラベルごとの Holm(D-012 と同じ)と、参考に2つのラベルを合わせた Holm。

試運転(--part trial、置換200回)はコードの確認だけ。結果は outputs/stage34/trial/ に出し、レポートに載せない。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage34\\run_stage34.py --part trial
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage34\\run_stage34.py --part main
出力: rocket/outputs/stage34/(ラベルごとのサブフォルダと、compare.csv・all_pvalues_both.csv・run_log.txt)、
  入力の保存: rocket/data/processed/stage34/
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
from scipy import stats

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage33'))
import run_stage33 as s33  # noqa: E402

s32 = s33.s32
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
FIXED = os.path.join(PROC, 'soilmoisture_timeseries_fixed.csv')
LAB = os.path.join(PROC, 'field_year_labeled.csv')
NEW_DATA = os.path.join(PROC, 'stage34')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage34')
S33_OUT = os.path.join(ROCKET_DIR, 'outputs', 'stage33')
LABELS = ['label_muniyear', 'label_muniyear_median']
T_LEN, K_MAIN, K_ALL, SPREAD_MAX = s33.T_LEN, s33.K_MAIN, s33.K_ALL, s33.SPREAD_MAX
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データ
def load_fixed():
    ts = pd.read_csv(FIXED, encoding='utf-8-sig')
    ts = ts[ts.field_id != 515].copy()
    ts['date'] = pd.to_datetime(ts.date)
    lab = pd.read_csv(LAB, encoding='utf-8-sig', dtype={'muniCd': str})
    for c in ['flower_date', 'seed_date', 'harvest_date']:
        lab[c] = pd.to_datetime(lab[c])
    return ts, lab


def build_inputs(label, ts, lab, log):
    """開花起点の窓 Xf (n, 60)、rain_days (n, 60)、meta、収穫日で打ち切った日単位の表(暦の窓用)。"""
    hv = lab[['field_id', 'year', 'harvest_date']]
    tsh = ts.merge(hv, on=['field_id', 'year'])
    n0 = len(tsh)
    tsh = tsh[tsh.date <= tsh.harvest_date].drop(columns=['harvest_date'])
    series = {k: g.set_index('date').VWC for k, g in tsh.groupby(['field_id', 'year'])}
    cand = lab[lab[label].notna()].copy()
    X, R, keep, excl = [], [], [], []
    for i, r in cand.iterrows():
        s = series.get((r.field_id, r.year))
        need = pd.date_range(r.flower_date, periods=T_LEN)
        if s is None:
            excl.append((r.field_id, r.year, '系列なし'))
            continue
        w = s.reindex(need)
        if w.isna().any():
            excl.append((r.field_id, r.year, f'窓の中に観測の無い日 {int(w.isna().sum())}'))
            continue
        x = w.to_numpy(float)
        prev = s.get(r.flower_date - pd.Timedelta(days=1), np.nan)
        th = np.concatenate([[prev], x])
        dl = th[1:] - th[:-1]
        X.append(x)
        R.append(np.nan_to_num(dl, nan=0.0) > 0)
        keep.append(i)
    meta = cand.loc[keep].reset_index(drop=True)
    meta['muniyear'] = meta.muniCd + '_' + meta.year.astype(str)
    X, R = np.array(X), np.array(R)
    log(f'[{label}] 候補 {len(cand)} 件 → 窓の60日がそろう {len(meta)} 件(多収 {int((meta[label] == 1).sum())} / '
        f'低収 {int((meta[label] == 0).sum())})、外した {len(excl)} 件。収穫日より後で除いた行 {n0 - len(tsh)}')
    os.makedirs(NEW_DATA, exist_ok=True)
    np.save(os.path.join(NEW_DATA, f'{label}_X.npy'), X[:, None, :])
    np.save(os.path.join(NEW_DATA, f'{label}_rain_days.npy'), R)
    meta.to_csv(os.path.join(NEW_DATA, f'{label}_meta.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(excl, columns=['field_id', 'year', 'reason']).to_csv(
        os.path.join(NEW_DATA, f'{label}_excluded.csv'), index=False, encoding='utf-8-sig')
    return X, R, meta, tsh


# ---------------------------------------------------------------- 図(段階33の図を、集団数の表記だけ変えたもの)
def plot_diff_curves(curves, G, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    names = {'D': 'D = Δθ', 'N': 'N = min(Δθ, 0)', 'P': 'P = max(Δθ, 0)(対照)', 'level': 'θ の水準(参考)'}
    for ax, nm in zip(axes.ravel(), ['D', 'N', 'P', 'level']):
        Dg = curves[('cal', nm)]
        x = np.arange(1, 60) if nm != 'level' else np.arange(0, 60)
        Gt = np.sum(~np.isnan(Dg), 0)
        m = np.nanmean(Dg, 0)
        h = stats.t.ppf(0.975, np.maximum(Gt - 1, 1)) * np.nanstd(Dg, 0, ddof=1) / np.sqrt(Gt)
        ax.fill_between(x, m - h, m + h, color='#2a78d6', alpha=0.15, lw=0)
        ax.plot(x, m, color='#2a78d6', lw=2, label='集団の差の平均(多収 − 低収)')
        ax.axhline(0, color=s33.MUTED, lw=0.8)
        ax.set_title(names[nm], loc='left', fontsize=10.5, color=s33.INK)
        ax.set_xlabel('集団の開花日の中央値からの日数(暦の日付)', color=s33.INK)
        s33.style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.005, f'帯は集団間の95%区間(各集団1票、{G}集団)', fontsize=8, color=s33.MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out_dir, 'fig_diff_curves.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 1つのラベル(段階33の run と同じ手順)
def run_label(label, P, out_dir, ts, lab, log):
    s33.LABEL = label                      # group_table・plot_panels が参照する(段階33のファイルは変更しない)
    Xf, rain_f, meta, tsh = build_inputs(label, ts, lab, log)
    y = meta[label].astype(int).to_numpy()
    groups = meta.muniyear.astype(str).to_numpy()
    gt = s33.group_table(meta)
    gt['one_vs_one'] = (gt.n_high == 1) & (gt.n_low == 1)
    gt.to_csv(os.path.join(out_dir, 'groups.csv'), index=False, encoding='utf-8-sig')
    used = gt[gt.used]
    G = len(used)
    sub = gt[gt.in_spread14]
    n11 = int((used.one_vs_one).sum())
    log(f'[{label}] 集団 {len(gt)}、使う集団 {G} 集団・{used.n.sum()} 件。除く(片方のクラスだけ): '
        + ', '.join(f'{r.group}({r.n}件)' for r in gt[~gt.used].itertuples()))
    log(f'  多収1・低収1 の集団 {n11}(検定2 では置換で E_g が変わらず、実質 {G - n11} 集団の検定)')
    log(f'  開花日の幅 ≤ {SPREAD_MAX}日の版: {len(sub)} 集団・{sub.n.sum()} 件。除く: '
        + ', '.join(f'{r.group}(幅{r.flower_spread}日、{r.n}件)' for r in gt[gt.used & ~gt.in_spread14].itertuples()))
    log('  年ごと(使う集団): ' + ', '.join(f'{yr} {int((used.year == yr).sum())}集団・{int(used[used.year == yr].n.sum())}件'
                                       for yr in sorted(used.year.unique())))

    Xc, Rc, outside = s33.calendar(meta, gt, tsh, rain_f)
    uidx = np.isin(groups, used.group)
    log(f'  暦の窓: 欠測 {int(np.isnan(Xc).sum())} 圃場日。rain_days の範囲外(雨の日に当たらないとする): '
        f'{int(outside[uidx].sum())} / {int(uidx.sum()) * T_LEN} 圃場日(使う集団)')
    dl = np.diff(Xc, axis=1) > 0
    inside = ~outside[:, 1:] & ~outside[:, :-1]
    log(f'  確認: 開花窓の中で、置き直した rain_days と暦の Δθ > 0 の一致率 {(Rc[:, 1:][inside] == dl[inside]).mean():.4f}')

    Ys = s32.make_perms(y, groups, P)
    ok_perm = all(np.array_equal(pd.Series(Ys[i]).groupby(groups).sum().values,
                                 pd.Series(y).groupby(groups).sum().values) for i in range(0, P + 1, max(1, P // 50)))
    log(f'  置換 {P} 回、集団ごとのクラス数が保たれる: {ok_perm}')

    gidx_all = {g: np.flatnonzero(groups == g) for g in gt.group}
    gn_used = list(used.group)
    w_used = used.n.to_numpy()
    gev = s33.group_events(Rc, gt, gidx_all, meta)
    pv = []

    def add_p(test, variant, weight, subset, year, stat, p, note=''):
        pv.append({'label': label, 'test': test, 'variant': variant, 'weight': weight, 'subset': subset, 'year': year,
                   'stat': stat, 'p': p, 'note': note})

    sG = f'{G}集団'
    # ---------------- 検定1
    log(f'\n  ==== 検定1 ====')
    Sc, Sf = s33.diff_series(Xc), s33.diff_series(Xf)
    curves, tc, summ, pts, cls, contrib = {}, {}, [], [], [], []
    gi_used = [gidx_all[g] for g in gn_used]
    for align, S in [('cal', Sc), ('flw', Sf)]:
        for nm in ['D', 'N', 'P', 'level']:
            off = 0 if nm == 'level' else 1
            Dg = s33.group_diffs(S[nm], Ys, gi_used)
            curves[(align, nm)] = Dg[:, 0, :]
            for wname, w in [('集団1票', None), ('件数', w_used)]:
                Tm, m, Gt = s33.t_across(Dg, w)
                r, cl, cmn, mx, padj = s33.maxT(Tm, off)
                r.update({'align': align, 'series': nm, 'weight': wname})
                summ.append(r)
                for a, b, sg, ms in cl:
                    cls.append({'align': align, 'series': nm, 'weight': wname, 'start': a, 'end': b, 'sign': sg,
                                'mass': ms, 'p': s32.p_upper(cmn, ms)})
                if wname == '集団1票':
                    tc[(align, nm)] = (Tm[0], r['thr95'])
                    if P >= 1000:
                        np.save(os.path.join(out_dir, f'null_maxT_{align}_{nm}.npy'), mx)
                    for j in range(Tm.shape[1]):
                        pts.append({'align': align, 'series': nm, 'day': j + off, 'G': int(Gt[0, j]),
                                    'mean_diff': m[0, j], 't': Tm[0, j], 'p_maxT_adj': padj[j]})
                    contrib.append(pd.Series(Gt[0], index=np.arange(off, off + len(Gt[0])), name=f'{align}_{nm}'))
                al = '暦' if align == 'cal' else '開花起点'
                add_p('検定1', f'{al}・{nm}・max-T', wname, sG, '全年', r['T_max'], r['p_maxT'],
                      f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}')
                add_p('検定1', f'{al}・{nm}・クラスタ質量', wname, sG, '全年', r['max_cluster_mass'], r['p_cluster'],
                      f'クラスタ {r["n_clusters"]} 個')
                log(f'  {align} {nm} [{wname}]: T_max = {r["T_max"]:.3f}(日 {r["arg"]}、t = {r["t_at_arg"]:+.2f})、'
                    f'95%点 {r["thr95"]:.3f}、p_maxT = {r["p_maxT"]:.4f}、クラスタ {r["n_clusters"]} 個'
                    f'(最大質量 {r["max_cluster_mass"]:.2f}、p = {r["p_cluster"]:.4f})')
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, 'test1_maxT.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(pts).to_csv(os.path.join(out_dir, 'test1_points.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(cls, columns=['align', 'series', 'weight', 'start', 'end', 'sign', 'mass', 'p']).to_csv(
        os.path.join(out_dir, 'clusters.csv'), index=False, encoding='utf-8-sig')
    pd.concat(contrib, axis=1).rename_axis('day').to_csv(os.path.join(out_dir, 'contrib_groups.csv'), encoding='utf-8-sig')
    dc = []
    for (align, nm), C in curves.items():
        off = 0 if nm == 'level' else 1
        d_ = pd.DataFrame(C, columns=[f'd{j + off}' for j in range(C.shape[1])])
        d_.insert(0, 'series', nm)
        d_.insert(0, 'align', align)
        d_.insert(0, 'group', gn_used)
        dc.append(d_)
    pd.concat(dc).to_csv(os.path.join(out_dir, 'diff_curves.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 検定2
    log(f'\n  ==== 検定2 ====')
    t2, Dms = [], {}
    for nm in ['D', 'N', 'P']:
        Dms[('a', nm)] = s33.euclid_within_complete(Sc[nm], gi_used)
        F, nkeep, ntot = s32.minirocket_features(Sf[nm], 0)
        Dms[('b', nm)] = s32.euclid(F)
    for (inp, nm), Dm in Dms.items():
        for wname, w in [('集団1票', None), ('件数', w_used)]:
            E, Eg = s33.energy_V(Dm, Ys, gi_used, w)
            nl = E[1:]
            r = {'input': inp, 'series': nm, 'weight': wname, 'E': E[0], 'null_mean': nl.mean(), 'null_sd': nl.std(ddof=1),
                 'z': (E[0] - nl.mean()) / nl.std(ddof=1), 'p': s32.p_upper(nl, E[0]), 'n_groups': G,
                 'n_groups_not_1v1': G - n11}
            t2.append(r)
            if wname == '集団1票' and P >= 1000:
                np.save(os.path.join(out_dir, f'null_E_{inp}_{nm}.npy'), nl)
            lab_ = '(a) 暦の系列' if inp == 'a' else '(b) MiniRocket'
            add_p('検定2', f'{lab_}・{nm}', wname, sG, '全年', E[0], r['p'], f'z = {r["z"]:.2f}')
            log(f'  {lab_} {nm} [{wname}]: E = {E[0]:.5f}、z = {r["z"]:.2f}、p = {r["p"]:.4f}')
    pd.DataFrame(t2).to_csv(os.path.join(out_dir, 'test2_energy.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 検定3
    log(f'\n  ==== 検定3 ====')
    evc = pd.DataFrame([{'group': g, 'n_group_rain_days': int(gev[g][0].sum()), 'n_events': len(gev[g][1])}
                        for g in gn_used])
    log(f'  集団の雨の日: 平均 {evc.n_group_rain_days.mean():.1f}({evc.n_group_rain_days.min()}〜'
        f'{evc.n_group_rain_days.max()})。イベント: 1集団 {evc.n_events.min()}〜{evc.n_events.max()}'
        f'(平均 {evc.n_events.mean():.2f})、イベントの無い集団 {int((evc.n_events == 0).sum())}')
    t3, recs = [], []
    state = {'gv': None}

    def run3(gnames, label_subset, year='全年'):
        gidx = [gidx_all[g] for g in gnames]
        w = gt.set_index('group').loc[gnames, 'n'].to_numpy()
        res = []
        for k, rate in [(K_MAIN, False), (K_MAIN, True)] + [(k, False) for k in K_ALL if k != K_MAIN]:
            V, rec = s33.test3_stat(Xc, Ys, gidx, gnames, gev, k, rate)
            if label_subset == sG and year == '全年':
                for r_ in rec:
                    r_['rate'] = rate
                recs.extend(rec)
            for wname, ww in [('集団1票', None), ('件数', w)]:
                st = s33.combine(V, ww)
                gobs = V[:, 0]
                ok = ~np.isnan(gobs)
                npos, nval = int((gobs[ok] > 0).sum()), int(ok.sum())
                psign = stats.binomtest(npos, nval, 0.5).pvalue if nval else np.nan
                vname = f'k = {k}' + ('・減少率' if rate else '・減少量')
                r = {'subset': label_subset, 'year': year, 'k': k, 'version': '減少率' if rate else '減少量',
                     'weight': wname, 'stat': st[0], 'null_sd': np.nanstd(st[1:], ddof=1), 'p': s33.p_two(st),
                     'n_groups': nval, 'n_high_more': npos, 'p_sign': psign}
                res.append(r)
                if (label_subset, year, k, rate, wname) == (sG, '全年', K_MAIN, False, '集団1票'):
                    state['gv'] = gobs
                    if P >= 1000:
                        np.save(os.path.join(out_dir, 'null_test3_k3.npy'), st[1:])
                add_p('検定3', vname, wname, label_subset, year, st[0], r['p'],
                      f'多収が多く減る集団 {npos}/{nval}(符号検定 p = {psign:.3f})')
                log(f'  [{label_subset}・{year}] {vname} [{wname}]: 統計量 {st[0]:+.5f}、p = {r["p"]:.4f}、'
                    f'多収が多く減る集団 {npos}/{nval}(符号検定 p = {psign:.3f})')
        return res

    t3 += run3(gn_used, sG)
    pd.DataFrame(recs).to_csv(os.path.join(out_dir, 'test3_events.csv'), index=False, encoding='utf-8-sig')
    evc['value_k3'] = state['gv']
    evc.to_csv(os.path.join(out_dir, 'test3_groups.csv'), index=False, encoding='utf-8-sig')
    hz = pd.DataFrame([r for r in recs if r['k'] == K_MAIN and not r['rate']])
    log(f'  k = 3 のイベント {len(hz)} 個、h の内訳 ' + str(hz[hz.used].h.value_counts().sort_index().to_dict()))

    # ---------------- 開花日の幅 ≤ 14日
    log(f'\n  ==== 開花日の幅 ≤ {SPREAD_MAX}日(記述) ====')
    gn_s = list(sub.group)
    gi_s = [gidx_all[g] for g in gn_s]
    w_s = sub.n.to_numpy()
    sub_rows = []
    sS = f'幅≤{SPREAD_MAX}日({len(gn_s)}集団)'
    for nm in ['D', 'N']:
        Dg = s33.group_diffs(Sc[nm], Ys, gi_s)
        for wname, w in [('集団1票', None), ('件数', w_s)]:
            Tm, m, Gt = s33.t_across(Dg, w)
            r, *_ = s33.maxT(Tm, 1)
            r.update({'align': 'cal', 'series': nm, 'weight': wname, 'subset': sS})
            sub_rows.append(r)
            add_p('検定1', f'暦・{nm}・max-T', wname, sS, '全年', r['T_max'], r['p_maxT'],
                  f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}')
            add_p('検定1', f'暦・{nm}・クラスタ質量', wname, sS, '全年', r['max_cluster_mass'], r['p_cluster'],
                  f'クラスタ {r["n_clusters"]} 個')
            log(f'  検定1 暦 {nm} [{wname}]: T_max = {r["T_max"]:.3f}(日 {r["arg"]}、t = {r["t_at_arg"]:+.2f})、'
                f'p_maxT = {r["p_maxT"]:.4f}、クラスタ {r["n_clusters"]} 個(p = {r["p_cluster"]:.4f})')
    pd.DataFrame(sub_rows).to_csv(os.path.join(out_dir, 'spread14_test1.csv'), index=False, encoding='utf-8-sig')
    t3 += run3(gn_s, sS)

    # ---------------- 年ごと
    log(f'\n  ==== 年ごと(記述) ====')
    yrows = []
    for yr in sorted(used.year.unique()):
        gy = list(used[used.year == yr].group)
        gi_y = [gidx_all[g] for g in gy]
        ny = int(used[used.year == yr].n.sum())
        for nm in ['D', 'N']:
            Dg = s33.group_diffs(Sc[nm], Ys, gi_y)
            Tm, m, Gt = s33.t_across(Dg)
            if len(gy) >= 3:
                r, *_ = s33.maxT(Tm, 1)
                yrows.append({'test': f'検定1 暦・{nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': r['T_max'],
                              'p': r['p_maxT'], 'note': f'日 {r["arg"]}、t = {r["t_at_arg"]:+.2f}'})
                add_p('検定1', f'暦・{nm}・max-T', '集団1票', sG, yr, r['T_max'], r['p_maxT'])
            else:
                yrows.append({'test': f'検定1 暦・{nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': np.nan,
                              'p': np.nan, 'note': '集団が2つで集団間の t を計算できない'})
        for (inp, nm), Dm in Dms.items():
            E, Eg = s33.energy_V(Dm, Ys, gi_y)
            nl = E[1:]
            sd = nl.std(ddof=1)
            z = (E[0] - nl.mean()) / sd if sd > 0 else np.nan
            yrows.append({'test': f'検定2 ({inp}) {nm}', 'year': yr, 'n_groups': len(gy), 'n': ny, 'stat': E[0],
                          'p': s32.p_upper(nl, E[0]), 'note': f'z = {z:.2f}'})
            add_p('検定2', f'{"(a) 暦の系列" if inp == "a" else "(b) MiniRocket"}・{nm}', '集団1票', sG, yr,
                  E[0], s32.p_upper(nl, E[0]), f'z = {z:.2f}')
        V, _ = s33.test3_stat(Xc, Ys, gi_y, gy, gev, K_MAIN, False)
        st = s33.combine(V)
        gobs = V[:, 0]
        ok = ~np.isnan(gobs)
        yrows.append({'test': '検定3 k = 3・減少量', 'year': yr, 'n_groups': int(ok.sum()), 'n': ny, 'stat': st[0],
                      'p': s33.p_two(st), 'note': f'多収が多く減る集団 {int((gobs[ok] > 0).sum())}/{int(ok.sum())}'})
        add_p('検定3', 'k = 3・減少量', '集団1票', sG, yr, st[0], s33.p_two(st))
    dfY = pd.DataFrame(yrows)
    dfY.to_csv(os.path.join(out_dir, 'by_year.csv'), index=False, encoding='utf-8-sig')
    for r in dfY.itertuples():
        log(f'  {r.test} {r.year}({r.n_groups}集団・{r.n}件): 統計量 {r.stat:.4f}、p = {r.p:.4f}  {r.note}')
    pd.DataFrame(t3).to_csv(os.path.join(out_dir, 'test3_summary.csv'), index=False, encoding='utf-8-sig')

    dfP = pd.DataFrame(pv)
    dfP['p_holm_label'] = s32.holm(dfP.p.to_numpy())
    dfP.to_csv(os.path.join(out_dir, 'all_pvalues.csv'), index=False, encoding='utf-8-sig')
    log(f'\n  [{label}] 全 p 値 {len(dfP)} 個。補正前 p < 0.05: {int((dfP.p < 0.05).sum())} 個、'
        f'ラベル内の Holm 後 p < 0.05: {int((dfP.p_holm_label < 0.05).sum())} 個')
    for r in dfP.sort_values('p').head(10).itertuples():
        log(f'    {r.test} {r.variant} [{r.weight}・{r.subset}・{r.year}]: p = {r.p:.4f}、Holm 後 {r.p_holm_label:.4f}  {r.note}')

    plot_diff_curves(curves, G, out_dir)
    s33.plot_t(tc, out_dir)
    s33.plot_panels(Xc, meta, gt, gev, out_dir)
    s33.plot_test3(state['gv'], gn_used, out_dir)
    return dfP, pd.DataFrame(summ), pd.DataFrame(t2), pd.DataFrame(t3)


# ---------------------------------------------------------------- 2つのラベルと段階33の比較表
def compare(res, out_dir):
    rows = []
    s33_t1 = pd.read_csv(os.path.join(S33_OUT, 'test1_maxT.csv'))
    s33_t2 = pd.read_csv(os.path.join(S33_OUT, 'test2_energy.csv'))
    s33_t3 = pd.read_csv(os.path.join(S33_OUT, 'test3_summary.csv'))

    def g1(df, nm, align='cal', w='集団1票'):
        r = df[(df['align'] == align) & (df['series'] == nm) & (df['weight'] == w)].iloc[0]
        return f'|t| {r.T_max:.2f}(日 {int(r.arg)})、p {r.p_maxT:.3f}'

    def g2(df, inp, nm, w='集団1票'):
        r = df[(df.input == inp) & (df.series == nm) & (df.weight == w)].iloc[0]
        return f'z {r.z:.2f}、p {r.p:.3f}'

    def g3(df, k, ver, w='集団1票', first=True):
        d = df[(df.k == k) & (df.version == ver) & (df.weight == w) & (df.year == '全年')]
        r = d.iloc[0]
        return f'{r.stat:+.4f}、p {r.p:.3f}、{int(r.n_high_more)}/{int(r.n_groups)}'

    items = [('検定1 暦・D', lambda t1, t2, t3: g1(t1, 'D')), ('検定1 暦・N', lambda t1, t2, t3: g1(t1, 'N')),
             ('検定1 暦・P(対照)', lambda t1, t2, t3: g1(t1, 'P')), ('検定1 暦・水準(参考)', lambda t1, t2, t3: g1(t1, 'level')),
             ('検定1 開花起点・D(参考)', lambda t1, t2, t3: g1(t1, 'D', 'flw')),
             ('検定1 開花起点・N(参考)', lambda t1, t2, t3: g1(t1, 'N', 'flw'))]
    for inp in ['a', 'b']:
        for nm in ['D', 'N', 'P']:
            items.append((f'検定2 ({inp}) {nm}', (lambda i, n: lambda t1, t2, t3: g2(t2, i, n))(inp, nm)))
    for k, ver in [(3, '減少量'), (3, '減少率'), (1, '減少量'), (2, '減少量'), (4, '減少量'), (5, '減少量')]:
        items.append((f'検定3 k = {k}・{ver}', (lambda kk, vv: lambda t1, t2, t3: g3(t3, kk, vv))(k, ver)))
    for name, f in items:
        row = {'item': name}
        for lb in LABELS:
            _, t1, t2, t3 = res[lb]
            row[lb] = f(t1, t2, t3)
        row['stage33_v2_label_muniyear_median(参考、データが違う)'] = f(s33_t1, s33_t2, s33_t3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'compare.csv'), index=False, encoding='utf-8-sig')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['trial', 'main'], required=True)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    P = 200 if a.part == 'trial' else 10000
    out_dir = os.path.join(OUT_MAIN, 'trial') if a.part == 'trial' else OUT_MAIN
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'段階34 --part {a.part}({time.strftime("%Y-%m-%d %H:%M:%S")})')
    if a.part == 'trial':
        log('試運転: コードが動くかの確認だけ(D-014)。この結果で設定を変えない。p 値はレポートに載せない')
    ts, lab = load_fixed()
    log(f'データ: {FIXED}(515 を除いて {len(ts)} 行)')
    res = {}
    for lb in LABELS:
        od = os.path.join(out_dir, lb)
        os.makedirs(od, exist_ok=True)
        log(f'\n================ {lb} ================')
        res[lb] = run_label(lb, P, od, ts, lab, log)
    both = pd.concat([res[lb][0] for lb in LABELS], ignore_index=True)
    both['p_holm_both'] = s32.holm(both.p.to_numpy())
    both.to_csv(os.path.join(out_dir, 'all_pvalues_both.csv'), index=False, encoding='utf-8-sig')
    log(f'\n2つのラベルを合わせた p {len(both)} 個。補正前 p < 0.05: {int((both.p < 0.05).sum())} 個、'
        f'合わせた Holm 後 p < 0.05: {int((both.p_holm_both < 0.05).sum())} 個')
    cmp_ = compare(res, out_dir)
    log('\n比較表(集団1票):')
    for r in cmp_.itertuples(index=False):
        log('  ' + ' | '.join(str(v) for v in r))
    log(f'\n合計 {time.time() - t0:.1f}秒')
    logf.close()


if __name__ == '__main__':
    main()
