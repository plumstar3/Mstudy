"""
段階35: 段階32(D-011)の検定A〜D を、label_year・fixed 版のデータに当てる(ラベルを変えた探索)(EXP-044)。設定の記録は D-015。

データ: data/processed/soilmoisture_timeseries_fixed.csv(D-013)。515 を除く。473(2018)は含める(D-015、M02・M03 のセンサー列の共有。label_year では重複しない)。
  窓は flower_date を0日目とした 0〜+59 日、60日そろう件だけ(収穫日より後は使わない)。雨の日は段階29の定義(段階34と同じ作り方)。
ラベル: label_year(同じ年の中の上位1/3・下位1/3)。
検定A〜D・補助・対照・年ごとは D-011 と同じ。統計の関数は段階32(run_stage32.py)のものを使う。
並べ替え(同じ統計量に2通り、各10,000回、default_rng([0, i])):
  year    年の中(主)
  muniyr  市町村×年の中(地域の構成を固定。片方のクラスしかない集団は動かない)
主の族 {A-D, A-N, B} の Holm は並べ替えごとに参考として出す(判定はしない)。max-T・クラスタ質量・擬似 F にも z を付ける。

試運転(--part trial、置換200回)はコードの確認だけ。結果は outputs/stage35/trial/ に出し、レポートに載せない。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage35\\run_stage35.py --part trial
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage35\\run_stage35.py --part main
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
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage32'))
import run_stage32 as s32  # noqa: E402

PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
FIXED = os.path.join(PROC, 'soilmoisture_timeseries_fixed.csv')
LAB = os.path.join(PROC, 'field_year_labeled.csv')
NEW_DATA = os.path.join(PROC, 'stage35')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage35')
LABEL = 'label_year'
T_LEN = 60
EXCLUDE = {(515, None): '使わない圃場(従来どおり、D-013)'}
# field_id 473(2018)は含める(D-015)。M02・M03 は元Excel で同じセンサー列を共有する(docs/data_issues.md の3)が、
# label_year では 472 にラベルが無く重複しない。段階32〜34と同じく除外しない。
SCHEMES = [('year', '年の中'), ('muniyr', '市町村×年の中')]
INK, MUTED, GRID = s32.INK, s32.MUTED, s32.GRID
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データ
def build_inputs(log):
    ts = pd.read_csv(FIXED, encoding='utf-8-sig')
    ts = ts[ts.field_id != 515].copy()
    ts['date'] = pd.to_datetime(ts.date)
    lab = pd.read_csv(LAB, encoding='utf-8-sig', dtype={'muniCd': str})
    for c in ['flower_date', 'seed_date', 'harvest_date']:
        lab[c] = pd.to_datetime(lab[c])
    tsh = ts.merge(lab[['field_id', 'year', 'harvest_date']], on=['field_id', 'year'])
    tsh = tsh[tsh.date <= tsh.harvest_date]
    series = {k: g.set_index('date').VWC for k, g in tsh.groupby(['field_id', 'year'])}
    cand = lab[lab[LABEL].notna()].copy()
    X, R, keep, excl = [], [], [], []
    for i, r in cand.iterrows():
        if r.field_id == 515:
            excl.append((r.field_id, r.year, r[LABEL], EXCLUDE[(515, None)]))
            continue
        s = series.get((r.field_id, r.year))
        need = pd.date_range(r.flower_date, periods=T_LEN)
        if s is None:
            excl.append((r.field_id, r.year, r[LABEL], '系列なし'))
            continue
        w = s.reindex(need)
        if w.isna().any():
            excl.append((r.field_id, r.year, r[LABEL], f'窓の中に観測の無い日 {int(w.isna().sum())}(60日そろわない)'))
            continue
        if (r.field_id, r.year) in EXCLUDE:
            x = w.to_numpy(float)
            excl.append((r.field_id, r.year, r[LABEL], EXCLUDE[(r.field_id, r.year)]
                         + f'。窓の1次差分の最大絶対値 {np.abs(np.diff(x)).max():.1e}'))
            continue
        x = w.to_numpy(float)
        prev = s.get(r.flower_date - pd.Timedelta(days=1), np.nan)
        th = np.concatenate([[prev], x])
        X.append(x)
        R.append(np.nan_to_num(th[1:] - th[:-1], nan=0.0) > 0)
        keep.append(i)
    meta = cand.loc[keep].reset_index(drop=True)
    meta['muniyear'] = meta.muniCd + '_' + meta.year.astype(str)
    X, R = np.array(X), np.array(R)
    exc = pd.DataFrame(excl, columns=['field_id', 'year', LABEL, 'reason'])
    os.makedirs(NEW_DATA, exist_ok=True)
    np.save(os.path.join(NEW_DATA, f'{LABEL}_X.npy'), X[:, None, :])
    np.save(os.path.join(NEW_DATA, f'{LABEL}_rain_days.npy'), R)
    meta.to_csv(os.path.join(NEW_DATA, f'{LABEL}_meta.csv'), index=False, encoding='utf-8-sig')
    exc.to_csv(os.path.join(NEW_DATA, f'{LABEL}_excluded.csv'), index=False, encoding='utf-8-sig')
    in473 = bool(((meta.field_id == 473) & (meta.year == 2018)).any())
    in472 = bool(((meta.field_id == 472) & (meta.year == 2018)).any())
    log(f'候補 {len(cand)} 件 → {len(meta)} 件(多収 {int((meta[LABEL] == 1).sum())} / 低収 {int((meta[LABEL] == 0).sum())})。'
        f'外した {len(exc)} 件。473・2018 を含む: {in473}、472・2018 を含む: {in472}')
    return X, R, meta


# ---------------------------------------------------------------- 統計(段階32の関数 + z)
def zval(null, obs):
    sd = np.nanstd(null, ddof=1)
    return (obs - np.nanmean(null)) / sd if sd > 0 else np.nan


def pw(C, Ys, offset):
    """max-T とクラスタ質量(段階32と同じ)。z も返す。"""
    T = s32.welch_t(C, Ys)
    t_obs, Tn = T[0], T[1:]
    mx = np.nanmax(np.abs(Tn), axis=1)
    Tobs = np.nanmax(np.abs(t_obs))
    arg = int(np.nanargmax(np.abs(t_obs)))
    cmn = s32.max_cluster(np.nan_to_num(Tn), s32.CLUSTER_THR)
    cl = s32.clusters_obs(np.nan_to_num(t_obs), s32.CLUSTER_THR, offset)
    mm = max([m for *_, m in cl], default=0.0)
    return {'T_max': Tobs, 'arg': arg + offset, 't_at_arg': t_obs[arg], 'thr95': np.quantile(mx, 0.95),
            'p_maxT': s32.p_upper(mx, Tobs), 'z_maxT': zval(mx, Tobs), 'n_clusters': len(cl), 'max_cluster_mass': mm,
            'p_cluster': s32.p_upper(cmn, mm) if cl else 1.0, 'z_cluster': zval(cmn, mm)}, t_obs, mx, cl


# ---------------------------------------------------------------- 図
def plot_t(tcur, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    for ax, (name, (x, t, thr, xl)) in zip(axes.ravel(), tcur.items()):
        ax.plot(x, t, color=INK, lw=2, marker='o' if len(x) < 15 else None, ms=4, label='Welch の t(多収 − 低収)')
        for s in (1, -1):
            ax.axhline(s * thr['year'], color='#2a78d6', ls='--', lw=1.2, label='max-T の95%点(年の中)' if s == 1 else None)
            ax.axhline(s * thr['muniyr'], color='#eb6834', ls='-.', lw=1.2,
                       label='max-T の95%点(市町村×年の中)' if s == 1 else None)
            ax.axhline(s * s32.CLUSTER_THR, color=MUTED, ls=':', lw=1, label='±2.0' if s == 1 else None)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(name, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel(xl, color=INK)
        ax.set_ylabel('t', color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_t_curves.png'), dpi=150)
    plt.close(fig)


def plot_energy(E0, nulls, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for i, (sc, scn) in enumerate(SCHEMES):
        for j, nm in enumerate(['D', 'N', 'P']):
            ax = axes[i, j]
            nl = nulls[(sc, nm)]
            ax.hist(nl, bins=50, color=MUTED, alpha=0.6, label='帰無分布')
            ax.axvline(E0[nm], color='#2a78d6', lw=2, label='観測値')
            role = '主' if nm != 'P' else '対照'
            ax.set_title(f'A-{nm}({role})・{scn}: z = {zval(nl, E0[nm]):.2f}、p = {s32.p_upper(nl, E0[nm]):.4f}',
                         loc='left', fontsize=9, color=INK)
            ax.set_xlabel('energy 統計量 E', color=INK)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_energy_null.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本体
def run(P, out_dir, log):
    t0 = time.time()
    X, rain, meta = build_inputs(log)
    y = meta[LABEL].astype(int).to_numpy()
    years = meta.year.to_numpy()
    muniyr = meta.muniyear.astype(str).to_numpy()
    log('年ごと: ' + ', '.join(f'{yr} {int((years == yr).sum())}件(多収 {int(y[years == yr].sum())})' for yr in sorted(set(years))))
    gt = pd.DataFrame({'g': muniyr, 'y': y}).groupby('g').y.agg(['size', 'sum'])
    both = gt[(gt['sum'] > 0) & (gt['sum'] < gt['size'])]
    log(f'市町村×年 {len(gt)} 集団。両クラスがある {len(both)} 集団・{int(both["size"].sum())} 件'
        f'(市町村×年の中の並べ替えで動く件)、片方のクラスだけ {len(gt) - len(both)} 集団・{int(len(y) - both["size"].sum())} 件')
    S = s32.series(X)
    log(f'確認: rain_days(日 1 … 59)= (Δθ > 0): {np.array_equal(rain[:, 1:], S["D"] > 0)}')

    Ys = {'year': s32.make_perms(y, years.astype(str), P), 'muniyr': s32.make_perms(y, muniyr, P)}
    for sc, scn in SCHEMES:
        okc = all(np.array_equal(pd.Series(Ys[sc][i]).groupby(years if sc == 'year' else muniyr).sum().values,
                                 pd.Series(y).groupby(years if sc == 'year' else muniyr).sum().values)
                  for i in range(0, P + 1, max(1, P // 50)))
        moved = (Ys[sc][1:] != y).any(0)
        log(f'置換({scn}): 層ごとのクラス数が保たれる {okc}、一度でもラベルが変わった件 {int(moved.sum())} / {len(y)}')

    raw, nrm, cnt = s32.response_curves(X, rain)
    cnt.insert(0, 'field_id', meta.field_id.values)
    cnt.insert(1, 'year', years)
    cnt.insert(2, 'label', y)
    cnt.to_csv(os.path.join(out_dir, 'event_counts.csv'), index=False, encoding='utf-8-sig')
    rc = []
    for kind, C in [('raw', raw), ('norm', nrm)]:
        d = pd.DataFrame(C, columns=[f'lag{l}' for l in range(s32.LAGS + 1)])
        d.insert(0, 'kind', kind)
        d.insert(0, 'label', y)
        d.insert(0, 'year', years)
        d.insert(0, 'field_id', meta.field_id.values)
        rc.append(d)
    pd.concat(rc).to_csv(os.path.join(out_dir, 'response_curves.csv'), index=False, encoding='utf-8-sig')
    log(f'雨のイベント: 1圃場 {cnt.n_events.min()}〜{cnt.n_events.max()} 回(平均 {cnt.n_events.mean():.2f})、'
        f'イベントの無い圃場 {int((cnt.n_events == 0).sum())}。正規化版のイベント {cnt.n_events_norm.sum()} / {cnt.n_events.sum()}')
    log('ラグごとに値のある圃場(生): ' + ', '.join(f'l{l} {int(np.sum(~np.isnan(raw[:, l])))}' for l in range(s32.LAGS + 1)))

    pv = []

    def add(test, variant, scope, year, sc, stat, z, p, note=''):
        pv.append({'test': test, 'variant': variant, 'scope': scope, 'year': year, 'scheme': dict(SCHEMES)[sc],
                   'stat': stat, 'z': z, 'p': p, 'note': note})

    # ---------------- 検定A
    log('\n==== 検定A: MiniRocket 特徴の energy ====')
    rowsA, Dms, E0, nullsA = [], {}, {}, {}
    for nm in ['D', 'N', 'P']:
        for k in s32.SEEDS:
            F, nkeep, ntot = s32.minirocket_features(S[nm], k)
            Dm = s32.euclid(F)
            if k == s32.MAIN_SEED:
                Dms[nm] = Dm
            role = '主' if nm in ('D', 'N') and k == 0 else ('対照' if nm == 'P' and k == 0 else '参考(乱数)')
            r = {'series': nm, 'seed': k, 'role': role, 'n_features_used': nkeep}
            for sc, scn in SCHEMES:
                E = s32.energy(Dm, Ys[sc])
                Ew, used = s32.energy_within(Dm, Ys[sc], muniyr)
                r.update({'E': E[0], f'z_{sc}': zval(E[1:], E[0]), f'p_{sc}': s32.p_upper(E[1:], E[0]),
                          'E_W': Ew[0], f'zW_{sc}': zval(Ew[1:], Ew[0]), f'pW_{sc}': s32.p_upper(Ew[1:], Ew[0]),
                          'n_strata_W': len(used), 'n_cases_W': int(sum(c for _, c in used))})
                add('検定A', f'A-{nm}・乱数{k}', '全件', '全年', sc, E[0], r[f'z_{sc}'], r[f'p_{sc}'], role)
                add('検定A', f'A-{nm}・乱数{k}・E_W', '同じ市町村×年のペア', '全年', sc, Ew[0], r[f'zW_{sc}'], r[f'pW_{sc}'],
                    f'{len(used)}集団・{r["n_cases_W"]}件')
                if k == 0:
                    E0[nm] = E[0]
                    nullsA[(sc, nm)] = E[1:]
                    if P >= 1000:
                        np.save(os.path.join(out_dir, f'null_A_{nm}_{sc}.npy'), E[1:])
            rowsA.append(r)
            log(f'  A-{nm} 乱数{k}: E = {r["E"]:.4f}、年の中 z {r["z_year"]:.2f}・p {r["p_year"]:.4f} / '
                f'市町村×年の中 z {r["z_muniyr"]:.2f}・p {r["p_muniyr"]:.4f}。E_W({r["n_strata_W"]}集団・{r["n_cases_W"]}件)'
                f' 年 z {r["zW_year"]:.2f}・p {r["pW_year"]:.4f} / 市町村×年 z {r["zW_muniyr"]:.2f}・p {r["pW_muniyr"]:.4f}')
    pd.DataFrame(rowsA).to_csv(os.path.join(out_dir, 'testA_energy.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 検定B・C
    log('\n==== 検定B・C: max-T とクラスタ質量 ====')
    specs = [('B_raw', raw[:, 1:], 1, 'lag', '検定B(主): 雨後の応答曲線(生)', '雨のイベント後の日数 l'),
             ('B_norm', nrm[:, 1:], 1, 'lag', '検定B(補助): 振幅で正規化', '雨のイベント後の日数 l'),
             ('C_D', S['D'], 1, 'day', '検定C(対照): D(開花起点)', '開花後の日数 t'),
             ('C_N', S['N'], 1, 'day', '検定C(対照): N(開花起点)', '開花後の日数 t')]
    rowsS, pts, cls, tcur = [], [], [], {}
    for name, C, off, unit, ttl, xl in specs:
        thr = {}
        for sc, scn in SCHEMES:
            r, t, mx, cl = pw(C, Ys[sc], off)
            thr[sc] = r['thr95']
            r.update({'test': name, 'scheme': scn})
            rowsS.append(r)
            if P >= 1000:
                np.save(os.path.join(out_dir, f'null_maxT_{name}_{sc}.npy'), mx)
            for a, b, sg, m in cl:
                cls.append({'test': name, 'scheme': scn, 'start': a, 'end': b, 'sign': sg, 'mass': m})
            mainB = '主' if name == 'B_raw' else ('補助' if name == 'B_norm' else '対照')
            add('検定B' if name.startswith('B') else '検定C', f'{name}・max-T', '全件', '全年', sc, r['T_max'], r['z_maxT'],
                r['p_maxT'], f'{mainB}、{unit} {r["arg"]}、t = {r["t_at_arg"]:+.2f}')
            add('検定B' if name.startswith('B') else '検定C', f'{name}・クラスタ質量', '全件', '全年', sc, r['max_cluster_mass'],
                r['z_cluster'], r['p_cluster'], f'クラスタ {r["n_clusters"]} 個')
            log(f'  {name}[{scn}]: T_max = {r["T_max"]:.3f}({unit} {r["arg"]}、t = {r["t_at_arg"]:+.2f})、'
                f'95%点 {r["thr95"]:.3f}、p = {r["p_maxT"]:.4f}、z = {r["z_maxT"]:.2f}、クラスタ {r["n_clusters"]} 個'
                f'(最大質量 {r["max_cluster_mass"]:.2f}、p = {r["p_cluster"]:.4f})')
        y0 = y
        for j in range(C.shape[1]):
            c = C[:, j]
            h, l = c[(y0 == 1) & ~np.isnan(c)], c[(y0 == 0) & ~np.isnan(c)]
            pts.append({'test': name, unit: j + off, 'n_high': len(h), 'n_low': len(l), 'mean_high': h.mean(),
                        'mean_low': l.mean(), 'diff': h.mean() - l.mean(), 't': t[j]})
        tcur[ttl] = (np.arange(off, off + C.shape[1]), t, thr, xl)
        if name.startswith('B'):
            log('    t: ' + ', '.join(f'l{j + off} {v:+.2f}' for j, v in enumerate(t)))
    pd.DataFrame(rowsS).to_csv(os.path.join(out_dir, 'maxT_summary.csv'), index=False, encoding='utf-8-sig')
    dpts = pd.DataFrame(pts)
    dpts[dpts.test.str.startswith('B')].dropna(axis=1, how='all').to_csv(
        os.path.join(out_dir, 'testB_points.csv'), index=False, encoding='utf-8-sig')
    dpts[dpts.test.str.startswith('C')].dropna(axis=1, how='all').to_csv(
        os.path.join(out_dir, 'testC_points.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(cls, columns=['test', 'scheme', 'start', 'end', 'sign', 'mass']).to_csv(
        os.path.join(out_dir, 'clusters.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 検定D
    log('\n==== 検定D: DTW の PERMANOVA ====')
    from sktime.dists_kernels.dtw import DtwDist
    X3 = S['D'][:, None, :]
    Dd = np.asarray(DtwDist(window=s32.DTW_WINDOW / X3.shape[2]).transform(X3, X3), dtype=float)
    Dd = (Dd + Dd.T) / 2
    np.fill_diagonal(Dd, 0.0)
    rowsD = []
    for sc, scn in SCHEMES:
        Fv = s32.permanova_F(Dd ** 2, Ys[sc])
        r = {'scheme': scn, 'pseudo_F': Fv[0], 'z': zval(Fv[1:], Fv[0]), 'p': s32.p_upper(Fv[1:], Fv[0])}
        rowsD.append(r)
        add('検定D', 'DTW PERMANOVA(D)', '全件', '全年', sc, Fv[0], r['z'], r['p'], '補助')
        log(f'  [{scn}] 擬似 F = {Fv[0]:.3f}、z = {r["z"]:.2f}、p = {r["p"]:.4f}')
    pd.DataFrame(rowsD).to_csv(os.path.join(out_dir, 'testD_permanova.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 主の族の Holm(参考)
    dfP = pd.DataFrame(pv)
    hrows = []
    for sc, scn in SCHEMES:
        fam = [('A-D', dfP[(dfP.variant == 'A-D・乱数0') & (dfP.scheme == scn)].p.iloc[0]),
               ('A-N', dfP[(dfP.variant == 'A-N・乱数0') & (dfP.scheme == scn)].p.iloc[0]),
               ('B', dfP[(dfP.variant == 'B_raw・max-T') & (dfP.scheme == scn)].p.iloc[0])]
        adj = s32.holm([p for _, p in fam])
        for (f, p), a in zip(fam, adj):
            hrows.append({'scheme': scn, 'test': f, 'p': p, 'p_holm_family': a})
            log(f'  主の族の Holm(参考)[{scn}] {f}: p = {p:.4f} → {a:.4f}')
    pd.DataFrame(hrows).to_csv(os.path.join(out_dir, 'holm.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 年ごと
    log('\n==== 年ごと(記述) ====')
    yrows = []
    for yr in sorted(set(years)):
        idx = np.flatnonzero(years == yr)
        nb = int(both.loc[[g for g in both.index if g.endswith(str(yr))], 'size'].sum())
        for nm in ['D', 'N', 'P']:
            for sc, scn in SCHEMES:
                Ey = s32.energy(Dms[nm][np.ix_(idx, idx)], Ys[sc][:, idx])
                z, p = zval(Ey[1:], Ey[0]), s32.p_upper(Ey[1:], Ey[0])
                yrows.append({'test': f'A-{nm}', 'year': yr, 'n': len(idx), 'n_high': int(y[idx].sum()),
                              'n_moving_muniyr': nb, 'scheme': scn, 'stat': Ey[0], 'z': z, 'p': p, 'note': ''})
                add('検定A', f'A-{nm}・乱数0', '年ごと', yr, sc, Ey[0], z, p)
        for sc, scn in SCHEMES:
            r, t, mx, cl = pw(raw[idx, 1:], Ys[sc][:, idx], 1)
            yrows.append({'test': 'B', 'year': yr, 'n': len(idx), 'n_high': int(y[idx].sum()), 'n_moving_muniyr': nb,
                          'scheme': scn, 'stat': r['T_max'], 'z': r['z_maxT'], 'p': r['p_maxT'],
                          'note': 't(l1…l7) = ' + ' '.join(f'{v:+.2f}' for v in t)})
            add('検定B', 'B_raw・max-T', '年ごと', yr, sc, r['T_max'], r['z_maxT'], r['p_maxT'])
    dfY = pd.DataFrame(yrows)
    dfY.to_csv(os.path.join(out_dir, 'by_year.csv'), index=False, encoding='utf-8-sig')
    for r in dfY.itertuples():
        log(f'  {r.test} {r.year}(n = {r.n}、多収 {r.n_high}、市町村×年で動く {r.n_moving_muniyr})[{r.scheme}]: '
            f'統計量 {r.stat:.4f}、z = {r.z:.2f}、p = {r.p:.4f}  {r.note}')

    dfP = pd.DataFrame(pv)
    dfP.to_csv(os.path.join(out_dir, 'all_pvalues.csv'), index=False, encoding='utf-8-sig')
    for sc, scn in SCHEMES:
        d = dfP[dfP.scheme == scn]
        log(f'\n全 p 値[{scn}]: {len(d)} 個、補正前 p < 0.05: {int((d.p < 0.05).sum())} 個')
        for r in d.sort_values('p').head(8).itertuples():
            log(f'  {r.test} {r.variant} [{r.scope}・{r.year}]: z = {r.z:.2f}、p = {r.p:.4f}  {r.note}')

    s32.plot_curves(raw, nrm, y, out_dir)
    plot_t(tcur, out_dir)
    plot_energy(E0, nullsA, out_dir)
    log(f'\n合計 {time.time() - t0:.1f}秒')


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

    log(f'段階35 --part {a.part}({time.strftime("%Y-%m-%d %H:%M:%S")})')
    if a.part == 'trial':
        log('試運転: コードが動くかの確認だけ(D-015)。この結果で設定を変えない。p 値はレポートに載せない')
    run(P, out_dir, log)
    logf.close()


if __name__ == '__main__':
    main()
