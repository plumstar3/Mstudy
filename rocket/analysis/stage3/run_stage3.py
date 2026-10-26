"""
段階3: 系列の揃え方の起点(dap / 開花起点 / 積算GDD)と窓の範囲を、ROCKET を使う前に決めるための集計。

  0. 事前確認: 品種ごとの件数と開花日の有無、加古川(28210)の播種日・品種
  1. 開花日の推定精度の比較(flower_date がある66件、市町村の Leave-One-Group-Out)
  2. 起点ごとの窓の候補と残存件数(label_muniyear 付き94件)
  3. 起点ごとの推奨窓
  4. 図(ヒートマップ、VWC 系列の例)

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage3\\run_stage3.py
出力: rocket/outputs/stage3/
"""
import os
import sqlite3
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
ROOT_DIR = os.path.dirname(ROCKET_DIR)
sys.path.insert(0, ROCKET_DIR)
from src.phenology import (GddDaylength, GddThreshold, MeanDap, SeedDoyRegression,  # noqa: E402
                           daylength)

PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DB = os.path.join(ROOT_DIR, 'data', 'processed', 'FieldData_fieldid.db')
GDD = os.path.join(ROOT_DIR, 'outputs', 'gdd', 'gdd_daily.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage3')

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
C_HIGH, C_LOW = '#e34948', '#2a78d6'  # 既存の図と同じく 赤=多収, 青=低収
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

DAP_STARTS, DAP_ENDS = [20, 30, 40, 50], [89, 99, 109, 119, 129]
FL_STARTS, FL_ENDS = [-30, -20, -10, 0], [49, 59, 69, 79]
GDD_SHIFTS = [-200, -100, 0, 100, 200]
MIN_N = 80
KAKOGAWA = '28210'


def save(df, name):
    df.to_csv(os.path.join(OUT_DIR, name), index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------- データ
def load():
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig',
                      dtype={'muniCd': str})
    d = lab[lab['label_muniyear'].notna()].copy()
    d['y'] = d['label_muniyear'].astype(int)
    d['muniyear'] = d['muniCd'] + '_' + d['year'].astype(str)
    d['key'] = list(zip(d['field_id'], d['year']))

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    q = pd.read_sql('SELECT field_id, year, flower_date FROM Questionaire WHERE field_id IS NOT NULL', conn)
    conn.close()
    q['field_id'] = pd.to_numeric(q['field_id'], errors='coerce')
    q = q.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})
    q = q.drop_duplicates(['field_id', 'year'])  # build_soilmoisture_dataset.py と同じく先頭を採る
    d = d.merge(q, on=['field_id', 'year'], how='left')

    d['seed_date'] = pd.to_datetime(d['seed_date'])
    d['harvest_date'] = pd.to_datetime(d['harvest_date'])
    d['seed_doy'] = d['seed_date'].dt.dayofyear
    d['flower_date'] = pd.to_datetime(d['flower_date'], errors='coerce')
    d['flower_dap_obs'] = (d['flower_date'] - d['seed_date']).dt.days
    d['has_flower'] = d['flower_date'].notna()

    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig',
                     parse_dates=['date'])
    ts = ts.merge(d[['field_id', 'year']], on=['field_id', 'year'])
    sh = ts.groupby(['field_id', 'year'])['dap'].agg(start_dap='min', end_dap='max').reset_index()
    d = d.merge(sh, on=['field_id', 'year'])
    d['harvest_dap'] = (d['harvest_date'] - d['seed_date']).dt.days

    g = pd.read_csv(GDD, encoding='utf-8-sig', parse_dates=['date'],
                    usecols=['field_id', 'year', 'date', 'seed_date', '累積GDD'])
    g = g.merge(d[['field_id', 'year']], on=['field_id', 'year'])
    g['dap'] = (g['date'] - pd.to_datetime(g['seed_date'])).dt.days
    cum = {}
    for k, s in g.sort_values('dap').groupby(['field_id', 'year']):
        assert s['dap'].iloc[0] == 0 and (np.diff(s['dap']) == 1).all(), f'{k} の GDD が連続していない'
        cum[k] = s['累積GDD'].values
    return d.reset_index(drop=True), ts, cum


# ---------------------------------------------------------------- 0. 事前確認
def precheck(d):
    b = d.assign(breed=d['breed'].fillna('(欠損)'))
    t = b.groupby('breed').agg(n=('y', 'size'), n_high=('y', 'sum'), n_flower=('has_flower', 'sum'),
                               n_kakogawa=('muniCd', lambda s: (s == KAKOGAWA).sum()))
    t['n_flower_missing'] = t['n'] - t['n_flower']
    save(t.sort_values('n', ascending=False).reset_index(), 'precheck_breed.csv')

    b['region'] = np.where(b['muniCd'] == KAKOGAWA, '加古川', 'その他')
    rows = []
    for (reg, yr), s in b.groupby(['region', 'year']):
        rows.append({'region': reg, 'year': yr, 'n': len(s),
                     'seed_doy_min': s.seed_doy.min(), 'seed_doy_median': s.seed_doy.median(),
                     'seed_doy_max': s.seed_doy.max(), 'lat_median': s.lat.median(),
                     'breeds': ', '.join(f'{k}:{v}' for k, v in s.breed.value_counts().items())})
    save(pd.DataFrame(rows), 'precheck_kakogawa.csv')
    return t


# ---------------------------------------------------------------- 1. 開花日の推定
def estimators():
    return [MeanDap(), SeedDoyRegression(), GddThreshold(),
            GddDaylength('seed'), GddDaylength('mean'),
            # 日長の定義を変えたときの感度(参考)
            GddDaylength('mean', p=0.0), GddDaylength('mean', p=6.0)]


def est_name(e):
    return e.name + ('' if getattr(e, 'p', 0.8333) == 0.8333 else f'_p{e.p:g}')


def flowering_cv(d, cum):
    obs = d[d['has_flower']].copy()
    obs['flower_dap'] = obs['flower_dap_obs']
    rows = []
    for e in estimators():
        for g in obs['muniCd'].unique():
            tr, te = obs[obs.muniCd != g], obs[obs.muniCd == g]
            e.fit(tr, cum_gdd=cum)
            pred = e.predict(te, cum_gdd=cum)
            for k, m, a, p in zip(te['key'], te['muniyear'], te['flower_dap'], pred):
                rows.append({'method': est_name(e), 'field_id': k[0], 'year': k[1], 'muniyear': m,
                             'flower_dap': a, 'pred_dap': p, 'error': p - a})
    err = pd.DataFrame(rows)
    save(err, 'flowering_cv_errors.csv')
    s = err.groupby('method', sort=False)['error'].agg(
        MAE=lambda x: x.abs().mean(), bias='mean', SD='std', max_abs=lambda x: x.abs().max(),
        n_nan=lambda x: x.isna().sum()).reset_index()
    save(s, 'flowering_cv_summary.csv')

    # 参考: dap 軸と GDD 軸での開花日のばらつき
    fg = np.array([cum[k][int(x)] for k, x in zip(obs['key'], obs['flower_dap'])])
    daily = np.mean(fg / (obs['flower_dap'].values + 1))  # 播種〜開花の1日あたり平均GDD
    spread = {'n': len(obs), 'flower_dap_sd': obs['flower_dap'].std(),
              'flower_gdd_mean': fg.mean(), 'flower_gdd_sd': fg.std(ddof=1),
              'daily_gdd_sow_to_flower': daily, 'flower_gdd_sd_in_days': fg.std(ddof=1) / daily,
              'n_groups': obs['muniCd'].nunique()}
    save(pd.DataFrame([spread]), 'flowering_spread.csv')
    return s, spread, obs


def estimate_missing(d, cum, obs, method):
    e = next(x for x in estimators() if est_name(x) == method)
    e.fit(obs, cum_gdd=cum)
    miss = d[~d['has_flower']]
    pred = e.predict(miss, cum_gdd=cum)
    d['flower_dap'] = d['flower_dap_obs']
    d.loc[~d['has_flower'], 'flower_dap'] = pred
    d['flower_estimated'] = ~d['has_flower']
    out = d.loc[~d['has_flower'], ['field_id', 'year', 'muniCd', 'y', 'seed_date', 'flower_dap']].copy()
    out['flower_date_est'] = out['seed_date'] + pd.to_timedelta(out['flower_dap'], unit='D')
    out['method'] = method
    params = {k: v for k, v in vars(e).items() if isinstance(v, (int, float, np.floating))}
    save(out, 'flower_estimates_missing28.csv')
    return d, params


# ---------------------------------------------------------------- 2. 窓の集計
def count_windows(d, start_col, end_col, windows, axis, extra=None):
    """windows: [(s, e, 表示用の情報 dict)]。start_col ≤ s かつ end_col ≥ e の件を数える。"""
    base_groups = d.groupby('muniyear').size()
    rows = []
    for s, e, info in windows:
        m = (d[start_col] <= s) & (d[end_col] >= e)
        sub = d[m]
        g = sub.groupby('muniyear')['y'].agg(['sum', 'size'])
        bal = int((g['sum'] * 2 == g['size']).sum())
        r = {'axis': axis, 'start': s, 'end': e, **info,
             'n': int(m.sum()), 'n_high': int(sub.y.sum()), 'n_low': int((1 - sub.y).sum()),
             'n_regions': sub.muniCd.nunique(),
             'n_2016': int((sub.year == 2016).sum()), 'n_2017': int((sub.year == 2017).sum()),
             'n_2018': int((sub.year == 2018).sum()),
             'groups_balanced': bal, 'groups_unbalanced': len(g) - bal,
             'groups_dropped': len(base_groups) - len(g),
             'n_common_caseA': int((m & d['caseA']).sum())}
        if extra is not None:
            r.update(extra(sub))
        rows.append(r)
    return pd.DataFrame(rows)


def windows_all(d, cum):
    d['caseA'] = (d.start_dap <= 50) & (d.end_dap >= 99)

    # 起点1: dap
    w1 = [(s, e, {'length_days': e - s + 1}) for s in DAP_STARTS for e in DAP_ENDS]
    r1 = count_windows(d, 'start_dap', 'end_dap', w1, 'dap')

    # 起点2: 開花起点(実測優先、無ければ推定)
    d['start_fl'] = d['start_dap'] - d['flower_dap']
    d['end_fl'] = d['end_dap'] - d['flower_dap']
    w2 = [(s, e, {'length_days': e - s + 1}) for s in FL_STARTS for e in FL_ENDS]
    r2 = count_windows(d, 'start_fl', 'end_fl', w2, 'flower',
                       extra=lambda sub: {'n_estimated': int(sub.flower_estimated.sum()),
                                          'n_estimated_kakogawa': int((sub.flower_estimated & (sub.muniCd == KAKOGAWA)).sum())})
    obs = d[d['has_flower']]
    r2o = count_windows(obs, 'start_fl', 'end_fl', w2, 'flower_observed_only')
    r2 = r2.merge(r2o[['start', 'end', 'n', 'n_high', 'n_low', 'n_regions']].rename(
        columns=lambda c: c if c in ('start', 'end') else f'obs66_{c}'), on=['start', 'end'])

    # 起点3: 積算GDD(収穫日より後の観測は GDD が無いので使わない)
    d['start_gdd'] = [cum[k][min(s, len(cum[k]) - 1)] for k, s in zip(d['key'], d['start_dap'])]
    d['end_gdd'] = [cum[k][min(e, len(cum[k]) - 1)] for k, e in zip(d['key'], d['end_dap'])]
    a = d[d['caseA']]
    g50 = np.median([cum[k][50] for k in a['key']])
    g99 = np.median([cum[k][99] for k in a['key']])
    daily = np.mean([(cum[k][99] - cum[k][50]) / 49 for k in a['key']])  # 案A 区間の1日あたり平均GDD

    def gdd_days(sub, s, e):
        # 窓を通過するのにかかった日数(覆う件の中央値)
        days = [first_reach_dap(cum[k], e) - first_reach_dap(cum[k], s) + 1 for k in sub['key']]
        return float(np.median(days)) if days else np.nan

    w3 = []
    for ds in GDD_SHIFTS:
        for de in GDD_SHIFTS:
            s, e = g50 + ds, g99 + de
            w3.append((round(s, 1), round(e, 1), {'shift_start': ds, 'shift_end': de,
                                                 'length_gdd': round(e - s, 1),
                                                 'length_days': round((e - s) / daily, 1)}))
    r3 = count_windows(d, 'start_gdd', 'end_gdd', w3, 'gdd')
    r3['median_days_covered'] = [
        gdd_days(d[(d.start_gdd <= s) & (d.end_gdd >= e)], s, e) for s, e in zip(r3.start, r3.end)]
    gref = {'gdd_dap50_median': g50, 'gdd_dap99_median': g99, 'daily_gdd_caseA': daily}

    for r, n in [(r1, 'windows_dap.csv'), (r2, 'windows_flower.csv'), (r3, 'windows_gdd.csv')]:
        save(r, n)
    save(d.drop(columns=['key']), 'cases.csv')
    return r1, r2, r3, gref


def first_reach_dap(cum, th):
    idx = np.flatnonzero(cum >= th)
    return idx[0] if len(idx) else np.nan


def recommend(r):
    c = r[r['n'] >= MIN_N]
    if c.empty:
        return None
    return c.sort_values(['length_days', 'groups_balanced', 'n'], ascending=False).iloc[0]


# ---------------------------------------------------------------- 4. 図
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def heatmap(ax, r, starts, ends, slabel, elabel, title, fmt_s=str, fmt_e=str):
    M = r.pivot(index='start', columns='end', values='n').loc[starts, ends].values
    ax.imshow(M, cmap='Blues', vmin=0, vmax=94, aspect='auto')
    for i in range(len(starts)):
        for j in range(len(ends)):
            v = M[i, j]
            ax.text(j, i, int(v), ha='center', va='center', fontsize=10,
                    color='white' if v > 55 else INK)
            if v >= MIN_N:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.5))
    ax.set_xticks(range(len(ends)), [fmt_e(e) for e in ends])
    ax.set_yticks(range(len(starts)), [fmt_s(s) for s in starts])
    ax.set_xlabel(elabel, color=INK)
    ax.set_ylabel(slabel, color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_heatmaps(r1, r2, r3):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    heatmap(axs[0, 0], r1, DAP_STARTS, DAP_ENDS, '始まり(dap)', '終わり(dap)', '起点1: dap(播種日起点)')
    r2a = r2.copy()
    heatmap(axs[0, 1], r2a, FL_STARTS, FL_ENDS, '始まり(開花からの日数)', '終わり(開花からの日数)',
            '起点2: 開花起点(実測+推定、94件中)', fmt_e=lambda e: f'+{e}')
    r2o = r2.assign(n=r2['obs66_n'])
    heatmap(axs[1, 0], r2o, FL_STARTS, FL_ENDS, '始まり(開花からの日数)', '終わり(開花からの日数)',
            '起点2: 開花起点(実測の66件だけ)', fmt_e=lambda e: f'+{e}')
    r3h = r3.assign(start=r3['shift_start'], end=r3['shift_end'])
    heatmap(axs[1, 1], r3h, GDD_SHIFTS, GDD_SHIFTS, '始まり(dap50 中央値からのずれ, ℃日)',
            '終わり(dap99 中央値からのずれ, ℃日)', '起点3: 積算GDD',
            fmt_s=lambda s: f'{s:+d}', fmt_e=lambda e: f'{e:+d}')
    fig.suptitle(f'起点ごとの窓の残存件数(label_muniyear 付き94件中、枠 = {MIN_N}件以上)',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'heatmap_windows.png'), dpi=150)
    plt.close(fig)


def plot_examples(d, ts):
    # 実測の開花日がある集団2つ(件数の多い順)と、推定に頼る加古川の1集団
    obs_groups = (d[d.has_flower].groupby('muniyear').filter(lambda s: s.y.nunique() == 2)
                  .groupby('muniyear').size().sort_values(ascending=False))
    picks = list(obs_groups.index[:2])
    kako = d[(d.muniCd == KAKOGAWA)].groupby('muniyear').size().sort_values(ascending=False)
    picks.append(kako.index[0])

    fig, axs = plt.subplots(len(picks), 2, figsize=(13, 3.4 * len(picks)), sharey='row')
    for i, g in enumerate(picks):
        cases = d[d.muniyear == g]
        for _, c in cases.iterrows():
            s = ts[(ts.field_id == c.field_id) & (ts.year == c.year)]
            col = C_HIGH if c.y == 1 else C_LOW
            ls = ':' if c.flower_estimated else '--'
            axs[i, 0].plot(s.dap, s.VWC, color=col, lw=1.4)
            axs[i, 0].axvline(c.flower_dap, color=col, ls=ls, lw=1)
            axs[i, 1].plot(s.dap - c.flower_dap, s.VWC, color=col, lw=1.4)
        axs[i, 1].axvline(0, color=INK, ls=ls, lw=1)
        est = '推定' if cases.flower_estimated.all() else '実測'
        axs[i, 0].set_title(f'{g}(多収{cases.y.sum()}・低収{(1 - cases.y).sum()}、開花日は{est})',
                            fontsize=10, color=INK, loc='left')
        axs[i, 0].set_ylabel('VWC', color=INK)
        for ax in axs[i]:
            style(ax)
            ax.grid(axis='y', color=GRID, lw=0.8)
    axs[-1, 0].set_xlabel('播種後日数(dap)。縦線 = 各圃場の開花日', color=INK)
    axs[-1, 1].set_xlabel('開花からの日数(0 = 開花日)', color=INK)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C_HIGH, lw=2), Line2D([], [], color=C_LOW, lw=2),
                Line2D([], [], color=INK, ls='--'), Line2D([], [], color=INK, ls=':')],
               ['多収', '低収', '開花日(実測)', '開花日(推定)'],
               loc='upper right', ncol=4, frameon=False, fontsize=10)
    fig.suptitle('VWC 系列の例: dap 軸(左)と開花起点の軸(右)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'vwc_examples_dap_vs_flower.png'), dpi=150)
    plt.close(fig)
    return picks


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d, ts, cum = load()
    print(f'対象: {len(d)}件(開花日あり {int(d.has_flower.sum())})')

    print('\n[0] 品種')
    print(precheck(d).sort_values('n', ascending=False).to_string())
    print(pd.read_csv(os.path.join(OUT_DIR, 'precheck_kakogawa.csv'), encoding='utf-8-sig').to_string())

    print('\n[1] 開花日の推定(LOGO by muniCd)')
    s, spread, obs = flowering_cv(d, cum)
    print(s.round(2).to_string())
    print({k: round(v, 2) for k, v in spread.items()})
    main_methods = s[~s.method.str.contains('_p')]
    best = main_methods.sort_values('MAE').iloc[0]['method']
    d, params = estimate_missing(d, cum, obs, best)
    print('推奨:', best, params)
    print(d.loc[~d.has_flower, 'flower_dap'].describe().round(1).to_dict())

    print('\n[2] 窓')
    r1, r2, r3, gref = windows_all(d, cum)
    print({k: round(v, 1) for k, v in gref.items()})
    cols = ['start', 'end', 'length_days', 'n', 'n_high', 'n_low', 'n_regions', 'n_2016', 'n_2017', 'n_2018',
            'groups_balanced', 'groups_unbalanced', 'groups_dropped', 'n_common_caseA']
    print(r1[cols].to_string())
    print(r2[cols + ['n_estimated', 'n_estimated_kakogawa', 'obs66_n']].to_string())
    print(r3[['shift_start', 'shift_end', 'length_gdd', 'median_days_covered'] + cols].to_string())

    print('\n[3] 推奨')
    rec = []
    for name, r in [('dap', r1), ('flower', r2), ('gdd', r3)]:
        b = recommend(r)
        print(name, None if b is None else b[[c for c in cols if c in b.index]].to_dict())
        if b is not None:
            rec.append(b.to_dict())
    save(pd.DataFrame(rec), 'recommended_windows.csv')

    plot_heatmaps(r1, r2, r3)
    print('\n[4] 例に使った集団:', plot_examples(d, ts))
    print(f'出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
