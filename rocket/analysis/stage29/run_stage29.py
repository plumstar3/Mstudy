"""
段階29: 減る動き N を3つの成分に分ける(EXP-038)。

段階27・28で全体 AUC が偶然水準を上回った N(min(Δθ, 0))を、性質の違う成分に分ける。
  R       雨の日の位置 1[Δθ(t) > 0](0/1)
  N_dry   無降雨日の減少: N(t)。ただし雨の直後の日は0(蒸発散に近い)
  N_post  雨の直後の減少: N(t)。ただし雨の直後以外の日は0(排水に近い)
雨の日: Δθ(t) > 0(灌水を含む)。雨の直後: 雨の日 s の翌日から k 日目まで(s+1 … s+k)のうち、雨の日でない日。
雨の日が続く場合は最後の雨の日から数える(上の定義でそうなる)。主の設定は k = 3。
窓の最初の日(t = 1〜k)は、開花日より前の観測(soilmoisture_timeseries.csv)で雨の日を判定する(t = 1−k … 0 の Δθ)。
観測が無い日は「雨の日なし」として扱う。N_dry + N_post = N。
判定(D-007、事前に固定): R・N_dry・N_post の全体 AUC を、置換(200回、市町村×年の中、置換 i は [0, i])の帰無分布と比べ、
3つの p に Holm 補正をかけて Holm 後 p < 0.05 を「上回った」とする。補正前の p も併記する。
記述: W・B、bacc、alpha の張り付き、差の区間(N_dry − N、N_post − N、R − N、N_dry − N_post、市町村のブートストラップ2000回、
段階27と同じリサンプル)、k = 2・5 の感度分析(置換なし)。
基準の N は data/processed/stage27/ のものを使い、段階27の保存済みの結果(partial_N.pkl)を使う(乱数0の先頭10 fold を計算し直して確認)。
モデル・分割は段階27と同じ(MiniRocket(10000、乱数0〜9)→ StandardScaler(with_mean=False)→ RidgeClassifierCV、
make_splits(y, muniCd, 5, 20, 0))。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage29\\run_stage29.py --part check
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage29\\run_stage29.py --part main
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage29\\run_stage29.py --part sens
  試運転: ... --part main --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage29/、新しい入力は rocket/data/processed/stage29/。入力ごとの途中結果を partial_<入力>.pkl に保存する。
"""
import argparse
import os
import pickle
import sqlite3
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
for st in ['stage13', 'stage27']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', st))
import run_stage13 as s13  # noqa: E402
import run_stage27 as s27  # noqa: E402
from src.evaluation import make_splits, split_table  # noqa: E402
from src.windows import first_diff, load_timeseries  # noqa: E402

s6, s12 = s13.s6, s13.s12
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage29')
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage29')
D13 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13', 'label_muniyear_median')
D27 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage27')
S27 = os.path.join(ROCKET_DIR, 'outputs', 'stage27')
WDB = os.path.join(os.path.dirname(ROCKET_DIR), 'data', 'processed', 'weather_database_fieldid.db')
KEY = 'median'
K_MAIN, K_SENS = 3, [2, 5]
THR = 0.002   # 記述用のしきい値版(ユーザーの決定、2026-11-17。判定は Δθ > 0 のまま)
INPUTS = ['R', 'N_dry', 'N_post']
NAMES = {'N': '負の差分(基準)', 'R': '雨の日の位置(0/1)', 'N_dry': '無降雨日の減少', 'N_post': '雨の直後の減少'}
INK, MUTED = s6.INK, s6.MUTED
COL = {'N': '#eb6834', 'R': '#9b59b6', 'N_dry': '#c9802a', 'N_post': '#2a78d6', 'D': '#52514e'}
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------- データと成分
def load():
    X = np.load(D13 + '_X.npy')
    meta = pd.read_csv(D13 + '_meta.csv', encoding='utf-8-sig', dtype={'muniCd': str})
    for c in ['flower_date', 'seed_date', 'harvest_date']:
        meta[c] = pd.to_datetime(meta[c])
    return X, meta


def pre_window_diffs(meta, kmax):
    """t = 1−kmax … 0 の Δθ(開花日より前の観測から)。観測が無ければ NaN。戻り値 (n, kmax)。列 j は t = j − kmax + 1。"""
    cases = meta.copy()
    cases['harvest_dap'] = (cases.harvest_date - cases.seed_date).dt.days
    ts = load_timeseries(cases)
    out = np.full((len(meta), kmax), np.nan)
    for i, r in meta.iterrows():
        s = ts[(ts.field_id == r.field_id) & (ts.year == r.year)].set_index('dap').VWC
        f = int(r.flower_dap)
        for j in range(kmax):
            t = j - kmax + 1          # 窓の中の日付(0 = 開花日)
            a, b = f + t - 1, f + t   # Δθ(t) = θ(t) − θ(t−1)
            if a in s.index and b in s.index:
                out[i, j] = s[b] - s[a]
    return out


def components(X, pre, k, thr=0.0):
    """R・N_dry・N_post・N(各 (n, 1, 59))と、雨の日・雨の直後のフラグ(窓の中、(n, 59))。
    pre: t = 1−kmax … 0 の Δθ(NaN は雨の日なしとして扱う)。
    thr: 雨の日のしきい値。主は 0(Δθ > 0)。記述用に 0.002(Δθ ≧ 0.002。ごく小さな上昇は雨の日に数えない)。"""
    d = first_diff(X)[:, 0, :]                       # t = 1 … 59
    kmax = pre.shape[1]
    ext = np.concatenate([np.nan_to_num(pre, nan=0.0), d], axis=1)   # t = 1−kmax … 59
    rain_ext = ext > 0 if thr == 0 else ext >= thr
    rain = rain_ext[:, kmax:]
    post = np.zeros_like(rain)
    for t in range(d.shape[1]):                      # 窓の中の t+1
        te = t + kmax                                # ext での位置
        prev = rain_ext[:, max(0, te - k):te]        # 直前 k 日に雨の日があるか
        post[:, t] = prev.any(axis=1) & ~rain[:, t]
    N = np.minimum(d, 0.0)
    out = {'R': rain.astype(float), 'N_dry': np.where(post, 0.0, N), 'N_post': np.where(post, N, 0.0), 'N': N}
    return {k_: v[:, None, :] for k_, v in out.items()}, rain, post


# ---------------------------------------------------------------- 定義の確認
def part_check(X, meta, log):
    log('\n==== 定義の確認 ====')
    kmax = max([K_MAIN] + K_SENS)
    pre = pre_window_diffs(meta, kmax)
    d = first_diff(X)[:, 0, :]
    n_missing = np.isnan(pre).sum(axis=1)
    log(f'開花日より前の Δθ(t = {1 - kmax}〜0、{kmax}日分)が観測から取れない件: '
        f'全部ある {int((n_missing == 0).sum())} / 一部欠け {int(((n_missing > 0) & (n_missing < kmax)).sum())} / '
        f'全部無い {int((n_missing == kmax).sum())}(全 {len(meta)} 件)')
    for k in [K_MAIN] + K_SENS:
        miss_k = np.isnan(pre[:, kmax - k:]).any(axis=1)
        log(f'  k = {k}: 窓の最初の日の判定に使う t = {1 - k}〜0 のどれかが無い件 {int(miss_k.sum())}(「雨の日なし」として扱う)')
    comps, rain, post = components(X, pre, K_MAIN)
    ok = np.allclose(comps['N_dry'] + comps['N_post'], comps['N']) and \
        np.allclose(comps['N'], np.load(os.path.join(D27, 'label_muniyear_median_N_X.npy')))
    log(f'確認 N_dry + N_post = N(すべての t)、N が段階27の入力と同じ: {ok}')
    assert ok
    nr = rain.sum(axis=1)
    npst = post.sum(axis=1)
    log(f'雨の日(Δθ > 0)の数(件ごと、59日中): 平均 {nr.mean():.1f}、中央値 {np.median(nr):.0f}、'
        f'最小 {nr.min()}、最大 {nr.max()}、5・25・75・95%点 {np.percentile(nr, [5, 25, 75, 95]).round(1).tolist()}')
    log(f'雨の直後(k = 3)の日の数: 平均 {npst.mean():.1f}、中央値 {np.median(npst):.0f}、最大 {npst.max()}。'
        f'無降雨日(雨の日でも雨の直後でもない): 平均 {(59 - nr - npst).mean():.1f}')
    pos = d[d > 0]
    log(f'正の Δθ({len(pos)} 個)の大きさ: 5・25・50・75・95%点 {np.percentile(pos, [5, 25, 50, 75, 95]).round(4).tolist()}、'
        f'最小 {pos.min():.5f}')
    steps = np.unique(np.round(np.abs(np.diff(np.sort(np.unique(X.ravel())))), 6))
    log(f'  VWC の値の刻み(異なる値どうしの差の最小): {steps[steps > 0][:3].tolist()}')
    for thr in [0.0005, 0.001, 0.002, 0.005]:
        log(f'  正の Δθ のうち {thr} 未満: {(pos < thr).mean():.3f}({int((pos < thr).sum())} 個)')
    # 気象 DB との照合
    con = sqlite3.connect(WDB)
    rows = []
    for i, r in meta.iterrows():
        dates = pd.date_range(r.flower_date + pd.Timedelta(days=1), periods=59)
        q = pd.read_sql('SELECT date, APCPRA FROM weather_data WHERE field_id = ? AND date BETWEEN ? AND ?', con,
                        params=(int(r.field_id), (dates[0] - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                                dates[-1].strftime('%Y-%m-%d')))
        q['date'] = pd.to_datetime(q.date)
        pr = q.set_index('date').APCPRA.reindex(pd.date_range(dates[0] - pd.Timedelta(days=1), dates[-1])).astype(float)
        p_today = pr.values[1:]
        p_prev = pr.values[:-1]
        rows.append({'i': i, 'irrigated': r.irrigation != '無し', 'rain_vwc': rain[i], 'p_today': p_today,
                     'p_prev': p_prev})
    con.close()
    tab = []
    for grp, sel in [('全件', [True] * len(rows)), ('灌水なし', [not x['irrigated'] for x in rows]),
                     ('灌水あり', [x['irrigated'] for x in rows])]:
        rr = [x for x, s in zip(rows, sel) if s]
        rv = np.concatenate([x['rain_vwc'] for x in rr])
        pt = np.concatenate([x['p_today'] for x in rr])
        pp = np.concatenate([x['p_prev'] for x in rr])
        valid = ~np.isnan(pt) & ~np.isnan(pp)
        wet = (np.nan_to_num(pt) > 0) | (np.nan_to_num(pp) > 0)
        wet1 = (np.nan_to_num(pt) >= 1) | (np.nan_to_num(pp) >= 1)
        tab.append({'集団': grp, '件数': len(rr), '日数': int(valid.sum()),
                    'VWC 雨の日': int((rv & valid).sum()),
                    'VWC 雨の日のうち降水 > 0(当日か前日)': (wet[rv & valid]).mean(),
                    'VWC 雨の日のうち降水 ≧ 1mm': (wet1[rv & valid]).mean(),
                    '降水 ≧ 1mm の日のうち VWC 雨の日': rv[wet1 & valid].mean(),
                    'VWC 雨の日でない日のうち降水 ≧ 1mm': wet1[~rv & valid].mean(),
                    'APCPRA が欠けている日': int((~valid).sum())})
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(OUT_DIR, 'check_rain_vs_apcpra.csv'), index=False, encoding='utf-8-sig')
    log('\n[VWC で判定した雨の日と、気象 DB の APCPRA(当日か前日)の照合]')
    log(tab.round(3).to_string(index=False))
    pd.DataFrame({'field_id': meta.field_id, 'year': meta.year, 'irrigation': meta.irrigation, 'n_rain_days': nr,
                  'n_post_days_k3': npst, 'n_pre_missing': n_missing}).to_csv(
        os.path.join(OUT_DIR, 'check_rain_days.csv'), index=False, encoding='utf-8-sig')
    plot_examples(X, meta, rain, post, comps)
    # 入力の保存(k ごと)
    os.makedirs(NEW_DATA, exist_ok=True)
    for k in [K_MAIN] + K_SENS:
        ck, _, _ = components(X, pre, k)
        for nm in INPUTS:
            np.save(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{k}_X.npy'), ck[nm])
        assert np.allclose(ck['N_dry'] + ck['N_post'], ck['N'])
    ct, rt, pt_ = components(X, pre, K_MAIN, thr=THR)
    for nm in INPUTS:
        np.save(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{K_MAIN}_thr{THR}_X.npy'), ct[nm])
    assert np.allclose(ct['N_dry'] + ct['N_post'], ct['N'])
    log(f'しきい値版(雨の日 = Δθ ≧ {THR}、k = {K_MAIN}、記述用): 雨の日 平均 {rt.sum(axis=1).mean():.1f}、雨の直後 平均 '
        f'{pt_.sum(axis=1).mean():.1f}、無降雨日 平均 {(59 - rt.sum(axis=1) - pt_.sum(axis=1)).mean():.1f}')
    meta.to_csv(os.path.join(NEW_DATA, 'label_muniyear_median_meta.csv'), index=False, encoding='utf-8-sig')
    np.save(os.path.join(NEW_DATA, 'pre_window_diffs.npy'), pre)
    log(f'\n新しい入力を保存: {NEW_DATA}(k = {[K_MAIN] + K_SENS})')
    return pre


def plot_examples(X, meta, rain, post, comps):
    nr = rain.sum(axis=1)
    irr = (meta.irrigation != '無し').values
    order = np.argsort(nr, kind='stable')
    mid = [i for i in order[len(nr) // 2:] if not irr[i]][0]            # 雨の日が中央値付近で灌水なしの件
    pick = [int(mid), int(np.argmin(nr)), int(np.argmax(nr)),
            int([i for i in np.flatnonzero(irr) if i not in (mid, np.argmin(nr), np.argmax(nr))][0])]
    fig, axs = plt.subplots(len(pick), 1, figsize=(12, 2.6 * len(pick)), sharex=True)
    t = np.arange(60)
    for ax, i in zip(axs, pick):
        ax.plot(t, X[i, 0], color=INK, lw=1.4)
        for tt in range(59):
            if rain[i, tt]:
                ax.axvspan(tt + 0.5, tt + 1.5, color='#2a78d6', alpha=0.25, lw=0)
            elif post[i, tt]:
                ax.axvspan(tt + 0.5, tt + 1.5, color='#eb6834', alpha=0.18, lw=0)
        r = meta.iloc[i]
        ax.set_title(f'field_id {r.field_id}・{r.year}・{r.muniCd}・{"多収" if r.y == 1 else "低収"}・灌水 {r.irrigation}・'
                     f'雨の日 {int(rain[i].sum())}・雨の直後 {int(post[i].sum())}', loc='left', fontsize=9.5, color=INK)
        ax.set_ylabel('VWC')
        s6.style(ax)
    axs[-1].set_xlabel('開花後の日数(0 = 開花日)')
    from matplotlib.patches import Patch
    fig.legend([Patch(color='#2a78d6', alpha=0.25), Patch(color='#eb6834', alpha=0.18)],
               ['雨の日(Δθ > 0)', '雨の直後(k = 3、N_post に入る日)'], loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('段階29: 雨の日と雨の直後の判定の例(中央値・最少・最多の雨の日の件と、灌水ありの件)', fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, 'rain_day_examples.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本番
def part_main(X, meta, seeds, P, n_boot, log):
    log('\n==== 本番(k = 3) ====')
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    ref = pd.read_csv(os.path.join(s13.OUT_DIR if False else os.path.join(ROCKET_DIR, 'outputs', 'stage13'), 'splits.csv'),
                      encoding='utf-8-sig')
    ref = ref[ref.label == KEY].drop(columns='label').reset_index(drop=True)
    mine = split_table(sp, y, meta.muniCd.values)
    ok = mine.shape == ref.shape and np.allclose(mine.values.astype(float), ref.values.astype(float))
    log(f'確認 分割が段階13・27と同じ: {ok}')
    assert ok
    inputs = {nm: np.load(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{K_MAIN}_X.npy')) for nm in INPUTS}
    Nb = np.load(os.path.join(D27, 'label_muniyear_median_N_X.npy'))
    ok = np.allclose(inputs['N_dry'] + inputs['N_post'], Nb)
    log(f'確認 N_dry + N_post = 段階27の N: {ok}')
    assert ok
    # 段階27の N の再計算(乱数0、先頭10 fold)
    with open(os.path.join(S27, 'partial_N.pkl'), 'rb') as fh:
        n27 = pickle.load(fh)
    f27 = pd.concat(n27['out']['folds'])
    f27 = f27[f27.seed == 0].reset_index(drop=True)
    n10 = s6.eval_cached(s6.fold_features('minirocket', 0, Nb, sp[:10]), y, sp[:10])
    okn = np.allclose(n10.auc, f27.auc[:10], rtol=0, atol=1e-12, equal_nan=True) and \
        np.allclose(n10.bacc, f27.bacc[:10], rtol=0, atol=1e-12)
    log(f'確認 段階27の N の再計算(乱数0、先頭10 fold)が段階27と一致: {okn}(段階27の N の結果をそのまま使う)')
    assert okn and n27['tag']['seeds'][:len(seeds)] == seeds
    log('確認(N_post): 段階13の prechecks')
    s13.prechecks(KEY, inputs['N_post'], meta, sp, log)
    s27.OUT_DIR = OUT_DIR   # run_input の null_*.npy の保存先をこの段階に
    res = {}
    for nm in INPUTS:
        part = os.path.join(OUT_DIR, f'partial_{nm}.pkl')
        tag = {'seeds': seeds, 'P': P, 'k': K_MAIN}
        if os.path.exists(part):
            with open(part, 'rb') as fh:
                saved = pickle.load(fh)
            if saved['tag'] == tag:
                res[nm] = saved['out']
                log(f'\n[{nm}: {NAMES[nm]}] 途中結果を読み込んだ')
                continue
        log(f'\n[{nm}: {NAMES[nm]}]')
        res[nm] = s27.run_input(nm, inputs[nm], meta, sp, seeds, P, log)
        with open(part, 'wb') as fh:
            pickle.dump({'tag': tag, 'out': res[nm]}, fh)
    # N(段階27)を表に入れる
    nsum = pd.DataFrame(n27['out']['summary'])
    if len(seeds) < 10:
        ps = pd.DataFrame(n27['out']['seed'])
        ps = ps[ps.seed.isin(seeds)]
        for met in ['all_auc', 'all_bacc', 'W', 'B', 'X']:
            nsum.loc[nsum.metric == met, 'mean'] = ps[met].mean()
    nsum['config'] = 'N'
    summary = pd.concat([nsum] + [pd.DataFrame(res[nm]['summary']) for nm in INPUTS], ignore_index=True)
    m = summary.metric == 'all_auc'
    pin = summary[m & summary.config.isin(INPUTS)].set_index('config').p
    adj = pd.Series(s27.holm(pin.values), index=pin.index)
    summary.loc[m & summary.config.isin(INPUTS), 'p_holm'] = summary.loc[m & summary.config.isin(INPUTS), 'config'].map(adj)
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary['judged_above'] = None
    summary.loc[m & summary.config.isin(INPUTS), 'judged_above'] = summary.loc[m & summary.config.isin(INPUTS), 'p_holm'] < 0.05
    keep = ['config', 'metric', 'n', 'n_seeds', 'n_perm', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p',
            'p_holm', 'judged_above', 'above_q95', 'n_pairs_per_repeat', 'alpha_at_max']
    summary = summary.reindex(columns=keep)
    # 差の区間
    S_N = n27['out']['S'][:len(seeds)]
    Cs, N0 = {}, None
    for nm, S in [('N', S_N)] + [(n, res[n]['S']) for n in INPUTS]:
        C, Nn = s27.pair_sums(S, meta, sp)
        N0 = Nn if N0 is None else N0
        assert np.array_equal(Nn, N0)
        Cs[nm] = C
    boot, pooled = s27.boot_diff(Cs, N0, meta, n_boot)
    boot.to_csv(os.path.join(OUT_DIR, 'bootstrap_auc.csv'), index=False, encoding='utf-8-sig')
    fN = f27_all = pd.concat(n27['out']['folds'])
    fN = fN[fN.seed.isin(seeds)]
    fm = {'N': fN.auc.mean()}
    for nm in INPUTS:
        f = pd.concat(res[nm]['folds'])
        fm[nm] = f.auc.mean()
    rows = []
    for a, b in [('N_dry', 'N'), ('N_post', 'N'), ('R', 'N'), ('N_dry', 'N_post')]:
        d = boot[a] - boot[b]
        rows.append({'comparison': f'{a} − {b}', 'point_fold_mean': fm[a] - fm[b],
                     'point_pooled_pairs': pooled[a] - pooled[b], 'boot_q025': d.quantile(.025),
                     'boot_q975': d.quantile(.975), 'boot_frac_pos': (d > 0).mean(), 'n_boot': n_boot})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(os.path.join(OUT_DIR, 'diffs.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat([f for nm in INPUTS for f in res[nm]['folds']]).to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False,
                                                                      encoding='utf-8-sig')
    pd.concat([pd.DataFrame(res[nm]['seed']) for nm in INPUTS]).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'),
                                                                       index=False, encoding='utf-8-sig')
    pd.concat([p for nm in INPUTS for p in res[nm]['pair']]).to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False,
                                                                    encoding='utf-8-sig')
    pd.concat([s for nm in INPUTS for s in res[nm]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                      index=False, encoding='utf-8-sig')
    plot_main(summary, diffs)
    return summary, diffs


def plot_main(summary, diffs):
    cfgs = ['N'] + INPUTS
    sm = summary.set_index(['config', 'metric'])
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8), gridspec_kw={'width_ratios': [1.1, 1, 1, 1.2]})
    for ax, (met, nm) in zip(axs[:3], [('all_auc', '全体 AUC(判定)'), ('W', '地域内 W(記述)'), ('B', '地域間 B(記述)')]):
        for i, c in enumerate(cfgs):
            r = sm.loc[(c, met)]
            ax.bar(i, r['mean'], color=COL[c], alpha=0.75)
            lab = f'{r["mean"]:.3f}'
            if met == 'all_auc' and c in INPUTS:
                lab += f'\nHolm p={r["p_holm"]:.3f}'
            ax.text(i, r['mean'] + 0.006, lab, ha='center', fontsize=8, color=INK)
            if pd.notna(r.get('null_q95')):
                ax.hlines(r['null_q95'], i - 0.4, i + 0.4, color=s6.C_RED, lw=2, ls='--')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_xticks(range(len(cfgs)), cfgs)
        ax.set_ylim(0.35, 0.74)
        ax.set_title(nm, loc='left', fontsize=11, color=INK)
        s6.style(ax)
    axs[0].set_ylabel('AUC')
    ax = axs[3]
    for i, (_, r) in enumerate(diffs.iterrows()):
        ax.errorbar(r.point_fold_mean, i, xerr=[[r.point_fold_mean - r.boot_q025], [r.boot_q975 - r.point_fold_mean]],
                    fmt='o', color=INK, capsize=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(diffs)), diffs.comparison)
    ax.invert_yaxis()
    ax.set_xlabel('AUC の差(点 = fold 平均の差、線 = ブートストラップ95%区間)')
    ax.set_title('差(記述)', loc='left', fontsize=11, color=INK)
    s6.style(ax)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=s6.C_RED, lw=2, ls='--')], ['帰無分布の95%点'], loc='upper right', frameon=False,
               fontsize=9)
    fig.suptitle('段階29: N を成分に分ける(R 雨の日の位置 / N_dry 無降雨日の減少 / N_post 雨の直後の減少、k = 3)',
                 fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'stage29_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 感度分析(置換なし)
def part_sens(meta, seeds, log):
    log('\n==== 感度分析: k = 2・5 の N_dry・N_post(置換なし、判定に使わない) ====')
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    rows = []
    for k in K_SENS:
        for nm in ['N_dry', 'N_post']:
            Xk = np.load(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{k}_X.npy'))
            per = []
            for seed in seeds:
                f = s6.eval_cached(s6.fold_features('minirocket', seed, Xk, sp), y, sp)
                per.append(f.auc.mean())
            rows.append({'k': k, 'input': nm, 'all_auc': np.mean(per), 'seed_min': np.min(per), 'seed_max': np.max(per),
                         'n_seeds': len(seeds), 'share_nonzero': float((Xk != 0).mean())})
            log(f'  k = {k} {nm}: 全体 AUC {np.mean(per):.3f}(乱数 {np.min(per):.3f}〜{np.max(per):.3f})、0でない日の割合 '
                f'{(Xk != 0).mean():.3f}')
    log(f'\n==== 記述: しきい値版(雨の日 = Δθ ≧ {THR}、k = {K_MAIN})の R・N_dry・N_post(置換なし、判定に使わない) ====')
    for nm in INPUTS:
        Xk = np.load(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{K_MAIN}_thr{THR}_X.npy'))
        per = []
        for seed in seeds:
            f = s6.eval_cached(s6.fold_features('minirocket', seed, Xk, sp), y, sp)
            per.append(f.auc.mean())
        rows.append({'k': K_MAIN, 'input': nm, 'thr': THR, 'all_auc': np.mean(per), 'seed_min': np.min(per),
                     'seed_max': np.max(per), 'n_seeds': len(seeds), 'share_nonzero': float((Xk != 0).mean()),
                     'note': f'しきい値版(Δθ ≧ {THR})'})
        log(f'  thr {THR} {nm}: 全体 AUC {np.mean(per):.3f}(乱数 {np.min(per):.3f}〜{np.max(per):.3f})')
    sm = pd.read_csv(os.path.join(OUT_DIR, 'summary.csv'), encoding='utf-8-sig')
    r = sm[(sm.config == 'R') & (sm.metric == 'all_auc')].iloc[0]
    rows.append({'k': K_MAIN, 'input': 'R', 'thr': 0.0, 'all_auc': r['mean'], 'seed_min': r.seed_min,
                 'seed_max': r.seed_max, 'n_seeds': r.n_seeds, 'note': '本番(再掲)'})
    for nm in ['N_dry', 'N_post']:
        r = sm[(sm.config == nm) & (sm.metric == 'all_auc')].iloc[0]
        Xk = np.load(os.path.join(NEW_DATA, f'label_muniyear_median_{nm}_k{K_MAIN}_X.npy'))
        rows.append({'k': K_MAIN, 'input': nm, 'all_auc': r['mean'], 'seed_min': r.seed_min, 'seed_max': r.seed_max,
                     'n_seeds': r.n_seeds, 'share_nonzero': float((Xk != 0).mean()), 'note': '本番(再掲)'})
    t = pd.DataFrame(rows).sort_values(['input', 'k'])
    t.to_csv(os.path.join(OUT_DIR, 'sensitivity_k.csv'), index=False, encoding='utf-8-sig')
    log(t.round(3).to_string(index=False))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='+', default=['check'], choices=['check', 'main', 'sens'])
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    s6.N_JOBS = s13.N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, f'run_log_{"_".join(args.part)}.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {s13.N_JOBS}')
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 段階29・EXP-038 が未使用(docs に EXP-038 なし): {"EXP-038" not in docs}')
    X, meta = load()
    trial = '' if args.n_seeds == 10 and args.n_perm == 200 and args.n_boot == 2000 else '(試運転)'
    if 'check' in args.part:
        part_check(X, meta, log)
    if 'main' in args.part:
        summary, diffs = part_main(X, meta, seeds, args.n_perm, args.n_boot, log)
        log('\n[まとめ]')
        log(summary.round(3).to_string(index=False))
        log('\n[差(記述)]')
        log(diffs.round(4).to_string(index=False))
        sm = summary.set_index(['config', 'metric'])
        up = [nm for nm in INPUTS if sm.loc[(nm, 'all_auc'), 'p_holm'] < 0.05]
        log('\n[判定] ' + '、'.join(f'{nm} {sm.loc[(nm, "all_auc"), "mean"]:.3f}(p = {sm.loc[(nm, "all_auc"), "p"]:.3f}、'
                                    f'Holm 後 {sm.loc[(nm, "all_auc"), "p_holm"]:.3f})' for nm in INPUTS)
            + f' → 上回った入力: {up if up else "なし"}' + trial)
    if 'sens' in args.part:
        part_sens(meta, seeds, log)
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
