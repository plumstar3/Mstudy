"""
段階31: 新しい1時間ごとの土壌水分データ(D-008)で入力の形ごとに評価し、同じ件の日平均と比べる(EXP-040)。

データ: rocket/大豆元データ/soilmoisture_hourly_v2.csv(1時間)と soilmoisture_daily_v2.csv(日平均)。どちらも読むだけ。
ラベル: label_muniyear_median(主ラベル)。開花日は field_year_labeled.csv の flower_date。field_id 328 は読み込み後に除く。
窓: 開花日 0日目 00:00 〜 +59日目 23:00 の1440時間。主の件数は、窓の1440時間がすべてそろう件
(補間は make_soilmoisture_v2.py の連続3時間以内だけ。この段階では新たに補間しない)。日平均の入力も同じ件で作る。

1時間の入力(θ(h) = 窓の中の1時間ごとの VWC、Δθ(h) = θ(h) − θ(h−1)、h = 1 … 1439):
  主(判定)  H_raw θ(h)(1440点) / H_D Δθ / H_N min(Δθ, 0) / H_Npost H_N のうち雨の直後の時間 /
             H_Ndry H_N のうち雨の日でも雨の直後でもない時間
  探索(記述) H_Npost_night H_Npost のうち 18:00〜翌6:00 / H_Npost_day H_Npost のうち 6:00〜18:00
雨の日: 同じ件の日平均の Δθ > 0 の日(段階29と同じ)。窓の最初の日は、開花日より前の日平均があれば使い、無ければ雨の日なし。
雨の直後: 雨の日の翌日から3日目まで(k = 3)のうち雨の日でない日のすべての時間(雨の日が続く場合は最後の雨の日から数える)。
雨の日当日の時間は H_Npost にも H_Ndry にも入れない。
Δθ(h) の日と昼夜は、θ(h−1) → θ(h) の1時間の区間で決める:
  日 = θ(h) の時刻の暦日(23:00 → 0:00 の区間は翌日に入る)。
  夜 = 区間が 18:00〜翌6:00 に入る(θ(h) の時刻が 19:00〜23:00・0:00〜6:00)、昼 = 6:00〜18:00(θ(h) の時刻が 7:00〜18:00)。
  昼夜の区切りは固定の時刻で、日の出・日の入りには合わせていない。
日平均の入力(段階27・29と同じ定義、60点の日平均から): Dy_raw / Dy_D / Dy_N / Dy_Npost / Dy_Ndry(k = 3)。

モデル: sktime 1.2.0 の MiniRocket(10000、乱数0〜9)→ StandardScaler(with_mean=False)→ RidgeClassifierCV(logspace(−3, 3, 10) × n_train)。
分割: make_splits(y, muniCd, 5, 20, 0)(件数が変わるので新しく作る)。置換: 200回、市町村×年の中(置換 i は [0, i])。
判定(事前に固定、D-007): 主の1時間5入力の全体 AUC を帰無分布の95%点と比べ、5つの p に Holm 補正。日平均5入力も同様(確認として)。
差: 市町村単位の対応のあるクラスターブートストラップ(段階14・27と同じ重み付け、2000回、乱数 default_rng(0))。
  補正前の両側 p = min(1, 2 × (1 + min(#{差 ≤ 0}, #{差 ≥ 0})) / (B + 1))。Holm 補正は比較のまとまりごと。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage31\\run_stage31.py --part check
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage31\\run_stage31.py --part main
  試運転: ... --part main --n-seeds 1 --n-perm 20 --n-boot 50(出力は outputs/stage31/trial/)
出力: rocket/outputs/stage31/、新しい入力は rocket/data/processed/stage31/。入力ごとの途中結果を partial_<入力>.pkl に保存し、
同じ引数で実行し直すと読み込んで再開する。
"""
import argparse
import os
import pickle
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
for st in ['stage13', 'stage27', 'stage29']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', st))
import run_stage13 as s13  # noqa: E402
import run_stage27 as s27  # noqa: E402
import run_stage29 as s29  # noqa: E402
from src.evaluation import make_splits, split_table  # noqa: E402
from src.windows import first_diff  # noqa: E402

s6 = s13.s6
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage31')
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage31')
RAWDIR = os.path.join(ROCKET_DIR, '大豆元データ')
HOURLY = os.path.join(RAWDIR, 'soilmoisture_hourly_v2.csv')
DAILY = os.path.join(RAWDIR, 'soilmoisture_daily_v2.csv')
LABEL = 'label_muniyear_median'
KEY = 'median'
K = 3
N_DAYS, N_HOURS = 60, 1440
PRE_DAYS = K + 1          # 開花日より前の日平均 θ(−4 … −1)。雨の日の判定は日 −3 から
H_MAIN = ['H_raw', 'H_D', 'H_N', 'H_Npost', 'H_Ndry']
H_EXPL = ['H_Npost_night', 'H_Npost_day']
D_MAIN = ['Dy_raw', 'Dy_D', 'Dy_N', 'Dy_Npost', 'Dy_Ndry']
ALL_INPUTS = H_MAIN + D_MAIN + H_EXPL
NAMES = {'raw': '生の値', 'D': '1次差分', 'N': '減る方向の1次差分', 'Npost': '雨の直後の減少', 'Ndry': '無降雨日の減少',
         'Npost_night': '雨の直後の夜の減少', 'Npost_day': '雨の直後の昼の減少'}
COMPARE = {
    '1時間の中': [('H_D', 'H_raw'), ('H_N', 'H_D'), ('H_Npost', 'H_N'), ('H_Ndry', 'H_Npost')],
    '日平均の中': [('Dy_D', 'Dy_raw'), ('Dy_N', 'Dy_D'), ('Dy_Npost', 'Dy_N'), ('Dy_Ndry', 'Dy_Npost')],
    '1時間と日平均': [('H_raw', 'Dy_raw'), ('H_D', 'Dy_D'), ('H_N', 'Dy_N'), ('H_Npost', 'Dy_Npost'), ('H_Ndry', 'Dy_Ndry')],
    '探索的': [('H_Npost_night', 'H_Npost_day')],
}
INK, MUTED = s6.INK, s6.MUTED
C_H, C_D = '#2a78d6', '#c9802a'
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def desc(nm):
    return ('1時間・' if nm.startswith('H_') else '日平均・') + NAMES[nm.split('_', 1)[1]]


# ---------------------------------------------------------------- データ
def load_sources():
    cases = s13.load_cases(LABEL)
    h = pd.read_csv(HOURLY, encoding='utf-8-sig')
    d = pd.read_csv(DAILY, encoding='utf-8-sig')
    h = h[h.field_id != 328].copy()
    d = d[d.field_id != 328].copy()
    h['datetime'] = pd.to_datetime(h.datetime)
    d['date'] = pd.to_datetime(d.date)
    return cases, h, d


def build(cases, h, d, log):
    """1時間の窓(1440)・日平均の窓(60)・開花日より前の日平均(4日)・interpolated。主の件 = 1440時間がそろう件。"""
    hk = {k: g for k, g in h.groupby(['field_id', 'year'])}
    dk = {k: g for k, g in d.groupby(['field_id', 'year'])}
    rows, Xh, Xd, PRE, INTP = [], [], [], [], []
    for i, r in cases.iterrows():
        key = (r.field_id, r.year)
        fd = r.flower_date.normalize()
        times = pd.date_range(fd, periods=N_HOURS, freq='h')
        days = pd.date_range(fd - pd.Timedelta(days=PRE_DAYS), periods=N_DAYS + PRE_DAYS, freq='D')
        gh = hk.get(key)
        gd = dk.get(key)
        vh = gh.set_index('datetime').reindex(times) if gh is not None else pd.DataFrame(index=times, columns=['VWC'])
        vd = gd.set_index('date').reindex(days) if gd is not None else pd.DataFrame(index=days, columns=['VWC', 'daf'])
        if gd is not None:   # daf = date − 開花日 の確認
            dd = gd.dropna(subset=['daf'])
            assert ((dd.date - fd).dt.days == dd.daf).all(), f'{key}: daf が flower_date と合わない'
        miss_h = vh.VWC.isna().values
        n_day = int(vd.VWC.iloc[PRE_DAYS:].notna().sum())
        gap = 0
        run = 0
        for m in miss_h:
            run = run + 1 if m else 0
            gap = max(gap, run)
        rows.append({'idx': i, 'field_id': r.field_id, 'year': r.year, 'y': r.y, 'muniCd': r.muniCd,
                     'n_days_present': n_day, 'n_hours_missing': int(miss_h.sum()), 'max_gap_hours': gap,
                     'complete_hourly': not miss_h.any(), 'complete_daily': n_day == N_DAYS,
                     'n_hours_lt24_days': int((vd.n_hours.iloc[PRE_DAYS:] < 24).sum()) if gd is not None else np.nan})
        Xh.append(vh.VWC.values.astype(float))
        Xd.append(vd.VWC.values[PRE_DAYS:].astype(float))
        PRE.append(vd.VWC.values[:PRE_DAYS].astype(float))
        INTP.append(vh.interpolated.values if gh is not None else np.full(N_HOURS, np.nan))
    st = pd.DataFrame(rows)
    keep = st.complete_hourly.values
    meta = cases.loc[keep].reset_index(drop=True)
    Xh = np.array([x for x, k in zip(Xh, keep) if k])[:, None, :]
    Xd = np.array([x for x, k in zip(Xd, keep) if k])[:, None, :]
    pre = np.array([x for x, k in zip(PRE, keep) if k])
    intp = np.array([x for x, k in zip(INTP, keep) if k]).astype(bool)
    return Xh, Xd, pre, intp, meta, st


def day_flags(Xd, pre):
    """雨の日 rain(n, 60、日 0 … 59)と雨の直後 post(n, 60)。
    日平均の Δθ(d) = θ(d) − θ(d−1) > 0 が雨の日。d = −3 … 59 で判定(θ が無い日は雨の日なし)。
    post(d) = d−3 … d−1 に雨の日があり、d が雨の日でない。"""
    th = np.concatenate([pre, Xd[:, 0, :]], axis=1)           # 日 −4 … 59
    dl = th[:, 1:] - th[:, :-1]                                # 日 −3 … 59
    rain_ext = np.nan_to_num(dl, nan=0.0) > 0
    off = PRE_DAYS - 1                                         # rain_ext の列 off が日 0
    rain = rain_ext[:, off:]
    post = np.zeros_like(rain)
    for t in range(N_DAYS):
        te = t + off
        post[:, t] = rain_ext[:, te - K:te].any(axis=1) & ~rain[:, t]
    return rain, post


def hourly_inputs(Xh, rain, post):
    d = Xh[:, 0, 1:] - Xh[:, 0, :-1]                           # h = 1 … 1439
    h = np.arange(1, N_HOURS)
    day = h // 24
    hod = h % 24                                              # θ(h) の時刻
    night = (hod >= 19) | (hod <= 6)                          # 区間 (h−1, h] が 18:00〜翌6:00
    R, Pst = rain[:, day], post[:, day]
    N = np.minimum(d, 0.0)
    out = {'H_raw': Xh[:, 0, :], 'H_D': d, 'H_N': N,
           'H_Npost': np.where(Pst, N, 0.0), 'H_Ndry': np.where(~R & ~Pst, N, 0.0)}
    out['H_Npost_night'] = np.where(night[None, :], out['H_Npost'], 0.0)
    out['H_Npost_day'] = np.where(~night[None, :], out['H_Npost'], 0.0)
    rain_part = np.where(R, N, 0.0)
    return {k: v[:, None, :] for k, v in out.items()}, rain_part, night, d


def daily_inputs(Xd, pre):
    """段階29の components をそのまま使う(pre は t = −2 … 0 の Δθ)。"""
    th = np.concatenate([pre, Xd[:, 0, :]], axis=1)
    dl = th[:, 1:] - th[:, :-1]                                # 日 −3 … 59
    pre29 = dl[:, 1:PRE_DAYS]                                  # 日 −2 … 0(k = 3 で使う分)
    comps, rain29, post29 = s29.components(Xd, pre29, K)
    out = {'Dy_raw': Xd, 'Dy_D': first_diff(Xd), 'Dy_N': comps['N'], 'Dy_Npost': comps['N_post'],
           'Dy_Ndry': comps['N_dry']}
    return out, rain29, post29


# ---------------------------------------------------------------- 開始時の確認
def part_check(log):
    log('\n==== データと開始時の確認 ====')
    cases, h, d = load_sources()
    log(f'ラベル付き {len(cases)} 件(field_year_labeled.csv の {LABEL})。1時間 {len(h)} 行・日平均 {len(d)} 行(328 を除いた後)')
    assert 328 not in set(h.field_id) and 328 not in set(d.field_id)
    Xh, Xd, pre, intp, meta, st = build(cases, h, d, log)
    st.to_csv(os.path.join(OUT_DIR, 'case_coverage.csv'), index=False, encoding='utf-8-sig')
    n121 = st[st.complete_daily]
    log(f'窓の60日の日平均がすべてある件(n_hours ≥ 1): {len(n121)}(多収 {int(n121.y.sum())} / 低収 {int((1 - n121.y).sum())})')
    log(f'窓の1440時間がすべてそろう件(主): {len(meta)}(多収 {int(meta.y.sum())} / 低収 {int((1 - meta.y).sum())})')
    out121 = n121[~n121.complete_hourly]
    log('121件から外れた件: ' + '; '.join(
        f'field_id {r.field_id}・{r.year}・{r.muniCd}・{"多収" if r.y == 1 else "低収"}・欠けた時間 {r.n_hours_missing}'
        f'(最長 {r.max_gap_hours} 時間連続)・24時間そろわない日 {int(r.n_hours_lt24_days)}' for r in out121.itertuples()))
    old = pd.read_csv(os.path.join(s29.D13 + '_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    ko = set(zip(old.field_id, old.year))
    kn = set(zip(meta.field_id, meta.year))
    log(f'段階29(旧日平均123件)との違い: 旧にだけある {sorted(ko - kn)}、新にだけある {sorted(kn - ko)}、共通 {len(ko & kn)}')
    y = meta.y.values
    g = meta.groupby('muniyear').y.agg(['size', 'sum'])
    both = g[(g['sum'] > 0) & (g['sum'] < g['size'])]
    w_pairs = int((both['sum'] * (both['size'] - both['sum'])).sum())
    log(f'主の件: {len(meta)} 件、地域 {meta.muniCd.nunique()}、市町村×年 {len(g)}(両クラスがそろう {len(both)})、'
        f'W のペア {w_pairs}、年別 {meta.year.value_counts().sort_index().to_dict()}、開花日 {meta.flower_source.value_counts().to_dict()}')
    ok = np.isfinite(Xh).all() and Xh.shape == (len(meta), 1, N_HOURS) and np.isfinite(Xd).all() \
        and Xd.shape == (len(meta), 1, N_DAYS)
    log(f'確認1 窓の1440時間・60日がすべてそろい、欠測が無い: {ok}')
    assert ok
    # 日平均 = 1時間の平均(窓の中)
    hm = Xh[:, 0, :].reshape(len(meta), N_DAYS, 24).mean(axis=2)
    ok = np.allclose(hm, Xd[:, 0, :], rtol=0, atol=1e-12)
    log(f'確認2 日平均の VWC = その日の1時間の VWC の平均(窓の60日すべて): {ok}(最大差 {np.abs(hm - Xd[:, 0, :]).max():.2e})')
    assert ok
    # interpolated
    fr = intp.mean(axis=1)
    meta_i = meta.assign(interp_frac=fr)
    log(f'interpolated の割合(件ごと、窓の1440時間): 平均 {fr.mean():.3f}、中央値 {np.median(fr):.3f}、0 の件 {int((fr == 0).sum())}、'
        f'0.4 超の件 {int((fr > 0.4).sum())}、最大 {fr.max():.3f}')
    for lab, nm in [(1, '多収'), (0, '低収')]:
        v = fr[y == lab]
        log(f'  {nm}: 平均 {v.mean():.3f}、中央値 {np.median(v):.3f}、0 の件 {int((v == 0).sum())} / {len(v)}')
    log('  年別の平均: ' + ', '.join(f'{yr}年 {v:.3f}' for yr, v in meta_i.groupby('year').interp_frac.mean().items()))
    # 入力
    rain, post = day_flags(Xd, pre)
    H, rain_part, night, dh = hourly_inputs(Xh, rain, post)
    D, rain29, post29 = daily_inputs(Xd, pre)
    ok_a = (H['H_N'] <= 0).all() and (H['H_Npost'] <= 0).all() and (H['H_Ndry'] <= 0).all() and (D['Dy_N'] <= 0).all()
    ok_b = np.allclose(H['H_Npost'][:, 0] + H['H_Ndry'][:, 0] + rain_part, H['H_N'][:, 0], rtol=0, atol=0)
    ok_c = np.allclose(H['H_Npost_night'] + H['H_Npost_day'], H['H_Npost'], rtol=0, atol=0)
    ok_d = np.array_equal(rain29, rain[:, 1:]) and np.array_equal(post29, post[:, 1:])
    ok_e = np.allclose(D['Dy_N'], s27.neg_diff(Xd)) and np.allclose(D['Dy_Npost'] + D['Dy_Ndry'], D['Dy_N']) \
        and np.allclose(D['Dy_D'], np.diff(Xd, axis=2))
    ok_f = all(H[k].shape == (len(meta), 1, N_HOURS - (0 if k == 'H_raw' else 1)) for k in H) and \
        all(D[k].shape == (len(meta), 1, N_DAYS - (0 if k == 'Dy_raw' else 1)) for k in D)
    log(f'確認3 H_N・H_Npost・H_Ndry・Dy_N ≦ 0: {ok_a}')
    log(f'確認4 H_Npost + H_Ndry + 雨の日当日の H_N = H_N(すべての時間、完全一致): {ok_b}、夜 + 昼 = H_Npost: {ok_c}')
    log(f'確認5 日平均の入力は段階29の components と同じ作り方: 雨の日・雨の直後(日1〜59)が一致 {ok_d}、'
        f'Dy_N = 段階27の neg_diff・Dy_Npost + Dy_Ndry = Dy_N・Dy_D = 1次差分 {ok_e}')
    log(f'確認6 長さ: H_raw 1440・ほかの1時間 1439・Dy_raw 60・ほかの日平均 59: {ok_f}')
    assert ok_a and ok_b and ok_c and ok_d and ok_e and ok_f
    # 雨の日の数
    nr = rain.sum(axis=1)
    npst = post.sum(axis=1)
    pre_miss = np.isnan(pre).sum(axis=1)
    log(f'雨の日(日平均の Δθ > 0)の数(件ごと、窓の60日中): 平均 {nr.mean():.1f}、中央値 {np.median(nr):.0f}、'
        f'最小 {nr.min()}、最大 {nr.max()}、5・25・75・95%点 {np.percentile(nr, [5, 25, 75, 95]).round(1).tolist()}')
    log(f'雨の直後(k = 3)の日: 平均 {npst.mean():.1f}(中央値 {np.median(npst):.0f})、無降雨日: 平均 {(N_DAYS - nr - npst).mean():.1f}')
    log(f'開花日より前の日平均 θ(−4 … −1): 全部ある {int((pre_miss == 0).sum())} / 一部欠け '
        f'{int(((pre_miss > 0) & (pre_miss < PRE_DAYS)).sum())} / 全部無い {int((pre_miss == PRE_DAYS).sum())}')
    # 1時間の Δθ > 0 の頻度(雨の日を1時間で判定しない理由の記述)
    hday = np.arange(1, N_HOURS) // 24
    pos = dh > 0
    log(f'1時間の Δθ > 0 の割合: 雨の日の時間 {pos[rain[:, hday]].mean():.3f}、雨の日でない時間 {pos[~rain[:, hday]].mean():.3f}'
        f'(雨の日でない日にも1時間の上昇が多い。中央値の大きさ {np.median(dh[pos & ~rain[:, hday]]):.5f})')
    frac_night = np.abs(H['H_Npost_night'][:, 0]).sum(axis=1) / np.maximum(np.abs(H['H_Npost'][:, 0]).sum(axis=1), 1e-300)
    log(f'H_Npost の減少量のうち夜の区間の割合(件ごと): 平均 {frac_night.mean():.3f}、中央値 {np.median(frac_night):.3f}'
        f'(時間数では夜 12 / 昼 12)')
    # 旧日平均(段階29)の雨の日との一致率
    Xo, mo = s29.load()
    preo = np.load(os.path.join(s29.NEW_DATA, 'pre_window_diffs.npy'))
    _, rain_o, post_o = s29.components(Xo, preo, K)
    io = {k: i for i, k in enumerate(zip(mo.field_id, mo.year))}
    pairs = [(i, io[k]) for i, k in enumerate(zip(meta.field_id, meta.year)) if k in io]
    a = np.array([rain[i, 1:] for i, _ in pairs])
    b = np.array([rain_o[j] for _, j in pairs])
    per = (a == b).mean(axis=1)
    pa = np.array([post[i, 1:] for i, _ in pairs])
    pb = np.array([post_o[j] for _, j in pairs])
    log(f'段階29(旧日平均)の雨の日との一致率(共通 {len(pairs)} 件、日1〜59): 全体 {(a == b).mean():.4f}、'
        f'件ごとの最小 {per.min():.3f}、完全一致の件 {int((per == 1).sum())}。雨の直後の一致率 {(pa == pb).mean():.4f}')
    bad = [(meta.field_id[i], meta.year[i], int((a[k] != b[k]).sum())) for k, (i, _) in enumerate(pairs) if per[k] < 1]
    log(f'  一致しない日がある件(field_id, year, 日数): {bad[:20]}{" ..." if len(bad) > 20 else ""}')
    pd.DataFrame({'field_id': meta.field_id, 'year': meta.year, 'y': y, 'muniCd': meta.muniCd, 'interp_frac': fr,
                  'n_rain_days': nr, 'n_post_days': npst, 'n_pre_missing': pre_miss,
                  'frac_Npost_night': frac_night}).to_csv(os.path.join(OUT_DIR, 'check_cases.csv'), index=False,
                                                          encoding='utf-8-sig')
    # 保存
    os.makedirs(NEW_DATA, exist_ok=True)
    for k, v in {**H, **D}.items():
        np.save(os.path.join(NEW_DATA, f'{LABEL}_{k}_X.npy'), v)
    np.save(os.path.join(NEW_DATA, 'rain_days.npy'), rain)
    np.save(os.path.join(NEW_DATA, 'post_days.npy'), post)
    np.save(os.path.join(NEW_DATA, 'pre_daily_theta.npy'), pre)
    np.save(os.path.join(NEW_DATA, 'interpolated.npy'), intp)
    meta.to_csv(os.path.join(NEW_DATA, f'{LABEL}_meta.csv'), index=False, encoding='utf-8-sig')
    log(f'新しい入力を保存: {NEW_DATA}')
    plot_examples(Xh, meta, rain, post)
    # MiniRocket の dilation と特徴量数
    from sktime.transformations.panel.rocket import MiniRocket
    for nm, L in [('H_raw', N_HOURS), ('H_D 他', N_HOURS - 1), ('Dy_raw', N_DAYS), ('Dy_D 他', N_DAYS - 1)]:
        Xs = H['H_raw'] if nm == 'H_raw' else H['H_D'] if nm.startswith('H_') else D['Dy_raw'] if nm == 'Dy_raw' else D['Dy_D']
        mr = MiniRocket(num_kernels=s6.N_KERNELS, random_state=0).fit(Xs)
        dil, nfd = mr.parameters[0], mr.parameters[1]
        log(f'MiniRocket(長さ {L}): dilation {len(dil)} 種類 {dil.tolist()}、各 dilation の特徴量/カーネル {nfd.tolist()}、'
            f'特徴量 {int(nfd.sum()) * 84}(カーネル 84 本)')
    return meta


def plot_examples(Xh, meta, rain, post):
    nr = rain.sum(axis=1)
    irr = (meta.irrigation != '無し').values
    order = np.argsort(nr, kind='stable')
    mid = [i for i in order[len(nr) // 2:] if not irr[i]][0]
    pick = [int(mid), int(np.argmin(nr)), int(np.argmax(nr))]
    rest = [i for i in np.flatnonzero(irr) if i not in pick]
    if rest:
        pick.append(int(rest[0]))
    fig, axs = plt.subplots(len(pick), 1, figsize=(13, 2.7 * len(pick)), sharex=True)
    t = np.arange(N_HOURS) / 24
    for ax, i in zip(axs, pick):
        for dd in range(N_DAYS):
            if rain[i, dd]:
                ax.axvspan(dd, dd + 1, color=C_H, alpha=0.22, lw=0)
            elif post[i, dd]:
                ax.axvspan(dd, dd + 1, color='#eb6834', alpha=0.16, lw=0)
        ax.plot(t, Xh[i, 0], color=INK, lw=0.9)
        dm = Xh[i, 0].reshape(N_DAYS, 24).mean(axis=1)
        ax.plot(np.arange(N_DAYS) + 0.5, dm, color=MUTED, lw=1.0, ls='--', marker='o', ms=2)
        r = meta.iloc[i]
        ax.set_title(f'field_id {r.field_id}・{r.year}・{r.muniCd}・{"多収" if r.y == 1 else "低収"}・灌水 {r.irrigation}・'
                     f'雨の日 {int(rain[i].sum())}・雨の直後 {int(post[i].sum())}', loc='left', fontsize=9.5, color=INK)
        ax.set_ylabel('VWC')
        s6.style(ax)
    axs[-1].set_xlabel('開花後の日数(0 = 開花日 0:00)')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Line2D([], [], color=INK, lw=0.9), Line2D([], [], color=MUTED, ls='--', marker='o', ms=2),
                Patch(color=C_H, alpha=0.22), Patch(color='#eb6834', alpha=0.16)],
               ['1時間の VWC', '日平均', '雨の日(日平均の Δθ > 0)', '雨の直後(k = 3、H_Npost に入る時間)'],
               loc='upper right', frameon=False, fontsize=9, ncol=2)
    fig.suptitle('段階31: 雨の日と雨の直後の判定の例(雨の日が中央値・最少・最多の件と灌水ありの件)', fontsize=12, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, 'rain_day_examples.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 本番
def load_inputs():
    meta = pd.read_csv(os.path.join(NEW_DATA, f'{LABEL}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    X = {k: np.load(os.path.join(NEW_DATA, f'{LABEL}_{k}_X.npy')) for k in ALL_INPUTS}
    return X, meta


def two_sided_p(d):
    B = len(d)
    return min(1.0, 2 * (1 + min(int((d <= 0).sum()), int((d >= 0).sum()))) / (B + 1))


def part_main(seeds, P, n_boot, out_dir, log):
    log('\n==== 本番 ====')
    X, meta = load_inputs()
    y = meta.y.values
    sp = make_splits(y, meta.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    tab = split_table(sp, y, meta.muniCd.values)
    tab.to_csv(os.path.join(out_dir, 'splits.csv'), index=False, encoding='utf-8-sig')
    log(f'分割: 5分割 × 20回、テストの件数 {tab.n_test.min()}〜{tab.n_test.max()}、テストが1クラスだけの fold '
        f'{int(((tab.n_test_pos == 0) | (tab.n_test_neg == 0)).sum())}')
    if not os.path.exists(os.path.join(out_dir, 'prechecks_done.txt')):
        for nm in ['H_Npost', 'Dy_Npost']:
            log(f'確認({nm}): 段階13の prechecks(Pipeline と特徴量の使い回し、置換で分割が変わらない、まとめて解いた置換 = sklearn)')
            s13.prechecks(KEY, X[nm], meta, sp, log)
        open(os.path.join(out_dir, 'prechecks_done.txt'), 'w').write('ok\n')
    else:
        log('確認: prechecks は済み(prechecks_done.txt)')
    s27.OUT_DIR = out_dir
    res = {}
    for nm in ALL_INPUTS:
        part = os.path.join(out_dir, f'partial_{nm}.pkl')
        tag = {'seeds': seeds, 'P': P, 'n': len(y)}
        if os.path.exists(part):
            with open(part, 'rb') as fh:
                saved = pickle.load(fh)
            if saved['tag'] == tag:
                res[nm] = saved['out']
                log(f'\n[{nm}: {desc(nm)}] 途中結果を読み込んだ')
                continue
        log(f'\n[{nm}: {desc(nm)}]')
        t0 = time.time()
        res[nm] = s27.run_input(nm, X[nm], meta, sp, seeds, P, log)
        res[nm]['sec'] = time.time() - t0
        with open(part, 'wb') as fh:
            pickle.dump({'tag': tag, 'out': res[nm]}, fh)
    summary = pd.concat([pd.DataFrame(res[nm]['summary']) for nm in ALL_INPUTS], ignore_index=True)
    m = summary.metric == 'all_auc'
    summary['family'] = summary.config.map(lambda c: '1時間(主)' if c in H_MAIN else '日平均(主)' if c in D_MAIN else '1時間(探索)')
    summary['p_holm'] = np.nan
    for fam in [H_MAIN, D_MAIN]:
        sel = m & summary.config.isin(fam)
        summary.loc[sel, 'p_holm'] = s27.holm(summary.loc[sel, 'p'].values)
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary['judged_above'] = None
    sel = m & summary.config.isin(H_MAIN + D_MAIN)
    summary.loc[sel, 'judged_above'] = summary.loc[sel, 'p_holm'] < 0.05
    keep = ['family', 'config', 'metric', 'n', 'n_seeds', 'n_perm', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95',
            'p', 'p_holm', 'judged_above', 'above_q95', 'n_pairs_per_repeat', 'alpha_at_max']
    summary = summary.reindex(columns=keep)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    # 差
    Cs, N0 = {}, None
    for nm in ALL_INPUTS:
        C, Nn = s27.pair_sums(res[nm]['S'], meta, sp)
        N0 = Nn if N0 is None else N0
        assert np.array_equal(Nn, N0)
        Cs[nm] = C
    boot, pooled = s27.boot_diff(Cs, N0, meta, n_boot)
    boot.to_csv(os.path.join(out_dir, 'bootstrap_auc.csv'), index=False, encoding='utf-8-sig')
    fm = {nm: pd.concat(res[nm]['folds']).auc.mean() for nm in ALL_INPUTS}
    rows = []
    for fam, prs in COMPARE.items():
        for a, b in prs:
            dd = boot[a] - boot[b]
            rows.append({'family': fam, 'comparison': f'{a} − {b}', 'point_fold_mean': fm[a] - fm[b],
                         'point_pooled_pairs': pooled[a] - pooled[b], 'boot_q025': dd.quantile(.025),
                         'boot_q975': dd.quantile(.975), 'p_two_sided': two_sided_p(dd.values),
                         'n_le0': int((dd <= 0).sum()), 'n_ge0': int((dd >= 0).sum()), 'n_boot': n_boot})
    diffs = pd.DataFrame(rows)
    diffs['p_holm_family'] = np.nan
    for fam in ['1時間の中', '日平均の中', '1時間と日平均']:
        sel = diffs.family == fam
        diffs.loc[sel, 'p_holm_family'] = s27.holm(diffs.loc[sel, 'p_two_sided'].values)
    diffs.to_csv(os.path.join(out_dir, 'diffs.csv'), index=False, encoding='utf-8-sig')
    pd.concat([f for nm in ALL_INPUTS for f in res[nm]['folds']]).to_csv(os.path.join(out_dir, 'folds.csv'), index=False,
                                                                         encoding='utf-8-sig')
    pd.concat([pd.DataFrame(res[nm]['seed']) for nm in ALL_INPUTS]).to_csv(os.path.join(out_dir, 'per_seed.csv'),
                                                                           index=False, encoding='utf-8-sig')
    pd.concat([p for nm in ALL_INPUTS for p in res[nm]['pair']]).to_csv(os.path.join(out_dir, 'pair_auc.csv'),
                                                                        index=False, encoding='utf-8-sig')
    pd.concat([s for nm in ALL_INPUTS for s in res[nm]['scores']]).to_csv(os.path.join(out_dir, 'oof_scores.csv'),
                                                                          index=False, encoding='utf-8-sig')
    log('\n入力ごとの計算時間(秒、この実行で計算したもの): ' + ', '.join(
        f'{nm} {res[nm]["sec"]:.0f}' for nm in ALL_INPUTS if 'sec' in res[nm]))
    plot_main(summary, diffs, out_dir)
    return summary, diffs


def plot_main(summary, diffs, out_dir):
    sm = summary.set_index(['config', 'metric'])
    forms = ['raw', 'D', 'N', 'Npost', 'Ndry']
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.6), gridspec_kw={'width_ratios': [1.5, 1, 1, 1.4]})
    for ax, (met, title) in zip(axs[:3], [('all_auc', '全体 AUC(判定)'), ('W', '地域内 W(記述)'), ('B', '地域間 B(記述)')]):
        xs = []
        for i, f in enumerate(forms):
            for j, (pref, col) in enumerate([('H_', C_H), ('Dy_', C_D)]):
                c = pref + f
                r = sm.loc[(c, met)]
                x = i + (j - 0.5) * 0.38
                ax.bar(x, r['mean'], width=0.36, color=col, alpha=0.8)
                lab = f'{r["mean"]:.3f}'
                top = max(r['mean'], r['null_q95'] if pd.notna(r.get('null_q95')) else 0)
                ax.text(x, top + 0.008, lab, ha='center', fontsize=6.5, color=INK, linespacing=1.0)
                if pd.notna(r.get('null_q95')):
                    ax.hlines(r['null_q95'], x - 0.18, x + 0.18, color=s6.C_RED, lw=1.8, ls='--')
            xs.append(i)
        if met == 'all_auc':
            for k, c in enumerate(H_EXPL):
                r = sm.loc[(c, met)]
                x = len(forms) + k * 0.45
                ax.bar(x, r['mean'], width=0.36, color=C_H, alpha=0.4, hatch='//')
                ax.text(x, r['mean'] + 0.006, f'{r["mean"]:.3f}', ha='center', fontsize=7, color=INK)
                ax.hlines(r['null_q95'], x - 0.18, x + 0.18, color=s6.C_RED, lw=1.8, ls='--')
            ax.set_xticks(list(range(len(forms))) + [len(forms), len(forms) + 0.45],
                          forms + ['夜', '昼'], fontsize=9)
        else:
            ax.set_xticks(range(len(forms)), forms)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
        ax.set_ylim(0.3, 0.8)
        ax.set_title(title, loc='left', fontsize=11, color=INK)
        s6.style(ax)
    axs[0].set_ylabel('AUC')
    ax = axs[3]
    for i, (_, r) in enumerate(diffs.iterrows()):
        col = C_H if r.family == '1時間の中' else C_D if r.family == '日平均の中' else INK if r.family == '1時間と日平均' else MUTED
        ax.errorbar(r.point_fold_mean, i, xerr=[[r.point_fold_mean - r.boot_q025], [r.boot_q975 - r.point_fold_mean]],
                    fmt='o', color=col, capsize=3)
        ax.text(1.02, i, f'p={r.p_two_sided:.3f}', transform=ax.get_yaxis_transform(), ha='left', va='center',
                fontsize=7.5, color=MUTED, clip_on=False)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(diffs)), diffs.comparison, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('AUC の差(点 = fold 平均、線 = 95%区間、p = 補正前の両側)', fontsize=9)
    ax.set_title('設定どうしの差', loc='left', fontsize=11, color=INK)
    s6.style(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend([Patch(color=C_H, alpha=0.8), Patch(color=C_D, alpha=0.8), Line2D([], [], color=s6.C_RED, lw=2, ls='--')],
               ['1時間', '日平均', '帰無分布の95%点'], loc='upper right', frameon=False, fontsize=9, ncol=3)
    fig.suptitle(f'段階31: 1時間と日平均の入力の形ごとの評価(主ラベル {int(summary.n.iloc[0])} 件、1440時間がそろう件)',
                 fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(os.path.join(out_dir, 'stage31_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='+', default=['check'], choices=['check', 'main'])
    ap.add_argument('--n-perm', type=int, default=200)
    ap.add_argument('--n-seeds', type=int, default=10)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    s6.N_JOBS = s13.N_JOBS
    trial = not (args.n_seeds == 10 and args.n_perm == 200 and args.n_boot == 2000)
    out_dir = os.path.join(OUT_DIR, 'trial') if trial and 'main' in args.part else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(out_dir, f'run_log_{"_".join(args.part)}.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {s13.N_JOBS}{"(試運転)" if trial else ""}')
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 段階31・EXP-040 が未使用(docs に EXP-040 なし): {"EXP-040" not in docs}')
    if 'check' in args.part:
        part_check(log)
    if 'main' in args.part:
        summary, diffs = part_main(seeds, args.n_perm, args.n_boot, out_dir, log)
        log('\n[まとめ]')
        log(summary.round(3).to_string(index=False))
        log('\n[差]')
        log(diffs.round(4).to_string(index=False))
        sm = summary.set_index(['config', 'metric'])
        for fam, lst in [('1時間', H_MAIN), ('日平均', D_MAIN)]:
            up = [nm for nm in lst if sm.loc[(nm, 'all_auc'), 'p_holm'] < 0.05]
            log(f'\n[判定 {fam}] ' + '、'.join(
                f'{nm} {sm.loc[(nm, "all_auc"), "mean"]:.3f}(p = {sm.loc[(nm, "all_auc"), "p"]:.3f}、'
                f'Holm 後 {sm.loc[(nm, "all_auc"), "p_holm"]:.3f})' for nm in lst) + f' → 上回った入力: {up if up else "なし"}')
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {out_dir}')
    logf.close()


if __name__ == '__main__':
    main()
