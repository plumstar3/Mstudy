"""
段階38: N_post は「排水の速さ」か「雨への反応の大きさ」か — 尺度合わせした N_post 入力の比較(EXP-047、D-018)。

雨の直後の減少 N_post(段階29の定義、k = 3、雨の日 Δθ > 0)を、
  NP       そのまま(基準)
  NP_rise  その値が属する雨のイベントの上昇量 θ(e) − θ(s−1) で割る(上昇量 < 0.002 のイベントの後は0)
  NP_fc    圃場ごとの θ30(吸引圧30cm の体積含水率)で割る
  NP_phi   圃場ごとの φ = 1 − 仮比重/2.6 で割る
の4通りにして、段階29と同じ MiniRocket の構成で全体 AUC を比べる。土の物性だけの基準線(LogisticRegression)も同じ分割で出す。
主の解析集合は、主ラベル120件のうち仮比重と θ30 の両方がある件。設定は docs/decisions.md の D-018(計算の前に記録)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage38\\run_stage38.py --part soil inputs   (土壌物性の表・入力の作成と確認)
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage38\\run_stage38.py --part trial         (乱数2・置換20・ブートストラップ100、確認だけ)
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage38\\run_stage38.py --part main          (本番)
出力: rocket/outputs/stage38/(試運転は outputs/stage38/trial/)、表と入力は rocket/data/processed/stage38/
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
from scipy import stats

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
for st in ['stage13', 'stage27', 'stage29']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', st))
import run_stage13 as s13  # noqa: E402
import run_stage27 as s27  # noqa: E402
import run_stage29 as s29  # noqa: E402
from src.evaluation import evaluate, make_splits, split_table  # noqa: E402

s6, s12, s5 = s13.s6, s13.s12, s13.s5
TS = os.path.join(ROCKET_DIR, 'data', 'processed', 'soilmoisture_timeseries_fixed.csv')
D36 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage36')
RAW = os.path.join(ROCKET_DIR, '大豆元データ')
NEW_DATA = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage38')
OUT_MAIN = os.path.join(ROCKET_DIR, 'outputs', 'stage38')
LABEL = 'label_muniyear_median'
K = 3
AMP_MIN = 0.002
RHO_S = 2.6
MAIN = ['NP', 'NP_rise', 'NP_fc', 'NP_phi']
REFS = {'ref_NP_all': ('NP', 'all'), 'ref_NP_phi_bd': ('NP_phi', 'bd')}
BASE = {'B_bd': ['bd'], 'B_th30': ['th30'], 'B_a30': ['a30'], 'B_bd_th30': ['bd', 'th30']}
NAMES = {'NP': 'N_post(基準)', 'NP_rise': 'N_post ÷ イベントの上昇量', 'NP_fc': 'N_post ÷ θ30', 'NP_phi': 'N_post ÷ φ',
         'ref_NP_all': '参考: NP(120件全体)', 'ref_NP_phi_bd': '参考: NP_phi(仮比重がある件)',
         'B_bd': '基準線: 仮比重', 'B_th30': '基準線: θ30', 'B_a30': '基準線: 粗孔隙量 a30', 'B_bd_th30': '基準線: 仮比重 + θ30'}
COL = {'NP': '#2a78d6', 'NP_rise': '#eb6834', 'NP_fc': '#1f9e74', 'NP_phi': '#9b59b6', 'ref_NP_all': '#7aa7e0',
       'ref_NP_phi_bd': '#c39bd3', 'B': '#9a9890'}
INK, MUTED = s6.INK, s6.MUTED
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 元Excel の列(0始まり)。見出しと中身を確認して決めた(D-018)
SOIL_SRC = {
    2016: dict(path=os.path.join(RAW, '2016', '多収プロ', '実態調査（データ）2016（まとめ）2017Aug神戸大.xlsx'),
               sheet='収量土壌分析', skip=7, bd=[44, 45, 46], th30=[47, 48, 49], th5=[50, 51], plow_depth=[63], pd=[]),
    2017: dict(path=os.path.join(RAW, '2017', '多収プロ', '2017実態調査データ（まとめ）180713配布版修正180731.xlsx'),
               sheet='収量土壌分析', skip=7, bd=[47, 48, 49, 50], th30=[51, 52, 53, 54], th5=[55, 56, 57, 58], plow_depth=[79],
               pd=[38]),
    2018: dict(path=os.path.join(RAW, '2018', 'data2018', '2018実証試験調査および管理状況アンケート（集計データ）20190727.xlsx'),
               sheet='2018土壌分析', skip=10, bd=[17], th30=[19], th5=[21], plow_depth=[29], pd=[34]),
}
HEADER_CHECK = {  # (行, 列) → 含むべき文字列(見出しの確認)
    2016: [(3, 44, '仮比重'), (2, 47, '吸引圧30'), (2, 50, '吸引圧5'), (3, 63, '作土層の深さ'), (4, 0, 'サンプル名')],
    2017: [(3, 47, '仮比重'), (2, 51, '吸引圧30'), (2, 55, '吸引圧5'), (3, 79, '作土層の深さ'), (2, 38, '土粒子密度'),
           (4, 0, 'サンプル名')],
    2018: [(8, 17, '仮比重'), (7, 19, '吸引圧30'), (7, 21, '吸引圧5'), (8, 29, '作土層の深さ'), (8, 34, '真比重'),
           (9, 0, '共通コード')],
}


# ---------------------------------------------------------------- 手順1: 土壌物性
def soil_table(log):
    q = pd.read_csv(os.path.join(RAW, 'questionaire_place.csv'), encoding='utf-8-sig')
    rows, issues = [], []
    for yr, c in SOIL_SRC.items():
        hdr = pd.read_excel(c['path'], sheet_name=c['sheet'], header=None, nrows=c['skip'])
        for r, col, word in HEADER_CHECK[yr]:
            ok = word in str(hdr.iat[r, col])
            assert ok, f'{yr} の見出し ({r}, {col}) に「{word}」が無い: {hdr.iat[r, col]}'
        log(f'  {yr}: 見出しの確認 OK({len(HEADER_CHECK[yr])}か所)')
        df = pd.read_excel(c['path'], sheet_name=c['sheet'], header=None, skiprows=c['skip'])
        df = df[df[0].notna()].copy()
        df['place'] = df[0].astype(str).str.strip()
        out = pd.DataFrame({'year': yr, 'place': df.place.values})
        for key in ['bd', 'th30', 'th5', 'plow_depth', 'pd']:
            cols = c[key]
            if not cols:
                out[key] = np.nan
                out[f'{key}_n_rep'] = 0
                continue
            raw = df[cols]
            num = raw.apply(pd.to_numeric, errors='coerce')
            txt = raw.notna() & num.isna()
            for col in cols:
                for i in np.flatnonzero(txt[col].values):
                    issues.append({'year': yr, 'place': df.place.values[i], 'item': key, 'col': col,
                                   'value': str(raw[col].values[i]), 'handling': '数値でない記入 → 欠測'})
            if yr == 2016 and key in ('th30', 'th5'):
                g = df.place.str.match(r'^G\d+$').values
                for i in np.flatnonzero(g & num.notna().any(axis=1).values | g & txt.any(axis=1).values):
                    issues.append({'year': yr, 'place': df.place.values[i], 'item': key, 'col': cols,
                                   'value': ' / '.join(str(v) for v in raw.values[i]),
                                   'handling': '三相分布(気相・液相・固相)が入っている → θ は欠測'})
                num.loc[g, :] = np.nan
            out[key] = num.mean(axis=1).values
            out[f'{key}_n_rep'] = num.notna().sum(axis=1).values
            if key in ('th30', 'th5'):
                out[f'{key}_rep_sd'] = num.std(axis=1, ddof=0).where(num.notna().sum(axis=1) >= 2).values
        if yr == 2018:   # 18列(pF1.5 試料の仮比重)は使わない。記録だけ
            alt = pd.to_numeric(df[18], errors='coerce')
            for i in np.flatnonzero(alt.notna().values):
                issues.append({'year': yr, 'place': df.place.values[i], 'item': 'bd', 'col': 18, 'value': str(alt.values[i]),
                               'handling': '18列(G01〜G10 では pF1.5 試料の仮比重 = 41列)。使わない'})
        rows.append(out)
    s = pd.concat(rows, ignore_index=True)
    dup = s.duplicated(['year', 'place'], keep=False)
    log(f'  同じ (year, place) の重複: {int(dup.sum())} 行')
    assert not dup.any()
    # % → 割合(1.5 未満はすでに割合とみなす)
    for key in ['th30', 'th5']:
        frac = s[key] < 1.5
        for _, r in s[frac].iterrows():
            issues.append({'year': r.year, 'place': r.place, 'item': key, 'col': '', 'value': f'{r[key]:.4f}',
                           'handling': '1.5 未満 → すでに割合とみなし、100で割らない'})
        s[key] = np.where(frac, s[key], s[key] / 100)
        s[f'{key}_rep_sd'] = np.where(frac, s[f'{key}_rep_sd'], s[f'{key}_rep_sd'] / 100)
    s['phi'] = 1 - s.bd / RHO_S
    s['a30'] = s.phi - s.th30
    s['a5'] = s.phi - s.th5
    s['phi_pd'] = 1 - s.bd / s.pd
    s = s.merge(q[['field_id', 'year', 'place']], on=['year', 'place'], how='left')
    log(f'  元Excel の行: {s.groupby("year").size().to_dict()}、questionaire_place.csv で field_id が付かない行 '
        f'{int(s.field_id.isna().sum())}')
    os.makedirs(NEW_DATA, exist_ok=True)
    s.to_csv(os.path.join(NEW_DATA, 'soil_physical.csv'), index=False, encoding='utf-8-sig')
    iss = pd.DataFrame(issues)
    iss.to_csv(os.path.join(NEW_DATA, 'soil_issues.csv'), index=False, encoding='utf-8-sig')
    log(f'  記入の問題 {len(iss)} 件(soil_issues.csv): ' + str(iss.groupby(['year', 'handling']).size().to_dict()))
    return s


def load_cases():
    TH = np.load(os.path.join(D36, 'theta.npy'))
    meta = pd.read_csv(os.path.join(D36, 'meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    keep = meta[LABEL].notna().values
    meta = meta[keep].reset_index(drop=True)
    meta['y'] = meta[LABEL].astype(int)
    for c in ['flower_date', 'seed_date', 'harvest_date']:
        if c in meta:
            meta[c] = pd.to_datetime(meta[c])
    return TH[keep], meta


def part_soil(log, out_dir):
    log('\n==== 手順1: 土壌物性の表 ====')
    s = soil_table(log)
    TH, meta = load_cases()
    s = s[s.field_id.notna()].astype({'field_id': int})
    m = meta.merge(s, on=['field_id', 'year'], how='left', validate='one_to_one')
    assert len(m) == len(meta) == 120
    log(f'  主ラベル {len(m)} 件(多収 {int(m.y.sum())} / 低収 {int((1 - m.y).sum())})のうち、元Excel に行が無い件 '
        f'{int(m.place.isna().sum())}')
    cov = []
    for yr, g in list(m.groupby('year')) + [('全体', m)]:
        r = {'year': yr, 'n': len(g)}
        for key in ['bd', 'th30', 'th5', 'plow_depth', 'pd']:
            r[key] = int(g[key].notna().sum())
        r['bd_and_th30'] = int((g.bd.notna() & g.th30.notna()).sum())
        r['bd_and_th30_hi'] = int((g.bd.notna() & g.th30.notna() & (g.y == 1)).sum())
        r['bd_and_th30_lo'] = int((g.bd.notna() & g.th30.notna() & (g.y == 0)).sum())
        cov.append(r)
    cov = pd.DataFrame(cov)
    cov.to_csv(os.path.join(out_dir, 'soil_coverage.csv'), index=False, encoding='utf-8-sig')
    log('  [主ラベル120件のうち各項目が揃う件数(年別)]')
    log(cov.to_string(index=False))
    log(f'  事前の目安(別のセッション): bd 118、θ30 110 → 今回 bd {int(m.bd.notna().sum())}、θ30 {int(m.th30.notna().sum())}')
    rep = m[['bd_n_rep', 'th30_n_rep', 'th5_n_rep']].apply(lambda c: c.value_counts().sort_index().to_dict())
    log(f'  反復の数(120件、値がある列の数): {rep.to_dict()}')
    sd = m.th30_rep_sd.dropna()
    log(f'  θ30 の反復の間の標準偏差(2列以上ある {len(sd)} 件): 中央値 {sd.median():.4f}、最大 {sd.max():.4f}')
    # 外れ値・物理的な矛盾
    log('  [物理的な矛盾・外れ値(主ラベル120件)]')
    for nm, cond in [('φ < θ30(気相がマイナス)', m.phi < m.th30), ('φ < θ5', m.phi < m.th5),
                     ('θ5 < θ30(吸引圧の弱い方が乾いている)', m.th5 < m.th30), ('bd < 0.5 または > 1.6', (m.bd < 0.5) | (m.bd > 1.6)),
                     ('θ30 < 0.15 または > 0.7', (m.th30 < 0.15) | (m.th30 > 0.7))]:
        sel = m[cond.fillna(False)]
        log(f'    {nm}: {len(sel)} 件' + (('  ' + ', '.join(f'{r.field_id}({r.year} {r.place}: bd {r.bd:.3f} θ30 {r.th30:.3f} '
                                                        f'θ5 {r.th5:.3f} φ {r.phi:.3f})' for r in sel.itertuples()))
                                       if len(sel) else ''))
    for key in ['bd', 'th30', 'phi', 'a30', 'th5', 'plow_depth']:
        v = m[key].dropna()
        log(f'    {key}: n {len(v)}、最小 {v.min():.3f}、5・50・95%点 {np.percentile(v, [5, 50, 95]).round(3).tolist()}、'
            f'最大 {v.max():.3f}')
    pdv = m[m.pd.notna() & m.bd.notna()]
    if len(pdv):
        log(f'  土粒子密度の実測がある件 {len(pdv)}(年 {pdv.year.value_counts().to_dict()}): 実測 {pdv.pd.min():.3f}〜'
            f'{pdv.pd.max():.3f}(中央値 {pdv.pd.median():.3f})。φ(固定2.6)− φ(実測)の差: 平均 '
            f'{(pdv.phi - pdv.phi_pd).mean():+.4f}、範囲 {(pdv.phi - pdv.phi_pd).min():+.4f}〜{(pdv.phi - pdv.phi_pd).max():+.4f}、'
            f'Spearman {stats.spearmanr(pdv.phi, pdv.phi_pd)[0]:.3f}')
    # 層内の分散の割合
    log('  [層(市町村×年)内の分散が全体の分散に占める割合(主ラベル120件、値がある件)]')
    rows = []
    for key in ['bd', 'th30', 'phi', 'a30', 'th5', 'plow_depth']:
        d = m[[key, 'muniyear']].dropna()
        tot = ((d[key] - d[key].mean()) ** 2).sum()
        within = ((d[key] - d.groupby('muniyear')[key].transform('mean')) ** 2).sum()
        rows.append({'item': key, 'n': len(d), 'n_strata': d.muniyear.nunique(), 'within_share': within / tot})
        log(f'    {key}: 層内 {within / tot:.3f}(n {len(d)}、{d.muniyear.nunique()} 層)')
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'soil_within_share.csv'), index=False, encoding='utf-8-sig')
    # 解析集合と層の釣り合い
    both = m.bd.notna() & m.th30.notna()
    log(f'\n  主の解析集合(bd と θ30 の両方): {int(both.sum())} 件(多収 {int(m.y[both].sum())} / 低収 '
        f'{int((1 - m.y[both]).sum())})・{m.muniyear[both].nunique()} 集団・{m.muniCd[both].nunique()} 市町村')
    g0 = m.groupby('muniyear').y.agg(n_all='size', hi_all='sum')
    g1 = m[both].groupby('muniyear').y.agg(n_main='size', hi_main='sum')
    g = g0.join(g1, how='left').fillna(0).astype(int)
    g['lo_all'] = g.n_all - g.hi_all
    g['lo_main'] = g.n_main - g.hi_main
    ch = g[(g.n_all != g.n_main)]
    log('  件数が減った層(多収・低収の数、全体 → 主):')
    for my, r in ch.iterrows():
        log(f'    {my}: {r.hi_all}/{r.lo_all} → {r.hi_main}/{r.lo_main}')
    unb = g[(g.n_main > 0) & (g.hi_main != g.lo_main)]
    log(f'  主の集合で多収と低収が同数でない層: ' + ', '.join(f'{my} {r.hi_main}/{r.lo_main}' for my, r in unb.iterrows()))
    g.reset_index().to_csv(os.path.join(out_dir, 'strata_balance.csv'), index=False, encoding='utf-8-sig')
    m.to_csv(os.path.join(NEW_DATA, 'cases_soil.csv'), index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------- 手順2: 入力
def theta_ext(TH, meta):
    """θ(−3) … θ(59)(63点)。θ(−1)〜θ(59) は段階36の theta.npy、θ(−3)・θ(−2) は fixed 版から。"""
    ts = pd.read_csv(TS, encoding='utf-8-sig')
    ts = ts[ts.field_id != 515]
    ts['date'] = pd.to_datetime(ts.date)
    ser = {k: g.set_index('date').VWC for k, g in ts.groupby(['field_id', 'year'])}
    pre = np.full((len(meta), 3), np.nan)
    for i, r in meta.iterrows():
        s = ser[(r.field_id, r.year)]
        for j, dd in enumerate([-3, -2, -1]):
            pre[i, j] = s.get(r.flower_date + pd.Timedelta(days=dd), np.nan)
    ok = np.allclose(pre[:, 2], TH[:, 0], equal_nan=True)
    ok2 = all(np.allclose(ser[(r.field_id, r.year)].reindex(pd.date_range(r.flower_date, periods=60)).values, TH[i, 1:])
              for i, r in meta.iterrows())
    return np.concatenate([pre[:, :2], TH], axis=1), ok and ok2


def build_inputs(TH, meta, soil, log):
    th = theta_ext(TH, meta)
    th, ok = th
    log(f'  確認 fixed 版から読んだ θ(−1)・窓の60日が theta.npy と一致: {ok}')
    assert ok
    X = th[:, 3:][:, None, :]                       # θ(0)〜θ(59)
    pre = th[:, 1:3 + 1] - th[:, 0:3]              # Δθ(−2), Δθ(−1), Δθ(0)
    comps, rain, post = s29.components(X, pre, K)
    NP = comps['N_post'][:, 0, :]
    n_pre_nan = np.isnan(pre).sum(axis=1)
    log(f'  開花日より前の Δθ(t = −2〜0)が無い件: 一部 {int(((n_pre_nan > 0) & (n_pre_nan < 3)).sum())}・全部 '
        f'{int((n_pre_nan == 3).sum())}(雨の日なしとして扱う、段階29と同じ)')
    # 雨のイベント(t = −2〜59)と NP_rise
    dext = np.concatenate([np.nan_to_num(pre, nan=0.0), np.diff(th[:, 3:], axis=1)], axis=1)   # t = −2 … 59
    rain_ext = dext > 0
    assert np.array_equal(rain_ext[:, 3:], rain)
    n, T = NP.shape
    NPr = np.zeros_like(NP)
    ev_rows, small_days, small_nz, small_ev, s_trunc = [], 0, 0, 0, 0
    feat = []
    for i in range(n):
        runs = s29_runs(rain_ext[i])                  # (s, e) は rain_ext の位置(t = 位置 − 2)
        ev_vals, ev_rise, ev_rate = [], [], []
        for s_, e_ in runs:
            ts_, te_ = s_ - 2, e_ - 2                 # 窓の日付
            days = []
            for t in range(te_ + 1, te_ + K + 1):     # 雨の直後の日(次の雨で打ち切り)
                if t > 59 or rain_ext[i, t + 2]:
                    break
                if t >= 1:
                    days.append(t)
            if not days:
                continue
            rise = th[i, te_ + 3] - th[i, ts_ - 1 + 3]
            if ts_ == -2:
                s_trunc += 1
            vals = NP[i, [t - 1 for t in days]]
            usable = np.isfinite(rise) and rise >= AMP_MIN
            if usable:
                NPr[i, [t - 1 for t in days]] = vals / rise
                ev_rate.append(vals.mean() / rise)
            else:
                small_ev += 1
                small_days += len(days)
                small_nz += int((vals != 0).sum())
            ev_vals.append(vals.mean())
            ev_rise.append(rise)
            ev_rows.append({'field_id': meta.field_id[i], 'year': meta.year[i], 't_start': ts_, 't_end': te_,
                            'n_post_days': len(days), 'rise': rise, 'np_mean': vals.mean(), 'usable_rise': usable})
        feat.append({'NP_ev': np.mean(ev_vals) if ev_vals else np.nan, 'rise_ev': np.mean(ev_rise) if ev_rise else np.nan,
                     'NP_rise_ev': np.mean(ev_rate) if ev_rate else np.nan, 'n_events': len(ev_vals),
                     'n_events_rise_ok': len(ev_rate)})
    # 確認: イベントごとに割り当てた日 = 段階29の雨の直後の日
    ok_days = True
    for i in range(n):
        er = [r for r in ev_rows if r['field_id'] == meta.field_id[i] and r['year'] == meta.year[i]]
        cnt = sum(r['n_post_days'] for r in er)
        ok_days &= cnt == int(post[i].sum())
    log(f'  確認 イベントに割り当てた雨の直後の日数の合計 = 段階29の定義の雨の直後の日数(全件): {ok_days}')
    assert ok_days
    ev = pd.DataFrame(ev_rows)
    log(f'  雨のイベント(雨の直後の日が窓の中にあるもの): {len(ev)} 個(1件あたり平均 {len(ev) / n:.1f})。最初の雨の日が t = −2'
        f'(それより前は見ていない)のもの {s_trunc} 個')
    log(f'  上昇量 < {AMP_MIN} のイベント: {small_ev} 個 → その後の雨の直後の日 {small_days} 日(うち NP が0でない日 {small_nz} 日)を'
        f' NP_rise で0にした(全 {int(post.sum())} 日・NP が0でない日 {int((NP != 0).sum())} 日のうち)')
    log(f'  上昇量の分布(全イベント): 5・25・50・75・95%点 {np.percentile(ev.rise, [5, 25, 50, 75, 95]).round(4).tolist()}')
    feat = pd.DataFrame(feat)
    nz = NPr[NPr != 0]
    log(f'  NP_rise の0でない値の分布: 1・5・50・95%点 {np.percentile(nz, [1, 5, 50, 95]).round(3).tolist()}、最小 {nz.min():.3f}。'
        f'−1 未満(1日で上昇量より大きく減る)の値 {int((nz < -1).sum())} 個・{int((NPr < -1).any(axis=1).sum())} 件(設定どおり残す)')
    soil = soil[soil.field_id.notna()].astype({'field_id': int})
    m = meta.merge(soil[['field_id', 'year', 'place', 'bd', 'th30', 'th5', 'phi', 'a30', 'plow_depth']], on=['field_id', 'year'],
                   how='left', validate='one_to_one')
    fc = (NP / m.th30.values[:, None])
    ph = (NP / m.phi.values[:, None])
    P = np.maximum(np.diff(th[:, 3:], axis=1), 0.0)
    inputs = {'NP': NP, 'NP_rise': NPr, 'NP_fc': fc, 'NP_phi': ph, 'P': P}
    feat = pd.concat([m[['field_id', 'year', 'muniCd', 'muniyear', 'y', 'place', 'bd', 'th30', 'th5', 'phi', 'a30', 'plow_depth']],
                      feat], axis=1)
    feat['NP_fc_ev'] = feat.NP_ev / feat.th30
    feat['NP_phi_ev'] = feat.NP_ev / feat.phi
    feat['P_mean'] = P.mean(axis=1)
    feat['NP_mean59'] = NP.mean(axis=1)
    f36 = pd.read_csv(os.path.join(ROCKET_DIR, 'outputs', 'stage36', 'features.csv'), encoding='utf-8-sig')
    keep36 = [c for c in f36.columns if c in ('N_post_k3|θ0', 'rate_k3|θ0')]
    feat = feat.merge(f36[['field_id', 'year'] + keep36], on=['field_id', 'year'], how='left')
    for c in keep36:
        mine = 'NP_ev' if c.startswith('N_post') else 'NP_rise_ev'
        log(f'  確認(参考) 段階36の {c} と今回の {mine} の Spearman: {stats.spearmanr(feat[c], feat[mine], nan_policy="omit")[0]:.4f}'
            f'(段階36は雨の日を窓の0〜59日目だけで判定し、t = −2・−1 を見ないので完全には一致しない)')
    os.makedirs(NEW_DATA, exist_ok=True)
    for k, v in inputs.items():
        np.save(os.path.join(NEW_DATA, f'{k}_X.npy'), v[:, None, :])
    m.to_csv(os.path.join(NEW_DATA, 'meta.csv'), index=False, encoding='utf-8-sig')
    ev.to_csv(os.path.join(NEW_DATA, 'rain_events.csv'), index=False, encoding='utf-8-sig')
    feat.to_csv(os.path.join(NEW_DATA, 'field_summaries.csv'), index=False, encoding='utf-8-sig')
    return inputs, m, feat


def s29_runs(rain):
    out, t, T = [], 0, len(rain)
    while t < T:
        if rain[t]:
            s = t
            while t + 1 < T and rain[t + 1]:
                t += 1
            out.append((s, t))
        t += 1
    return out


def part_inputs(log, out_dir):
    log('\n==== 手順2: 入力系列 ====')
    TH, meta = load_cases()
    soil = pd.read_csv(os.path.join(NEW_DATA, 'soil_physical.csv'), encoding='utf-8-sig')
    inputs, m, feat = build_inputs(TH, meta, soil, log)
    for k, v in inputs.items():
        fin = np.isfinite(v).all(axis=1)
        log(f'  {k}: 有限な件 {int(fin.sum())}/{len(v)}、0でない日の割合 {np.mean(v[fin] != 0):.3f}、'
            f'値の範囲 {np.nanmin(v):.4f}〜{np.nanmax(v):.4f}')
    main = m.bd.notna() & m.th30.notna()
    cols = ['NP_ev', 'rise_ev', 'NP_rise_ev', 'NP_fc_ev', 'NP_phi_ev', 'P_mean', 'NP_mean59', 'n_events', 'bd', 'th30', 'phi',
            'a30']
    f = feat[main.values]
    rho = f[cols].corr(method='spearman')
    # 層内の正規化順位どうしの相関(段階36の S2 と同じ考え方)
    u = f[cols].groupby(f.muniyear.values).transform(lambda c: (c.rank() - 0.5) / c.notna().sum())
    rho_w = u.corr()
    out = pd.concat([rho.assign(kind='Spearman(全件)'), rho_w.assign(kind='層内の正規化順位の Pearson')])
    out.index.name = 'feature'
    out.to_csv(os.path.join(out_dir, 'feature_corr.csv'), encoding='utf-8-sig')
    log(f'  [圃場ごとの要約の相関(主の集合 {int(main.sum())} 件)]  上昇量 rise_ev との相関:')
    for c in ['NP_ev', 'NP_rise_ev', 'NP_fc_ev', 'NP_phi_ev', 'P_mean', 'th30', 'phi', 'bd', 'a30']:
        log(f'    {c:11s}: Spearman {rho.loc[c, "rise_ev"]:+.3f}、層内 {rho_w.loc[c, "rise_ev"]:+.3f}')
    log(f'    NP_ev と NP_fc_ev {rho.loc["NP_ev", "NP_fc_ev"]:+.3f}、NP_ev と NP_phi_ev {rho.loc["NP_ev", "NP_phi_ev"]:+.3f}、'
        f'NP_ev と NP_rise_ev {rho.loc["NP_ev", "NP_rise_ev"]:+.3f}、NP_ev と P_mean {rho.loc["NP_ev", "P_mean"]:+.3f}')
    plot_scatter(f, out_dir)


def plot_scatter(f, out_dir):
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, (c, nm) in zip(axs, [('NP_ev', 'NP(基準)'), ('NP_rise_ev', 'NP ÷ 上昇量'), ('NP_fc_ev', 'NP ÷ θ30'),
                                 ('NP_phi_ev', 'NP ÷ φ')]):
        for yy, col, lab in [(1, '#2a78d6', '多収'), (0, '#eb6834', '低収')]:
            g = f[f.y == yy]
            ax.scatter(g.rise_ev, g[c], s=16, color=col, alpha=0.7, label=lab, lw=0)
        r = stats.spearmanr(f.rise_ev, f[c], nan_policy='omit')[0]
        ax.set_title(f'{nm}(Spearman {r:+.2f})', loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('雨のイベントの上昇量の平均(圃場ごと)')
        s6.style(ax)
    axs[0].set_ylabel('雨の直後の減少(イベント平均、圃場ごと)')
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f'段階38: 尺度合わせの前後の要約と上昇量({len(f)} 件、主の集合)', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out_dir, 'fig_scatter_rise.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 手順3: 評価
def subset(meta, X, mask):
    idx = np.flatnonzero(mask)
    return meta.iloc[idx].reset_index(drop=True), X[idx]


def run_baseline(name, F, meta, sp, P, n_seeds, log, out_dir):
    y = meta.y.values
    ps = s13.perm_sets('median', meta, P)
    t0 = time.time()
    decs = s12.lr_decisions(F, y, sp, ps[0])
    obs, null = s13.analyse(decs, meta, sp, ps)
    ref = evaluate(s5.lr(), F, y, sp)
    s13.check_obs(name, obs, ref, log)
    np.save(os.path.join(out_dir, f'null_{name}.npy'), null['all'])
    S = np.full((s6.N_REPEATS, len(y)), np.nan)
    for (r, f, tr, te), d in zip(sp, decs):
        S[r, te] = d[0]
    summ = []
    for met, j, v in [('all_auc', 1, ref.auc.mean()), ('all_bacc', 0, ref.bacc.mean())]:
        nl = null['all'][:, j]
        summ.append({'config': name, 'metric': met, 'n': len(y), 'n_seeds': 1, 'n_perm': P, 'mean': v, 'seed_min': v,
                     'seed_max': v, 'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                     'p': (1 + np.sum(nl >= v)) / (1 + P)})
    log(f'  {name}: 全体 AUC {summ[0]["mean"]:.3f}(q95 {summ[0]["null_q95"]:.3f}、p = {summ[0]["p"]:.4f})  bacc '
        f'{summ[1]["mean"]:.3f}  [{time.time() - t0:.0f}s]')
    return {'summary': summ, 'S': np.repeat(S[None], n_seeds, axis=0), 'folds': ref.assign(config=name)}


def run_mr(name, X, meta, sp, seeds, P, log, out_dir):
    part = os.path.join(out_dir, f'partial_{name}.pkl')
    tag = {'seeds': seeds, 'P': P, 'n': len(meta), 'xsum': float(np.round(np.abs(X).sum(), 8))}
    if os.path.exists(part):
        with open(part, 'rb') as fh:
            saved = pickle.load(fh)
        if saved['tag'] == tag:
            log(f'\n[{name}: {NAMES[name]}] 途中結果を読み込んだ')
            return saved['out']
    log(f'\n[{name}: {NAMES[name]}]  {len(meta)} 件')
    s27.OUT_DIR = out_dir
    out = s27.run_input(name, X, meta, sp, seeds, P, log)
    with open(part, 'wb') as fh:
        pickle.dump({'tag': tag, 'out': out}, fh)
    return out


def check_splits(meta, sp, log, tag):
    t = split_table(sp, meta.y.values, meta.muniCd.values)
    log(f'  分割({tag}): {len(sp)} fold、テスト件数 {t.n_test.min()}〜{t.n_test.max()}、テストの多収の割合 '
        f'{t.test_pos_ratio.min():.2f}〜{t.test_pos_ratio.max():.2f}、片方のクラスだけの fold '
        f'{int(((t.n_test_pos == 0) | (t.n_test_neg == 0)).sum())}')
    grp_ok = all(len(set(meta.muniCd.values[tr]) & set(meta.muniCd.values[te])) == 0 for r, f, tr, te in sp)
    log(f'  確認 学習とテストで市町村が重ならない: {grp_ok}')
    assert grp_ok


def part_eval(log, out_dir, seeds, P_mr, P_lr, n_boot):
    log(f'\n==== 手順3: 評価(乱数 {seeds}、置換 MiniRocket {P_mr} 回・基準線 {P_lr} 回、ブートストラップ {n_boot} 回) ====')
    s6.N_JOBS = s13.N_JOBS
    s12.N_JOBS = s13.N_JOBS
    meta = pd.read_csv(os.path.join(NEW_DATA, 'meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    Xall = {k: np.load(os.path.join(NEW_DATA, f'{k}_X.npy')) for k in MAIN}
    main = (meta.bd.notna() & meta.th30.notna()).values
    mm, _ = subset(meta, Xall['NP'], main)
    log(f'  主の集合: {len(mm)} 件(多収 {int(mm.y.sum())} / 低収 {int((1 - mm.y).sum())})・{mm.muniyear.nunique()} 集団・'
        f'{mm.muniCd.nunique()} 市町村')
    X = {k: v[main] for k, v in Xall.items()}
    for k, v in X.items():
        assert np.isfinite(v).all(), k
    sp = make_splits(mm.y.values, mm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
    check_splits(mm, sp, log, '主')
    log('  確認(NP): 段階13の prechecks(Pipeline と特徴量の使い回し、置換で分割が変わらない、まとめて解いた置換 = sklearn)')
    s13.prechecks('median', X['NP'], mm, sp, log)
    res = {}
    for nm in MAIN:
        res[nm] = run_mr(nm, X[nm], mm, sp, seeds, P_mr, log, out_dir)
    log('\n[基準線(土の物性だけ、StandardScaler → LogisticRegression C = 1)]')
    for nm, cols in BASE.items():
        res[nm] = run_baseline(nm, mm[cols].to_numpy(float), mm, sp, P_lr, len(seeds), log, out_dir)
    # 参考
    ref_meta = {}
    for nm, (inp, which) in REFS.items():
        mask = np.ones(len(meta), bool) if which == 'all' else meta.bd.notna().values
        rm, rx = subset(meta, Xall[inp], mask)
        assert np.isfinite(rx).all()
        rsp = make_splits(rm.y.values, rm.muniCd.values, s6.N_SPLITS, s6.N_REPEATS, s6.SEED0)
        log(f'\n  参考 {nm}: {len(rm)} 件(多収 {int(rm.y.sum())} / 低収 {int((1 - rm.y).sum())})')
        check_splits(rm, rsp, log, nm)
        res[nm] = run_mr(nm, rx, rm, rsp, seeds, P_mr, log, out_dir)
        ref_meta[nm] = rm
    # まとめ
    rows = []
    for nm in MAIN + list(REFS) + list(BASE):
        for r in res[nm]['summary']:
            if r['metric'] in ('all_auc', 'all_bacc'):
                rows.append({k: r.get(k) for k in ['config', 'metric', 'n', 'n_seeds', 'n_perm', 'mean', 'seed_min', 'seed_max',
                                                   'null_mean', 'null_q95', 'p', 'alpha_at_max']})
    sm = pd.DataFrame(rows)
    sm['role'] = sm.config.map(lambda c: '主' if c in MAIN else ('参考' if c in REFS else '基準線'))
    sm['name'] = sm.config.map(NAMES)
    m = (sm.metric == 'all_auc') & sm.config.isin(MAIN)
    pv = sm[m].set_index('config').p.reindex(MAIN)
    adj = pd.Series(s27.holm(pv.values), index=MAIN)
    sm.loc[m, 'p_holm'] = sm.loc[m, 'config'].map(adj)
    sm['above_q95'] = sm['mean'] > sm.null_q95
    wide = sm.pivot_table(index='config', columns='metric', values=['mean', 'null_q95', 'p'], aggfunc='first')
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.rename(columns={'mean_all_auc': 'auc', 'mean_all_bacc': 'bacc', 'null_q95_all_auc': 'auc_null_q95',
                                'null_q95_all_bacc': 'bacc_null_q95', 'p_all_auc': 'p_auc', 'p_all_bacc': 'p_bacc'})
    a = sm[sm.metric == 'all_auc'].set_index('config')
    wide = wide.join(a[['role', 'name', 'n', 'n_seeds', 'n_perm', 'seed_min', 'seed_max', 'null_mean', 'p_holm', 'alpha_at_max']])
    wide = wide.reindex(MAIN + list(BASE) + list(REFS)).reset_index()
    wide = wide[['config', 'name', 'role', 'n', 'n_seeds', 'n_perm', 'auc', 'seed_min', 'seed_max', 'null_mean', 'auc_null_q95',
                 'p_auc', 'p_holm', 'bacc', 'bacc_null_q95', 'p_bacc', 'alpha_at_max']]
    wide.to_csv(os.path.join(out_dir, 'auc_summary.csv'), index=False, encoding='utf-8-sig')
    # 差(主の集合、同じ分割)
    best_b = a.loc[list(BASE), 'mean'].idxmax()
    Cs, N0 = {}, None
    for nm in MAIN + [best_b]:
        C, Nn = s27.pair_sums(res[nm]['S'], mm, sp)
        N0 = Nn if N0 is None else N0
        assert np.array_equal(Nn, N0)
        Cs[nm] = C
    boot, pooled = s27.boot_diff(Cs, N0, mm, n_boot)
    boot.to_csv(os.path.join(out_dir, 'bootstrap_auc.csv'), index=False, encoding='utf-8-sig')
    fm = {nm: a.loc[nm, 'mean'] for nm in MAIN + [best_b]}
    drows = []
    for x, yv in [('NP_rise', 'NP'), ('NP_fc', 'NP'), ('NP_phi', 'NP')] + [(nm, best_b) for nm in MAIN]:
        d = boot[x] - boot[yv]
        drows.append({'comparison': f'{x} − {yv}', 'point_fold_mean': fm[x] - fm[yv], 'point_pooled_pairs': pooled[x] - pooled[yv],
                      'boot_q025': d.quantile(.025), 'boot_q975': d.quantile(.975), 'boot_frac_pos': (d > 0).mean(),
                      'n_boot': n_boot, 'note': '最良の基準線(結果を見て選択、記述)' if yv == best_b else '主'})
    diffs = pd.DataFrame(drows)
    diffs.to_csv(os.path.join(out_dir, 'diff_bootstrap.csv'), index=False, encoding='utf-8-sig')
    pd.concat([pd.concat(res[nm]['folds']) if isinstance(res[nm]['folds'], list) else res[nm]['folds']
               for nm in MAIN + list(BASE) + list(REFS)]).to_csv(os.path.join(out_dir, 'folds.csv'), index=False,
                                                                 encoding='utf-8-sig')
    pd.concat([pd.DataFrame(res[nm]['seed']) for nm in MAIN + list(REFS)]).to_csv(os.path.join(out_dir, 'per_seed.csv'),
                                                                               index=False, encoding='utf-8-sig')
    log('\n[全体 AUC のまとめ]')
    log(wide.drop(columns=['name']).round(4).to_string(index=False))
    log(f'\n[差(市町村単位のクラスターブートストラップ {n_boot} 回)]  最良の基準線 = {best_b}')
    log(diffs.round(4).to_string(index=False))
    plot_auc(wide, out_dir)
    return wide, diffs


def plot_auc(wide, out_dir):
    cfgs = MAIN + list(BASE) + list(REFS)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, c in enumerate(cfgs):
        nl = np.load(os.path.join(out_dir, f'null_{c}.npy'))[:, 1]
        r = wide.set_index('config').loc[c]
        vp = ax.violinplot([nl[np.isfinite(nl)]], positions=[i], widths=0.7, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor('#d8d6cf')
            b.set_alpha(0.8)
        ax.hlines(r.auc_null_q95, i - 0.3, i + 0.3, color=s6.C_RED, lw=2, ls='--')
        col = COL.get(c, COL['B'])
        ax.scatter(i, r.auc, s=70, color=col, zorder=3)
        lab = f'{r.auc:.3f}\np={r.p_auc:.3f}'
        if c in MAIN:
            lab += f'\nHolm {r.p_holm:.3f}'
        ax.text(i + 0.12, r.auc + 0.01, lab, fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=':')
    ax.set_ylim(0.28, 0.78)
    ax.axvline(len(MAIN) - 0.5, color=MUTED, lw=0.6)
    ax.axvline(len(MAIN) + len(BASE) - 0.5, color=MUTED, lw=0.6)
    ax.set_xticks(range(len(cfgs)), [c.replace('ref_', '参考\n') for c in cfgs], fontsize=9)
    ax.set_ylabel('全体 AUC')
    ax.set_title('点 = 観測(乱数平均)、灰 = 置換の帰無分布、赤の破線 = 95%点。左から主の4入力 / 基準線(主の集合) / 参考',
                 loc='left', fontsize=10, color=INK)
    s6.style(ax)
    fig.suptitle('段階38: 尺度合わせした N_post 入力の全体 AUC と帰無分布', fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, 'fig_auc_null.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='+', default=['soil', 'inputs'], choices=['soil', 'inputs', 'trial', 'main'])
    args = ap.parse_args()
    trial = 'trial' in args.part
    out_dir = os.path.join(OUT_MAIN, 'trial') if trial else OUT_MAIN
    os.makedirs(out_dir, exist_ok=True)
    # 本番は準備(soil・inputs)のログの後ろに書き足す
    append = 'main' in args.part and 'soil' not in args.part
    logf = open(os.path.join(out_dir, 'run_log.txt'), 'a' if append else 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'段階38({" ".join(args.part)})  並列数 {s13.N_JOBS}  設定は D-018')
    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 EXP-047 が docs/experiments.md に無い(段階38・EXP-047 が未使用): {"## EXP-047" not in docs}')
    if 'soil' in args.part:
        part_soil(log, OUT_MAIN)
    if 'inputs' in args.part:
        part_inputs(log, OUT_MAIN)
    if trial:
        part_eval(log, out_dir, [0, 1], 20, 20, 100)
    if 'main' in args.part:
        part_eval(log, out_dir, list(range(10)), 200, 1000, 2000)
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {out_dir}')
    logf.close()


if __name__ == '__main__':
    main()
