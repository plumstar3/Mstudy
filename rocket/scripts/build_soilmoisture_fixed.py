"""
旧の日単位データ data/processed/soilmoisture_timeseries.csv(DB の SolidMoisture)の値の誤りだけを直した
data/processed/soilmoisture_timeseries_fixed.csv を作る(D-013、docs/data_issues.md の4)。元のファイルは変更しない。

直す内容(元Excel との照合で見つかった値の誤りだけ):
  1. field_id 362(K1・2017): 2017-10-09 以降の行を削除(K1 の元データは 10-08 で終わる。10-09〜11-30 は J11 の値が混入)
  2. field_id 607・609(P7・P8・2017): 2017-10-21 以降の行を削除(元データは 10-20 14:00 で終わる)
  3. field_id 42(B9・2017): 元Excel の K列(大豆元データ/soilmoisture_raw_long.csv の series_id 213)から作った日平均に置き換える。
     作り方は DB の処理の再現(大豆元データ/README_soilmoisture_raw.md、42 の推定手順は「interp 1h; daily mean」)に合わせる:
       %VWC を100で割る → 丸めない(2017年)→ 負値は残す(この系列には無い)→ 1時間間隔に線形補間 → 暦日で単純平均。
       v2 の「24時間そろう日だけ」は使わない。日付は旧ファイルにある行だけを置き換え、dap はそのまま。
  4. field_id 515: ファイルには残す(使う側で読み込み後に除く。従来どおり)
直さないもの: 585・94・103 の欠けた期間、欠測日の補間、2016年の丸め(旧データのまま)。

実行: rocket\\venv_rocket\\Scripts\\python rocket\\scripts\\build_soilmoisture_fixed.py
出力: rocket/data/processed/soilmoisture_timeseries_fixed.csv、rocket/outputs/soilmoisture_fixed/(changes.csv、run_log.txt)
"""
import os
import sys

import numpy as np
import pandas as pd

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROCKET_DIR, 'data', 'processed', 'soilmoisture_timeseries.csv')
DST = os.path.join(ROCKET_DIR, 'data', 'processed', 'soilmoisture_timeseries_fixed.csv')
RAW = os.path.join(ROCKET_DIR, '大豆元データ', 'soilmoisture_raw_long.csv')
CAT = os.path.join(ROCKET_DIR, '大豆元データ', 'soilmoisture_series_catalog.csv')
V2 = os.path.join(ROCKET_DIR, '大豆元データ', 'soilmoisture_daily_v2.csv')
LAB = os.path.join(ROCKET_DIR, 'data', 'processed', 'field_year_labeled.csv')
OUT = os.path.join(ROCKET_DIR, 'outputs', 'soilmoisture_fixed')
CUTS = [(362, 2017, '2017-10-09'), (607, 2017, '2017-10-21'), (609, 2017, '2017-10-21')]
B9 = (42, 2017, 213)
LABEL = 'label_muniyear_median'


def daily_from_raw(series_id):
    raw = pd.read_csv(RAW, encoding='utf-8-sig', dtype={'raw_text': str})
    cat = pd.read_csv(CAT, encoding='utf-8-sig')
    unit = cat.loc[cat.series_id == series_id, 'unit'].iloc[0]
    r = raw[raw.series_id == series_id].copy()
    r['datetime'] = pd.to_datetime(r.datetime)
    r = r.dropna(subset=['value']).sort_values('datetime').set_index('datetime').value.astype(float)
    if unit == '%VWC':
        r = r / 100.0
    grid = pd.date_range(r.index.min(), r.index.max(), freq='1h')
    h = r.reindex(r.index.union(grid)).interpolate(method='time').reindex(grid)   # 1時間間隔に線形補間
    d = h.groupby(h.index.normalize()).mean()                                      # 暦日で単純平均
    return d, unit, r, h


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'run_log.txt'), 'w', encoding='utf-8')

    def log(m):
        print(m)
        logf.write(m + '\n')

    old = pd.read_csv(SRC, encoding='utf-8-sig')
    log(f'元のファイル: {SRC}({len(old)} 行、列 {list(old.columns)})')
    new = old.copy()
    dates = pd.to_datetime(new.date)
    lab = pd.read_csv(LAB, encoding='utf-8-sig', dtype={'muniCd': str})
    lab['flower_date'] = pd.to_datetime(lab.flower_date)
    rows = []

    def window(fid, yr):
        r = lab[(lab.field_id == fid) & (lab.year == yr)].iloc[0]
        return r.flower_date, r.flower_date + pd.Timedelta(days=59), r[LABEL]

    # 1・2 削除
    drop = np.zeros(len(new), bool)
    for fid, yr, cut in CUTS:
        m = (new.field_id == fid) & (new.year == yr) & (dates >= pd.Timestamp(cut))
        ws, we, lb = window(fid, yr)
        dd = dates[m]
        inwin = int(((dd >= ws) & (dd <= we)).sum())
        rows.append({'field_id': fid, 'year': yr, 'action': f'{cut} 以降を削除', 'rows_deleted': int(m.sum()),
                     'rows_changed': 0, 'first_deleted': dd.min().date(), 'last_deleted': dd.max().date(),
                     'window': f'{ws.date()}〜{we.date()}', 'window_days_affected': inwin, LABEL: lb})
        log(f'{fid}・{yr}: {cut} 以降の {int(m.sum())} 行を削除({dd.min().date()}〜{dd.max().date()})。'
            f'窓 {ws.date()}〜{we.date()} のうち {inwin} 日が無くなる')
        drop |= m.to_numpy()

    # 3 field_id 42 の置き換え
    fid, yr, sid = B9
    d, unit, r, h = daily_from_raw(sid)
    log(f'\n{fid}・{yr}: series_id {sid}(単位 {unit})、{len(r)} 値、{r.index.min()}〜{r.index.max()}、'
        f'負値 {int((r < 0).sum())}、間隔 {r.index.to_series().diff().value_counts().to_dict()}。1時間格子 {len(h)} 点')
    m = ((new.field_id == fid) & (new.year == yr)).to_numpy()
    idx = np.flatnonzero(m)
    dd = dates[m].dt.normalize()
    newv = d.reindex(dd.values).to_numpy()
    if np.isnan(newv).any():
        raise RuntimeError(f'元データに無い日がある: {dd[np.isnan(newv)].tolist()}')
    oldv = new.loc[idx, 'VWC'].to_numpy()
    diff = np.abs(newv - oldv)
    ws, we, lb = window(fid, yr)
    inw = ((dd >= ws) & (dd <= we)).to_numpy()
    changed = diff > 1e-12
    new.loc[idx, 'VWC'] = newv
    log(f'  旧ファイルの {len(idx)} 行({dd.min().date()}〜{dd.max().date()})を置き換え。値が変わった行 {int(changed.sum())}、'
        f'旧との MAE {diff.mean():.5f}(最大 {diff.max():.5f})。窓 {ws.date()}〜{we.date()} の中で変わった日 '
        f'{int((changed & inw).sum())} / {int(inw.sum())}')
    # 不一致の原因の確認: 旧の日付 D の値が、元データの D − 1 日の日平均と一致するか(1日のずれ)
    for s in (1, -1):
        sh = pd.Series(oldv, index=dd.values).shift(-s, freq='D').reindex(d.index)
        ok_ = sh.notna()
        dd_ = (sh[ok_] - d[ok_]).abs()
        log(f'  旧を {s:+d} 日ずらして比べる(旧の日付 D + {s} の値 ↔ 元データの D): 比べられる日 {int(ok_.sum())}、'
            f'完全一致(< 1e-12)の割合 {(dd_ < 1e-12).mean():.3f}、MAE {dd_.mean():.2e}')
    extra = sorted(set(d.index.date) - set(dd.dt.date))
    log(f'  元データにあって旧ファイルに無い日(追加しない): {extra}')
    hours = h.groupby(h.index.normalize()).size().reindex(dd.values).to_numpy()
    log(f'  1日の時間数が24でない日: ' + ', '.join(f'{x.date()}({n})' for x, n in zip(dd, hours) if n != 24))
    rows.append({'field_id': fid, 'year': yr, 'action': f'元Excel K列(series {sid})の日平均に置き換え', 'rows_deleted': 0,
                 'rows_changed': int(changed.sum()), 'first_deleted': None, 'last_deleted': None,
                 'window': f'{ws.date()}〜{we.date()}', 'window_days_affected': int((changed & inw).sum()), LABEL: lb})
    # 確認: v2 日平均(同じ K列、24時間そろう日)と一致するか
    v2 = pd.read_csv(V2, encoding='utf-8-sig')
    v2 = v2[(v2.field_id == fid) & (v2.year == yr) & (v2.n_hours == 24)].copy()
    v2['date'] = pd.to_datetime(v2.date)
    cmp_ = pd.Series(newv, index=dd.values).reindex(v2.date.values)
    ok = cmp_.notna()
    log(f'  確認: v2 日平均(24時間そろう日)と比べられる日 {int(ok.sum())}、最大差 '
        f'{np.abs(cmp_[ok].to_numpy() - v2.VWC.to_numpy()[ok.to_numpy()]).max():.2e}')

    new = new.loc[~drop].reset_index(drop=True)
    log(f'\n515: 行はそのまま残す(使う側で除く)。515 の行数 {int((new.field_id == 515).sum())}')
    new.to_csv(DST, index=False, encoding='utf-8-sig')
    log(f'\n出力: {DST}({len(new)} 行、削除 {int(drop.sum())} 行)')

    # 確認: 直していない圃場は元と完全に同じ
    key = ['field_id', 'year', 'date']
    a = old.merge(new, on=key, how='left', suffixes=('', '_new'), indicator=True)
    touched = a.field_id.isin([362, 607, 609, 42])
    same = (a.loc[~touched, 'VWC'] == a.loc[~touched, 'VWC_new']).all() and (a.loc[~touched, 'dap'] == a.loc[~touched, 'dap_new']).all()
    log(f'確認: 直していない圃場の行は元と完全に同じ: {same}(行数 {int((~touched).sum())})')

    # 主ラベルで開花起点 0〜+59 日がそろう件数(旧 / fixed)
    def n_full(ts):
        ts = ts[ts.field_id != 515].copy()
        ts['date'] = pd.to_datetime(ts.date)
        L = lab[lab[LABEL].notna()]
        ok = []
        for x in L.itertuples():
            s = ts[(ts.field_id == x.field_id) & (ts.year == x.year)]
            need = pd.date_range(x.flower_date, periods=60)
            hv = pd.to_datetime(x.harvest_date)
            ok.append(s.date.isin(need).sum() == 60 and need[-1] <= hv)
        L = L.assign(ok=ok)
        k = L[L.ok]
        return len(k), int((k[LABEL] == 1).sum()), int((k[LABEL] == 0).sum()), set(zip(k.field_id, k.year))
    no, ho, lo, so = n_full(old)
    nf, hf, lf, sf = n_full(new)
    log(f'主ラベル {LABEL} で開花起点 0〜+59 日の60日がそろう件数: 旧 {no}(多収{ho}/低収{lo})→ fixed {nf}(多収{hf}/低収{lf})。'
        f'外れた件 {sorted(so - sf)}、入った件 {sorted(sf - so)}')

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'changes.csv'), index=False, encoding='utf-8-sig')
    logf.close()


if __name__ == '__main__':
    main()
