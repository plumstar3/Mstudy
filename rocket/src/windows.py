"""
段階5以降で使う入力系列の前処理(段階4で作成)。

  load_cases()        label_muniyear 付きの圃場×年と、開花日・灌水・観測範囲を読む
  load_timeseries()   VWC 時系列(収穫日より後の観測は除く)
  fit_flowering()     開花日の推定式(実測開花日のある全件で学習した最終版)
  add_flowering()     開花日を付ける(実測を優先し、無ければ推定。flower_estimated フラグ付き)
  extract_windows()   起点(開花日 / 播種日)からの窓で系列を切り出す。覆わない件は理由付きで除外
  to_raw / znorm / first_diff   入力の形の変換(どれを使うかは段階5で決める)

配列の形は (件数, 1, 窓の長さ)。aeon / sktime の ROCKET にそのまま入る。
元データ(FieldData_fieldid.db)は読み取り専用で開く。
"""
import os
import sqlite3

import numpy as np
import pandas as pd

from src.phenology import SeedDoyRegression

ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(ROCKET_DIR)
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DB = os.path.join(ROOT_DIR, 'data', 'processed', 'FieldData_fieldid.db')

# irrigation 列の生育時期の区分(文字列、複数はカンマ区切り)。日付や回数は無い
IRRIGATION_PERIODS = {'出芽期': 'irr_emergence', '開花期以前': 'irr_before_flowering',
                      '開花期~直後': 'irr_flowering', '莢肥大期': 'irr_pod_filling'}


def parse_irrigation(s):
    """irrigation 列の値を、灌水の有無と時期ごとのフラグにする。

    '無し' → irrigated=0、時期の区分 → irrigated=1、空文字・NULL → 不明(NaN)。
    """
    out = {c: np.nan for c in IRRIGATION_PERIODS.values()}
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == '':
        return {'irrigated': np.nan, **out}
    s = str(s).strip()
    if s == '無し':
        return {'irrigated': 0, **{c: 0 for c in out}}
    parts = [p.strip() for p in s.split(',')]
    unknown = [p for p in parts if p not in IRRIGATION_PERIODS]
    if unknown:
        raise ValueError(f'irrigation に想定外の値: {unknown}')
    return {'irrigated': 1, **{c: int(k in parts) for k, c in IRRIGATION_PERIODS.items()}}


def load_cases(label='label_muniyear'):
    """label が付いた圃場×年(1行=1件)。開花日の実測値、灌水、観測範囲を付ける。"""
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig',
                      dtype={'muniCd': str})
    d = lab[lab[label].notna()].copy()
    d['y'] = d[label].astype(int)
    d['muniyear'] = d['muniCd'] + '_' + d['year'].astype(str)

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    q = pd.read_sql('SELECT field_id, year, flower_date, irrigation FROM Questionaire '
                    'WHERE field_id IS NOT NULL', conn)
    conn.close()
    q['field_id'] = pd.to_numeric(q['field_id'], errors='coerce')
    q = q.dropna(subset=['field_id']).astype({'field_id': int, 'year': int})
    q = q.drop_duplicates(['field_id', 'year'])  # build_soilmoisture_dataset.py と同じく先頭を採る
    q = q.rename(columns={'irrigation': 'irrigation_raw'})
    d = d.drop(columns=['irrigation'], errors='ignore').merge(q, on=['field_id', 'year'], how='left')
    irr = pd.DataFrame([parse_irrigation(v) for v in d['irrigation_raw']], index=d.index)
    d = pd.concat([d, irr], axis=1)

    d['seed_date'] = pd.to_datetime(d['seed_date'])
    d['harvest_date'] = pd.to_datetime(d['harvest_date'])
    d['seed_doy'] = d['seed_date'].dt.dayofyear
    d['harvest_dap'] = (d['harvest_date'] - d['seed_date']).dt.days
    d['flower_date_obs'] = pd.to_datetime(d['flower_date'], errors='coerce')
    d['flower_dap_obs'] = (d['flower_date_obs'] - d['seed_date']).dt.days
    return d.drop(columns=['flower_date']).reset_index(drop=True)


def load_timeseries(cases):
    """cases の圃場×年の VWC 時系列。収穫日より後の観測は除く。"""
    ts = pd.read_csv(os.path.join(PROC, 'soilmoisture_timeseries.csv'), encoding='utf-8-sig',
                     parse_dates=['date'])
    ts = ts.merge(cases[['field_id', 'year', 'harvest_dap']], on=['field_id', 'year'])
    n0 = len(ts)
    ts = ts[ts['dap'] <= ts['harvest_dap']].drop(columns=['harvest_dap'])
    ts.attrs['n_dropped_after_harvest'] = n0 - len(ts)
    return ts.sort_values(['field_id', 'year', 'dap']).reset_index(drop=True)


def fit_flowering(cases):
    """開花日の推定式 (ii) を、実測開花日のある全件で学習する(ラベルは使わない)。"""
    obs = cases[cases['flower_dap_obs'].notna()].assign(flower_dap=lambda x: x['flower_dap_obs'])
    model = SeedDoyRegression().fit(obs)
    model.n_train = len(obs)
    return model


def add_flowering(cases, model):
    """開花日を付ける。flower_estimated=True は推定値。"""
    d = cases.copy()
    d['flower_estimated'] = d['flower_dap_obs'].isna()
    d['flower_dap'] = d['flower_dap_obs']
    d.loc[d['flower_estimated'], 'flower_dap'] = model.predict(d[d['flower_estimated']])
    d['flower_dap'] = d['flower_dap'].astype(int)
    d['flower_date'] = d['seed_date'] + pd.to_timedelta(d['flower_dap'], unit='D')
    return d


def extract_windows(cases, ts, origin, start, end):
    """起点からの日数 start..end(両端を含む)の VWC を切り出す。

    origin='flower': 開花日 = 0、origin='sowing': 播種日 = 0(= dap)。
    窓の全日に観測がある件だけを残す。戻り値は (X, meta, excluded)。
      X: (件数, 1, end - start + 1)、meta: X と同じ順の表、excluded: 除外した件と理由
    """
    if origin not in ('flower', 'sowing'):
        raise ValueError(origin)
    L = end - start + 1
    zero = cases['flower_dap'] if origin == 'flower' else pd.Series(0, index=cases.index)
    series = {k: s for k, s in ts.groupby(['field_id', 'year'])}
    X, keep, excl = [], [], []
    for i, r in cases.iterrows():
        lo, hi = int(zero[i]) + start, int(zero[i]) + end  # dap での窓
        s = series.get((r.field_id, r.year))
        if s is None:
            excl.append((i, '系列なし'))
            continue
        w = s[(s['dap'] >= lo) & (s['dap'] <= hi)]
        reasons = []
        if s['dap'].min() > lo:
            reasons.append(f'観測開始が窓の始まりより {int(s.dap.min() - lo)} 日遅い')
        if s['dap'].max() < hi:
            reasons.append(f'観測終了(収穫日で打ち切り)が窓の終わりより {int(hi - s.dap.max())} 日早い')
        if not reasons and len(w) != L:
            reasons.append(f'窓の中に欠けが {L - len(w)} 日ある')
        if reasons:
            excl.append((i, ' / '.join(reasons)))
            continue
        X.append(w['VWC'].to_numpy(dtype=float))
        meta = r.to_dict()
        meta.update({'window_origin': origin, 'window_start': start, 'window_end': end,
                     'window_start_dap': lo, 'window_end_dap': hi,
                     'window_start_date': w['date'].iloc[0], 'window_end_date': w['date'].iloc[-1]})
        keep.append(meta)
    X = np.stack(X)[:, None, :] if X else np.empty((0, 1, L))
    excluded = cases.loc[[i for i, _ in excl]].assign(reason=[m for _, m in excl])
    return X, pd.DataFrame(keep).reset_index(drop=True), excluded.reset_index(drop=True)


def to_raw(X):
    return np.asarray(X, dtype=float).copy()


def znorm(X, eps=1e-8):
    """系列ごとに平均0・標準偏差1にする(標準偏差が0に近い系列は平均を引くだけ)。"""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / np.where(sd < eps, 1.0, sd)


def first_diff(X):
    """1次差分。長さは1つ短くなる(60点 → 59点)。"""
    return np.diff(np.asarray(X, dtype=float), axis=-1)
