"""
Weather テーブルの全行(圃場 x 日付)について、日長を天文計算で再計算する。

計算式は Forsythe et al. (1995) Ecological Modelling 80:87-95 の CBM モデル
(src/phenology.py の daylength())。
  day_length_sun   : p = 0.8333(太陽の上端が見かけ上地平線に接する。主に使う値)
  day_length_civil : p = 6.0   (常用薄明を含む。比較用)
緯度は field_master.lat(欠損時は Questionaire.lat)。
既存列 Weather.day_length(出典不明)は day_length_orig として残し、比較する。

検算: astral が import できれば、数圃場について astral の日の出〜日の入り
(および常用薄明の dawn〜dusk)の長さと比べる。astral は venv_rocket に入れていないので、
使うときは PYTHONPATH に astral を入れたフォルダを指定して実行する。

入力(読み取り専用): rocket/data/raw/FieldData_fieldid.db
出力:
  rocket/data/processed/daylength.csv            field_id, date, lat, day_length_sun, day_length_civil, day_length_orig
  rocket/outputs/daylength/summary.md            緯度の確認・既存列との比較・astral 検算の要約
  rocket/outputs/daylength/daylength_compare.png 既存値と計算値の散布図、差のヒストグラム、差の季節変化
"""
import datetime as dt
import os
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# このファイルは rocket/scripts/ にある。パスはファイル位置基準で解決する。
ROCKET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROCKET_DIR)
from src.phenology import daylength  # noqa: E402

DB = os.path.join(ROCKET_DIR, 'data', 'raw', 'FieldData_fieldid.db')
OUT_CSV = os.path.join(ROCKET_DIR, 'data', 'processed', 'daylength.csv')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'daylength')
OUT_MD = os.path.join(OUT_DIR, 'summary.md')
OUT_PNG = os.path.join(OUT_DIR, 'daylength_compare.png')

P_SUN, P_CIVIL = 0.8333, 6.0
LAT_TOL = 0.001      # 緯度の不一致とみなす差(度)
DIFF_TOL = 0.1       # 既存列との差の報告しきい値(時間)
CHECK_YEAR = 2016    # astral 検算に使う年(この年の全日)
JST = dt.timezone(dt.timedelta(hours=9))


def load(conn):
    fm = pd.read_sql('SELECT field_id, place, lat, lon FROM field_master', conn)
    q = pd.read_sql('SELECT field_id, year, place, lat, lon FROM Questionaire', conn)
    w = pd.read_sql('SELECT field_id, place, year, date, day_length FROM Weather', conn)
    return fm, q, w


def check_lat(fm, q):
    """手順1: field_master.lat と Questionaire.lat を field_id で突き合わせる。"""
    m = q.dropna(subset=['field_id']).merge(fm, on='field_id', suffixes=('_q', '_fm'))
    m['abs_diff'] = (m['lat_q'] - m['lat_fm']).abs()
    info = {
        'n_q_rows': len(q),
        'n_q_no_fid': int(q['field_id'].isna().sum()),
        'n_matched_rows': len(m),
        'n_matched_fields': m['field_id'].nunique(),
        'n_fm_lat_na': int(fm['lat'].isna().sum()),
        'max_abs_diff': m['abs_diff'].max(),
    }
    bad = m[m['abs_diff'] > LAT_TOL].sort_values('abs_diff', ascending=False)
    return info, bad


def field_lat(fm, q):
    """計算に使う緯度。field_master.lat、欠損時は Questionaire.lat(圃場内の平均)。"""
    lat = fm.set_index('field_id')['lat']
    q_lat = q.dropna(subset=['field_id', 'lat']).groupby('field_id')['lat'].mean()
    return lat.fillna(q_lat)


def compare(df):
    """手順3: 既存列 − day_length_sun の要約(全体と圃場ごと)。"""
    d = df['day_length_orig'] - df['day_length_sun']
    overall = {
        'n': int(d.notna().sum()),
        'mean': d.mean(), 'std': d.std(), 'min': d.min(), 'max': d.max(),
        'max_abs': d.abs().max(),
        'corr': df[['day_length_orig', 'day_length_sun']].corr().iloc[0, 1],
        'n_over': int((d.abs() > DIFF_TOL).sum()),
    }
    per_field = (df.assign(diff=d).groupby('field_id')
                 .apply(lambda g: pd.Series({
                     'lat': g['lat'].iloc[0],
                     'n': len(g),
                     'mean_diff': g['diff'].mean(),
                     'max_abs_diff': g['diff'].abs().max(),
                     'corr': g['day_length_orig'].corr(g['day_length_sun']),
                 }), include_groups=False))
    over = df.assign(diff=d)[d.abs() > DIFF_TOL]
    return overall, per_field, over


def p_fit(df, doy):
    """既存列に最も近い p の目安(候補ごとの平均絶対差)。"""
    rows = []
    for p in [-0.8333, 0.0, 0.2667, 0.5, 0.8333, 1.0, 6.0]:
        calc = daylength(df['lat'].values, doy, p)
        rows.append((p, np.nanmean(np.abs(df['day_length_orig'].values - calc))))
    return rows


def astral_check(lat_by_field, fm, field_ids):
    """手順4: astral による日の出〜日の入り(dawn〜dusk)の長さとの比較。"""
    try:
        from astral import Observer
        from astral.sun import sun
    except ImportError:
        return None
    days = pd.date_range(f'{CHECK_YEAR}-01-01', f'{CHECK_YEAR}-12-31')
    doy = days.dayofyear.values
    lon = fm.set_index('field_id')['lon']
    rows = []
    for fid in field_ids:
        lat = lat_by_field[fid]
        obs = Observer(latitude=lat, longitude=lon[fid])
        a_sun, a_civ = [], []
        for d in days:
            s = sun(obs, date=d.date(), tzinfo=JST)  # dawn/dusk の既定は常用薄明(6度)
            a_sun.append((s['sunset'] - s['sunrise']).total_seconds() / 3600)
            a_civ.append((s['dusk'] - s['dawn']).total_seconds() / 3600)
        e_sun = (daylength(lat, doy, P_SUN) - np.array(a_sun)) * 60
        e_civ = (daylength(lat, doy, P_CIVIL) - np.array(a_civ)) * 60
        rows.append({
            'field_id': fid, 'lat': lat,
            'sun_mean_min': e_sun.mean(), 'sun_max_abs_min': np.abs(e_sun).max(),
            'civil_mean_min': e_civ.mean(), 'civil_max_abs_min': np.abs(e_civ).max(),
        })
    import astral
    return astral.__version__, pd.DataFrame(rows)


def plot(df, doy):
    d = (df['day_length_orig'] - df['day_length_sun']) * 60
    color, ink, grid = '#2a78d6', '#52514e', '#e4e3df'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax in axes:
        ax.set_facecolor('#fcfcfb')
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_color(ink)
        ax.tick_params(colors=ink, labelsize=9)

    ax = axes[0]
    ax.scatter(df['day_length_sun'], df['day_length_orig'], s=2, color=color,
               alpha=0.15, lw=0, rasterized=True)
    lo, hi = df['day_length_sun'].min(), df['day_length_sun'].max()
    ax.plot([lo, hi], [lo, hi], color=ink, lw=1, ls='--', label='y = x')
    ax.set_xlabel('day_length_sun(CBM, p=0.8333)[h]')
    ax.set_ylabel('day_length_orig(既存列)[h]')
    ax.set_title('既存値と計算値', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.hist(d, bins=80, color=color, edgecolor='#fcfcfb', lw=0.5)
    ax.axvline(0, color=ink, lw=1, ls='--')
    ax.set_xlabel('既存 − 計算 [分]')
    ax.set_ylabel('行数')
    ax.set_title(f'差の分布(平均 {d.mean():+.2f} 分, n={len(d):,})', loc='left', fontsize=11)

    ax = axes[2]
    ax.scatter(doy, d, s=2, color=color, alpha=0.15, lw=0, rasterized=True)
    ax.axhline(0, color=ink, lw=1, ls='--')
    ax.set_xlabel('通日')
    ax.set_ylabel('既存 − 計算 [分]')
    ax.set_title('差の季節変化', loc='left', fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def fmt_table(df, floatfmt='.4f'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in r]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    fm, q, w = load(conn)
    conn.close()

    # 手順1
    lat_info, lat_bad = check_lat(fm, q)
    lat_by_field = field_lat(fm, q)

    # 手順2
    df = w.rename(columns={'day_length': 'day_length_orig'})
    df['lat'] = df['field_id'].map(lat_by_field)
    doy = pd.to_datetime(df['date']).dt.dayofyear.values
    df['day_length_sun'] = daylength(df['lat'].values, doy, P_SUN)
    df['day_length_civil'] = daylength(df['lat'].values, doy, P_CIVIL)
    n_dup = int(df.duplicated(['field_id', 'date']).sum())
    out = df[['field_id', 'date', 'lat', 'day_length_sun', 'day_length_civil', 'day_length_orig']]
    out = out.sort_values(['field_id', 'date'])
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # 手順3
    overall, per_field, over = compare(df)
    pfit = p_fit(df, doy)
    plot(df, doy)

    # 手順4: 最北・最南・中央付近の圃場と、参考例の B1(field_id=17)
    order = lat_by_field.sort_values()
    picks = [order.index[0], order.index[len(order) // 2], order.index[-1], 17]
    picks = list(dict.fromkeys(int(f) for f in picks))
    chk = astral_check(lat_by_field, fm, picks)

    # 要約
    L = ['# 日長の再計算(CBM モデル)の要約', '',
         f'- 作成スクリプト: `rocket/scripts/compute_daylength.py`(実行日 {dt.date.today()})',
         f'- 入力: `rocket/data/raw/FieldData_fieldid.db`(読み取り専用で開いた)',
         '- 計算式: Forsythe et al. (1995) CBM モデル(`src/phenology.py` の `daylength()`)。'
         f'day_length_sun は p = {P_SUN}、day_length_civil は p = {P_CIVIL}', '',
         '## 1. 緯度の確認', '',
         f'- Questionaire {lat_info["n_q_rows"]} 行のうち field_id が空欄の {lat_info["n_q_no_fid"]} 行は突き合わせの対象外',
         f'- 突き合わせた {lat_info["n_matched_rows"]} 行({lat_info["n_matched_fields"]} 圃場)で、'
         f'field_master.lat との差の最大は {lat_info["max_abs_diff"]:.6f}°',
         f'- 差が {LAT_TOL}° を超える行: **{len(lat_bad)} 件**',
         f'- field_master.lat の欠損: {lat_info["n_fm_lat_na"]} 圃場(Questionaire.lat で補う必要なし)'
         if lat_info['n_fm_lat_na'] == 0 else
         f'- field_master.lat の欠損: {lat_info["n_fm_lat_na"]} 圃場(Questionaire.lat で補った)',
         f'- Weather の全 {len(df):,} 行({df["field_id"].nunique()} 圃場)で緯度の欠損 {int(df["lat"].isna().sum())} 行。'
         f'(field_id, date) の重複 {n_dup} 行', '']
    if len(lat_bad):
        L += [fmt_table(lat_bad[['field_id', 'year', 'place_q', 'lat_q', 'lat_fm', 'abs_diff']].head(50), '.6f'), '']
    L += ['## 2. 既存列 Weather.day_length との比較(差 = 既存 − day_length_sun)', '',
          '| 項目 | 値 |', '|---|---|',
          f'| 行数 | {overall["n"]:,} |',
          f'| 平均差 | {overall["mean"]:+.4f} h({overall["mean"]*60:+.2f} 分) |',
          f'| 標準偏差 | {overall["std"]:.4f} h |',
          f'| 最小 / 最大 | {overall["min"]:+.4f} / {overall["max"]:+.4f} h |',
          f'| 最大絶対差 | {overall["max_abs"]:.4f} h({overall["max_abs"]*60:.2f} 分) |',
          f'| 相関 | {overall["corr"]:.6f} |',
          f'| 差が {DIFF_TOL} h を超える行 | {overall["n_over"]} 件 |', '']
    if len(over):
        L += ['差が大きい行の例(上位20件):', '',
              fmt_table(over.reindex(over['diff'].abs().sort_values(ascending=False).index)
                        [['field_id', 'date', 'lat', 'day_length_orig', 'day_length_sun', 'diff']].head(20)), '']
    b1 = df[(df['field_id'] == 17) & (df['date'] == '2015-04-01')]
    if len(b1):
        r = b1.iloc[0]
        L += [f'- 参考例 B1(field_id=17, 緯度 {r["lat"]})の 2015-04-01: 既存 {r["day_length_orig"]:.4f} h、'
              f'計算 {r["day_length_sun"]:.4f} h、差 {r["day_length_orig"] - r["day_length_sun"]:+.4f} h', '']
    pf = per_field[['mean_diff', 'max_abs_diff', 'corr']].describe().T[['min', '50%', 'max']]
    L += ['圃場ごとの統計の分布(611圃場):', '',
          fmt_table(pf.reset_index().rename(columns={'index': '統計'})), '',
          '既存列に最も近い p の目安(候補ごとの平均絶対差 [h]):', '',
          '| p | 平均絶対差 |', '|---|---|'] + [f'| {p} | {e:.4f} |' for p, e in pfit] + ['',
          '圃場ごとの全件は末尾の表を参照。', '']

    L += [f'## 3. astral による検算({CHECK_YEAR}年の全日)', '']
    if chk is None:
        L += ['astral を import できなかったため未実施。', '']
    else:
        ver, cdf = chk
        L += [f'astral {ver} の sunrise〜sunset(太陽上端 + 大気差、仰角 −0.833°)と dawn〜dusk(常用薄明、6°)を、'
              'それぞれ day_length_sun と day_length_civil と比べた。誤差 = CBM − astral [分]。'
              '圃場は緯度が最南・中央・最北のものと B1(field_id=17)。', '',
              fmt_table(cdf, '.3f'), '']

    L += ['## 付録: 圃場ごとの比較(差 = 既存 − day_length_sun [h])', '',
          fmt_table(per_field.reset_index().astype({'field_id': int, 'n': int}), '.5f'), '']
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    print(f'保存: {OUT_CSV}\n保存: {OUT_MD}\n保存: {OUT_PNG}')
    print(f'緯度の不一致 {len(lat_bad)} 件 / 差の平均 {overall["mean"]:+.4f} h, 最大絶対差 {overall["max_abs"]:.4f} h, '
          f'相関 {overall["corr"]:.6f}, >{DIFF_TOL}h {overall["n_over"]} 件')
    if chk is not None:
        print(chk[1].to_string())


if __name__ == '__main__':
    main()
