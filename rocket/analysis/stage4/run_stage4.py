"""
段階4: 段階5で使う入力データセットを作る(ROCKET はまだ使わない)。

  main             開花起点 0〜+59 日(60点)
  dap              dap 50–109(60点)
  main_common      main と dap の両方を覆う件を、main の窓で切り出したもの
  dap_common       同じ件を、dap の窓で切り出したもの
  main_obs         main のうち、実測開花日の件
  main_no_kakogawa main から加古川(28210)を除いたもの

保存先: rocket/data/processed/stage4/<名前>_X.npy((件数, 1, 60)、生の VWC)と <名前>_meta.csv
図・確認表: rocket/outputs/stage4/

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage4\\run_stage4.py
"""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
from src.windows import (IRRIGATION_PERIODS, add_flowering, extract_windows, first_diff,  # noqa: E402
                         fit_flowering, load_cases, load_timeseries, znorm)

DATA_DIR = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage4')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage4')
KAKOGAWA = '28210'
EXPECTED = {'main': 82, 'dap': 81}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
C_HIGH, C_LOW = '#e34948', '#2a78d6'  # 既存の図と同じく 赤=多収, 青=低収
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

META_COLS = ['field_id', 'year', 'muniCd', 'muniyear', 'y', 'seed_date', 'harvest_date',
             'flower_date', 'flower_dap', 'flower_estimated',
             'window_origin', 'window_start', 'window_end', 'window_start_dap', 'window_end_dap',
             'window_start_date', 'window_end_date', 'irrigation_raw', 'irrigated',
             *IRRIGATION_PERIODS.values(), 'irrigation_day_in_window']


def subset(X, meta, mask):
    mask = np.asarray(mask)
    return X[mask], meta[mask].reset_index(drop=True)


def build(cases, ts):
    X_m, m_m, ex_m = extract_windows(cases, ts, 'flower', 0, 59)
    X_d, m_d, ex_d = extract_windows(cases, ts, 'sowing', 50, 109)
    common = set(zip(m_m.field_id, m_m.year)) & set(zip(m_d.field_id, m_d.year))
    in_c = lambda m: [k in common for k in zip(m.field_id, m.year)]  # noqa: E731
    ds = {
        'main': (X_m, m_m),
        'dap': (X_d, m_d),
        'main_common': subset(X_m, m_m, in_c(m_m)),
        'dap_common': subset(X_d, m_d, in_c(m_d)),
        'main_obs': subset(X_m, m_m, ~m_m.flower_estimated.values),
        'main_no_kakogawa': subset(X_m, m_m, (m_m.muniCd != KAKOGAWA).values),
    }
    # 灌水日の位置: irrigation 列は生育時期の区分だけで日付が無いので求められない(NaN)
    for name, (X, m) in ds.items():
        m['irrigation_day_in_window'] = np.nan
        ds[name] = (X, m[META_COLS])
    excluded = pd.concat([ex_m.assign(dataset='main'), ex_d.assign(dataset='dap')])
    return ds, excluded


def verify(name, X, meta, ts):
    """行の順番の一致・長さ・NaN を確かめる。"""
    assert X.ndim == 3 and X.shape[1] == 1 and X.shape[2] == 60, X.shape
    assert len(X) == len(meta)
    assert not np.isnan(X).any(), f'{name} に NaN'
    s = ts.set_index(['field_id', 'year'])
    for i, r in meta.iterrows():  # 配列の行 i が meta の行 i と同じ圃場×年か、元の系列から取り直して比べる
        w = s.loc[(r.field_id, r.year)]
        w = w[(w.dap >= r.window_start_dap) & (w.dap <= r.window_end_dap)]['VWC'].to_numpy()
        assert np.array_equal(w, X[i, 0]), f'{name} の {i} 行目が一致しない'


def describe(name, meta, n_groups_all):
    g = meta.groupby('muniyear')['y'].agg(['sum', 'size'])
    bal = int((g['sum'] * 2 == g['size']).sum())
    irr = meta[meta.irrigated == 1]
    return {'dataset': name, 'n': len(meta), 'n_high': int(meta.y.sum()), 'n_low': int((1 - meta.y).sum()),
            'n_regions': meta.muniCd.nunique(),
            'n_2016': int((meta.year == 2016).sum()), 'n_2017': int((meta.year == 2017).sum()),
            'n_2018': int((meta.year == 2018).sum()),
            'groups_balanced': bal, 'groups_unbalanced': len(g) - bal, 'groups_dropped': n_groups_all - len(g),
            'n_flower_estimated': int(meta.flower_estimated.sum()), 'n_kakogawa': int((meta.muniCd == KAKOGAWA).sum()),
            'n_irrigated': len(irr), 'n_irrigated_high': int(irr.y.sum()), 'n_irrigated_low': int((1 - irr.y).sum()),
            'n_irrigation_unknown': int(meta.irrigated.isna().sum())}


# ---------------------------------------------------------------- 図
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_mean_iqr(ds):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (name, xlabel, x0) in zip(axs, [('main', '開花からの日数', 0), ('dap', '播種後日数(dap)', 50)]):
        X, m = ds[name]
        t = np.arange(X.shape[2]) + x0
        for yv, col, lab in [(1, C_HIGH, '多収'), (0, C_LOW, '低収')]:
            v = X[m.y.values == yv, 0]
            q1, q3 = np.percentile(v, [25, 75], axis=0)
            ax.fill_between(t, q1, q3, color=col, alpha=0.15, lw=0)
            ax.plot(t, v.mean(axis=0), color=col, lw=2, label=f'{lab}(n={len(v)})平均')
        ax.set_title(f'{name}({len(m)}件)', fontsize=11, color=INK, loc='left')
        ax.set_xlabel(xlabel, color=INK)
        style(ax)
    axs[0].set_ylabel('VWC(帯 = 四分位範囲)', color=INK)
    axs[0].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('多収・低収の VWC 系列の平均と四分位範囲(生の値)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mean_iqr_main_dap.png'), dpi=150)
    plt.close(fig)


def plot_irrigation_examples(ds):
    """灌水ありの件を、灌水の時期の区分ごとに並べる(灌水日は不明なので印は付けられない)。"""
    X, m = ds['main']
    irr = m[m.irrigated == 1]
    groups = list(irr.groupby('irrigation_raw'))
    fig, axs = plt.subplots(len(groups), 1, figsize=(11, 2.8 * len(groups)), sharex=True, squeeze=False)
    t = np.arange(X.shape[2])
    for ax, (val, g) in zip(axs[:, 0], groups):
        for i, r in g.iterrows():
            col = C_HIGH if r.y == 1 else C_LOW
            ax.plot(t, X[i, 0], color=col, lw=1.3, alpha=0.9)
            ax.text(t[-1] + 0.5, X[i, 0, -1], f'{r.field_id}/{r.year}', fontsize=7, color=MUTED, va='center')
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f'irrigation = 「{val}」({len(g)}件: 多収{int(g.y.sum())}・低収{int((1 - g.y).sum())})',
                     fontsize=10, color=INK, loc='left')
        ax.set_ylabel('VWC', color=INK)
        style(ax)
    axs[-1, 0].set_xlabel('開花からの日数(0 = 開花日)。灌水の日付は記録が無いため印は付けていない', color=INK)
    axs[-1, 0].set_xlim(-1, 66)
    fig.suptitle('main の灌水ありの件(赤=多収, 青=低収)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'irrigation_examples_main.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = load_cases()
    print(f'対象: label_muniyear 付き {len(cases)} 件')
    print('irrigation の値:', cases.irrigation_raw.map(repr).value_counts(dropna=False).to_dict())
    model = fit_flowering(cases)
    print(f'開花日の推定式: 開花 dap = {model.a:.4f} + ({model.b:.6f}) × 播種日の通日(学習 {model.n_train} 件)')
    cases = add_flowering(cases, model)
    ts = load_timeseries(cases)
    print('収穫日より後で除いた観測:', ts.attrs['n_dropped_after_harvest'], '行')

    ds, excluded = build(cases, ts)
    save_ex = excluded[['dataset', 'field_id', 'year', 'muniCd', 'y', 'flower_dap', 'flower_estimated', 'reason']]
    save_ex.to_csv(os.path.join(OUT_DIR, 'excluded.csv'), index=False, encoding='utf-8-sig')
    print('\n除外:')
    print(save_ex.to_string(index=False))

    n_groups_all = cases.muniyear.nunique()
    rows = []
    for name, (X, m) in ds.items():
        verify(name, X, m, ts)
        np.save(os.path.join(DATA_DIR, f'{name}_X.npy'), X)
        m.to_csv(os.path.join(DATA_DIR, f'{name}_meta.csv'), index=False, encoding='utf-8-sig')
        rows.append(describe(name, m, n_groups_all))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    print('\n確認(行の順番・長さ60・NaN なし はすべて assert 済み):')
    print(summary.to_string(index=False))
    for k, v in EXPECTED.items():
        assert summary.set_index('dataset').loc[k, 'n'] == v, f'{k} の件数が段階3の {v} 件と違う'
    print('段階3・3b の件数と一致: main 82 / dap 81')

    # 変換の確認(段階5で選ぶ)
    Xm = ds['main'][0]
    print('\n変換: raw', Xm.shape, '/ znorm', znorm(Xm).shape, '/ first_diff', first_diff(Xm).shape)

    manifest = {
        'created': pd.Timestamp.now().isoformat(timespec='seconds'),
        'label': 'label_muniyear',
        'flowering_model': {'method': '(ii) 開花 dap = a + b × 播種日の通日', 'a': model.a, 'b': model.b,
                            'n_train': model.n_train, 'rounding': '整数の日に丸める'},
        'windows': {'main': '開花日 0〜+59 日', 'dap': 'dap 50–109'},
        'harvest_truncation_rows': int(ts.attrs['n_dropped_after_harvest']),
        'array_shape': {k: list(v[0].shape) for k, v in ds.items()},
        'values': '生の VWC。znorm / first_diff は src/windows.py の関数で段階5で変換する',
    }
    with open(os.path.join(DATA_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    plot_mean_iqr(ds)
    plot_irrigation_examples(ds)
    print(f'\n出力: {DATA_DIR}\n      {OUT_DIR}')


if __name__ == '__main__':
    main()
