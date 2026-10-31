"""
段階3b: 開花起点の窓を1日刻みで調べる。「開花前 a 日〜開花後 b 日」の窓で何件残るか。

  1. 圃場×年ごとの開花前後のデータ日数(pre = 開花 − 観測開始、post = 観測終了 − 開花)
  2. (a, b) = (0〜40, 40〜90) の残存件数表(94件 / 実測開花日の66件)
  3. 図(ヒートマップ+等高線、窓の長さごとの最大件数)
  4. 0〜+59 から1日ずつ広げたときに落ちる圃場
  5. 件数の下限ごとの最長の窓

開花日は実測を優先し、無い件は段階3の (ii) 播種日の通日による回帰で推定する。
観測終了は収穫日で打ち切る(収穫後の VWC は作物の状態を反映しないため)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage3b\\run_stage3b.py
出力: rocket/outputs/stage3b/
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage3'))
from run_stage3 import load  # noqa: E402
from src.phenology import SeedDoyRegression  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage3b')
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
C_OBS, C_EST = '#2a78d6', '#eb6834'  # 実測 / 推定
C_HIGH, C_LOW = '#e34948', '#2a78d6'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

A_RANGE = np.arange(0, 41)
B_RANGE = np.arange(40, 91)
BASE = (0, 59)  # 段階3の主設定
LEVELS = [70, 75, 80, 85, 90]
KAKOGAWA = '28210'


def save(df, name):
    df.to_csv(os.path.join(OUT_DIR, name), index=False, encoding='utf-8-sig')


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------- 1. pre / post
def build_cases():
    d, ts, cum = load()
    obs = d[d['has_flower']].assign(flower_dap=lambda x: x['flower_dap_obs'])
    reg = SeedDoyRegression().fit(obs)
    d['flower_estimated'] = ~d['has_flower']
    d['flower_dap'] = d['flower_dap_obs']
    d.loc[d.flower_estimated, 'flower_dap'] = reg.predict(d[d.flower_estimated])

    # 観測終了を収穫日で打ち切る
    d['end_dap_raw'] = d['end_dap']
    d['end_dap'] = np.minimum(d['end_dap_raw'], d['harvest_dap'])
    d['cut_days'] = d['end_dap_raw'] - d['end_dap']

    d['pre'] = (d['flower_dap'] - d['start_dap']).astype(int)
    d['post'] = (d['end_dap'] - d['flower_dap']).astype(int)
    d['kakogawa'] = d['muniCd'] == KAKOGAWA
    cols = ['field_id', 'year', 'muniCd', 'muniyear', 'y', 'breed', 'seed_date', 'harvest_dap',
            'start_dap', 'end_dap_raw', 'end_dap', 'cut_days', 'flower_dap', 'flower_estimated',
            'pre', 'post', 'kakogawa']
    save(d[cols], 'cases_pre_post.csv')
    return d, reg


def plot_pre_post(d):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (col, label, bins) in zip(axs, [('pre', '開花前の日数 pre = 開花日 − 観測開始日', np.arange(-15, 51, 5)),
                                            ('post', '開花後の日数 post = 観測終了日 − 開花日', np.arange(-10, 106, 5))]):
        ax.hist([d.loc[~d.flower_estimated, col], d.loc[d.flower_estimated, col]], bins=bins,
                stacked=True, color=[C_OBS, C_EST], edgecolor='white', linewidth=1,
                label=['開花日が実測', '開花日が推定'])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(col, fontsize=10, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        style(ax)
        ax.set_xlabel(label, color=INK)
    axs[0].set_ylabel('圃場×年の数', color=INK)
    axs[1].legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('開花前後のデータ日数(label_muniyear 付き94件、観測終了は収穫日で打ち切り)',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pre_post_hist.png'), dpi=150)
    plt.close(fig)


def plot_spans(d):
    """圃場×年ごとの測定期間を、開花日 = 0 の軸で横棒にする(上から観測開始が早い順)。"""
    s = d.sort_values(['pre', 'post'], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 15))
    ax.axvspan(BASE[0], BASE[1], color='#9a9890', alpha=0.15, lw=0)
    ax.axvline(63, color=MUTED, ls='--', lw=1)
    ax.axvline(0, color=INK, lw=1.2)
    for i, r in s.iterrows():
        col = C_HIGH if r.y == 1 else C_LOW
        ax.barh(i, r.post + r.pre, left=-r.pre, height=0.7, color=col,
                alpha=0.45 if r.flower_estimated else 1.0, lw=0)
    ax.set_yticks(range(len(s)), [f'{f}/{y}' for f, y in zip(s.field_id, s.year)], fontsize=6)
    ax.set_ylim(len(s) - 0.5, -0.5)
    ax.set_xlim(-50, 105)
    ax.set_xticks(range(-50, 101, 10))
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel('開花からの日数(0 = 開花日)', color=INK)
    ax.set_ylabel('圃場×年(field_id/年)。上から観測開始が早い順', color=INK)
    style(ax)
    ax.tick_params(axis='y', length=0)
    ax.text(BASE[1] / 2, -1.2, '段階3の主設定 0〜+59', ha='center', va='bottom', fontsize=9, color=MUTED)
    ax.text(63, -1.2, '+63', ha='center', va='bottom', fontsize=9, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_HIGH), plt.Rectangle((0, 0), 1, 1, color=C_LOW),
               plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45)]
    fig.legend(handles, ['多収', '低収', '薄い色 = 開花日が推定'],
               loc='upper right', ncol=3, frameon=False, fontsize=10)
    fig.suptitle('各圃場の VWC 測定期間(開花日を0日目とした軸、観測終了は収穫日で打ち切り)',
                 fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(os.path.join(OUT_DIR, 'spans_by_field.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. 残存件数表
def grid_counts(d):
    n_groups = d['muniyear'].nunique()
    pre, post = d['pre'].values, d['post'].values
    rows = []
    for a in A_RANGE:
        for b in B_RANGE:
            m = (pre >= a) & (post >= b)
            s = d[m]
            g = s.groupby('muniyear')['y'].agg(['sum', 'size'])
            bal = int((g['sum'] * 2 == g['size']).sum())
            rows.append({'a': a, 'b': b, 'length': a + b + 1, 'n': int(m.sum()),
                         'n_high': int(s.y.sum()), 'n_low': int((1 - s.y).sum()),
                         'n_regions': s.muniCd.nunique(),
                         'n_2016': int((s.year == 2016).sum()), 'n_2017': int((s.year == 2017).sum()),
                         'n_2018': int((s.year == 2018).sum()),
                         'groups_balanced': bal, 'groups_unbalanced': len(g) - bal,
                         'groups_dropped': n_groups - len(g),
                         'n_flower_observed': int((~s.flower_estimated).sum()),
                         'n_flower_estimated': int(s.flower_estimated.sum()),
                         'n_kakogawa': int(s.kakogawa.sum())})
    return pd.DataFrame(rows)


def best_by_length(g):
    """長さ L ごとに件数最大の (a, b)。同数なら釣り合いの集団が多い方、さらに同じなら a が小さい方。"""
    g = g.sort_values(['length', 'n', 'groups_balanced', 'a'], ascending=[True, False, False, True])
    return g.groupby('length').head(1).reset_index(drop=True)


def candidates(g, levels=(85, 80, 75, 70)):
    rows = []
    for lv in levels:
        c = g[g.n >= lv].sort_values(['length', 'groups_balanced', 'n', 'a'],
                                      ascending=[False, False, False, True])
        if c.empty:
            rows.append({'min_n': lv, 'note': f'該当なし(この範囲の最大は {int(g.n.max())} 件)'})
            continue
        top = c.iloc[0]
        same = c[c.length == top.length]
        rows.append({'min_n': lv, **top.to_dict(), 'includes_pre_flowering': bool(top.a > 0),
                     'n_alternatives_same_length': len(same) - 1,
                     'alternatives': '; '.join(f'({int(r.a)},{int(r.b)}) n={int(r.n)} 釣り合い{int(r.groups_balanced)}'
                                               for _, r in same.iloc[1:].iterrows())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 図
def plot_heatmap(g, title, fname):
    M = g.pivot(index='a', columns='b', values='n').loc[A_RANGE, B_RANGE].values
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(M, origin='lower', cmap='Blues', vmin=0, vmax=94, aspect='auto',
                   extent=(B_RANGE[0] - .5, B_RANGE[-1] + .5, A_RANGE[0] - .5, A_RANGE[-1] + .5))
    cs = ax.contour(B_RANGE, A_RANGE, M, levels=LEVELS, colors=INK, linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f'{int(v)}件', fontsize=9, inline=True)
    # 窓の長さ L = a + b + 1 が一定の斜線
    for L in range(50, 131, 10):
        bb = np.array([B_RANGE[0] - .5, B_RANGE[-1] + .5])
        aa = L - 1 - bb
        ax.plot(bb, aa, color='#9a9890', lw=0.8, ls='--')
        bt = L - 1 - 39.5  # 上端(a=39.5)での b
        if B_RANGE[0] <= bt <= B_RANGE[-1]:
            ax.text(bt, 40.2, f'L={L}', fontsize=8, color=MUTED, ha='center', va='bottom')
        else:
            ax.text(B_RANGE[0] - .3, L - 1 - B_RANGE[0], f'L={L}', fontsize=8, color=MUTED, ha='left', va='bottom')
    ax.plot(BASE[1], BASE[0], marker='o', ms=8, mfc='none', mec='#e34948', mew=2)
    ax.annotate('段階3の主設定 (0, +59)', (BASE[1], BASE[0]), xytext=(-40, -2), textcoords='offset points',
                fontsize=9, color=INK, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(B_RANGE[0] - .5, B_RANGE[-1] + .5)
    ax.set_ylim(A_RANGE[0] - .5, A_RANGE[-1] + .5)
    ax.set_xlabel('開花後 b 日まで', color=INK)
    ax.set_ylabel('開花前 a 日から', color=INK)
    style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('残る件数', color=INK)
    ax.set_title(title, fontsize=12, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def plot_by_length(best94, best66):
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2]})
    ax = axs[0]
    ax.plot(best94.length, best94.n, color=C_OBS, lw=2, label='94件(実測+推定)')
    ax.plot(best66.length, best66.n, color=C_EST, lw=2, label='実測開花日の66件だけ')
    for lv in [70, 75, 80, 85, 90]:
        ax.axhline(lv, color=GRID, lw=1, zorder=0)
        ax.text(132, lv, f'{lv}', fontsize=8, color=MUTED, va='center')
    for L in [60, 70, 80, 90, 100]:
        r = best94[best94.length == L].iloc[0]
        ax.annotate(f'L={L}: ({int(r.a)}, +{int(r.b)}) {int(r.n)}件', (L, r.n), xytext=(4, 6),
                    textcoords='offset points', fontsize=8, color=INK)
        ax.plot(L, r.n, 'o', color=C_OBS, ms=4)
    ax.set_ylabel('その長さで取れる最大件数', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    ax = axs[1]
    ax.plot(best94.length, best94.a, color=INK, lw=2, label='a(開花前の日数)')
    ax.plot(best94.length, best94.b, color=MUTED, lw=2, ls='--', label='b(開花後の日数)')
    ax.set_ylabel('最大件数のときの a, b(94件)', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left')

    ax = axs[2]
    ax.plot(best94.length, best94.groups_balanced, color=C_OBS, lw=2, label='釣り合い(多収=低収)')
    ax.plot(best94.length, best94.groups_unbalanced, color='#e34948', lw=2, label='崩れ')
    ax.plot(best94.length, best94.groups_dropped, color=MUTED, lw=2, ls=':', label='消失')
    ax.set_ylabel('市町村×年の集団の数(94件)', color=INK)
    ax.set_xlabel('窓の長さ L = a + b + 1(日)', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    for a in axs:
        style(a)
        a.grid(axis='y', color=GRID, lw=0.8)
        a.set_axisbelow(True)
    fig.suptitle('窓の長さごとの最大件数(開花起点)', fontsize=13, color=INK, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'max_n_by_length.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. 落ちる圃場
def drop_lists(d):
    a0, b0 = BASE
    base = d[(d.pre >= a0) & (d.post >= b0)]
    cols = ['field_id', 'year', 'muniCd', 'y', 'flower_estimated', 'pre', 'post']
    # マイナス側: a を増やすと pre < a で落ちる。落ちる日 = pre + 1(その a で初めて外れる)
    minus = base[base.pre + 1 <= A_RANGE[-1]][cols].assign(side='minus', drop_at=lambda x: -(x.pre + 1))
    plus = base[base.post + 1 <= B_RANGE[-1]][cols].assign(side='plus', drop_at=lambda x: x.post + 1)
    out = pd.concat([minus, plus]).sort_values(['side', 'drop_at'], key=lambda s: s.abs() if s.name == 'drop_at' else s)
    out['label'] = np.where(out.y == 1, '多収', '低収')
    out['flower'] = np.where(out.flower_estimated, '推定', '実測')
    save(out, 'drop_list.csv')
    return base, out


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d, reg = build_cases()
    print(f'推定式: 開花 dap = {reg.a:.2f} + {reg.b:.4f} × 播種日の通日')
    cut = d[d.cut_days > 0]
    print(f'収穫日で打ち切った件: {len(cut)}件、日数 {cut.cut_days.tolist()}')
    print(cut[['field_id', 'year', 'muniCd', 'end_dap_raw', 'harvest_dap', 'cut_days', 'post']].to_string(index=False))
    print('field_id=90 は対象に含まれるか:', bool((d.field_id == 90).any()))

    print('\n[1] pre / post')
    print(d[['pre', 'post']].describe().round(1).T.to_string())
    for lab, s in [('実測', d[~d.flower_estimated]), ('推定', d[d.flower_estimated]),
                   ('多収', d[d.y == 1]), ('低収', d[d.y == 0]), ('加古川', d[d.kakogawa])]:
        print(f'  {lab}(n={len(s)}): pre ' + ', '.join(f'{v:.0f}' for v in s.pre.quantile([0, .25, .5, .75, 1]))
              + ' / post ' + ', '.join(f'{v:.0f}' for v in s.post.quantile([0, .25, .5, .75, 1])))
    print('  pre < 0(開花後に観測開始):', int((d.pre < 0).sum()), '件')
    plot_pre_post(d)
    plot_spans(d)

    print('\n[2] 残存件数表')
    g94 = grid_counts(d)
    g66 = grid_counts(d[~d.flower_estimated])
    save(g94, 'grid_counts_94.csv')
    save(g66, 'grid_counts_obs66.csv')
    print('段階3の主設定 (0, +59):', g94[(g94.a == 0) & (g94.b == 59)].iloc[0].to_dict())

    best94, best66 = best_by_length(g94), best_by_length(g66)
    save(best94, 'best_by_length_94.csv')
    save(best66, 'best_by_length_obs66.csv')
    plot_heatmap(g94, '開花前 a 日〜開花後 b 日の窓に残る件数(94件、実測+推定開花日)', 'heatmap_ab_94.png')
    plot_heatmap(g66, '開花前 a 日〜開花後 b 日の窓に残る件数(実測開花日の66件)', 'heatmap_ab_obs66.png')
    plot_by_length(best94, best66)
    print(best94[best94.length.isin(range(50, 121, 5))][['length', 'a', 'b', 'n', 'n_high', 'n_low', 'n_regions',
                                                         'groups_balanced', 'groups_unbalanced', 'groups_dropped',
                                                         'n_flower_estimated', 'n_kakogawa']].to_string(index=False))

    print('\n[4] 落ちる圃場')
    base, drops = drop_lists(d)
    print(f'出発点 (0, +59) の件数: {len(base)}')
    for side in ['minus', 'plus']:
        s = drops[drops.side == side]
        print(f'--- {side}: 範囲内で落ちる {len(s)}件')
        print(s[['drop_at', 'field_id', 'year', 'muniCd', 'label', 'flower', 'pre', 'post']].to_string(index=False))

    print('\n[5] 候補')
    c94 = candidates(g94)
    c66 = candidates(g66, levels=(85, 80, 75, 70, 60, 55, 50))  # 66件では 60〜50 件も参考に出す
    save(c94, 'candidates_94.csv')
    save(c66, 'candidates_obs66.csv')
    show = ['min_n', 'a', 'b', 'length', 'n', 'n_high', 'n_low', 'n_regions', 'n_2016', 'n_2017', 'n_2018',
            'groups_balanced', 'groups_unbalanced', 'groups_dropped', 'n_flower_estimated', 'n_kakogawa',
            'includes_pre_flowering', 'alternatives']
    print(c94.reindex(columns=show + ['note']).to_string(index=False))
    print(c66.reindex(columns=[c for c in show if c != 'n_kakogawa'] + ['note']).to_string(index=False))
    print('(0, 40) の件数: 94件', int(g94[(g94.a == 0) & (g94.b == 40)].n.iloc[0]),
          '/ 66件', int(g66[(g66.a == 0) & (g66.b == 40)].n.iloc[0]))
    print(f'出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
