"""
教授への報告用の表と図(段階6〜8 の保存済みの結果を集めるだけ。計算はし直さない)

  図1 主な結果(段階6の実験D: main・1次差分・MiniRocket)の置換検定の帰無分布と観測値
  図2 main(82件)での比較: 非時系列の基準線(段階6)と、MiniRocket の入力の形(生の値・z 正規化・1次差分、段階7)
  図3 感度分析(段階6の F3)
  表  report_table.csv(報告に使う設定の値をまとめたもの)

実行(保存済みの CSV / npy を読むだけなので、どの PC でもよい):
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\report_prof\\make_report.py
入力: rocket/outputs/stage6/, stage7/, stage7/pooling/, stage8/
出力: rocket/outputs/report_prof/
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
OUTPUTS = os.path.join(ROCKET_DIR, 'outputs')
OUT_DIR = os.path.join(OUTPUTS, 'report_prof')

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
C_MR, C_BASE, C_NULL = '#2a78d6', '#9a9890', '#e34948'

# 報告に使う設定(表の並び順)。(ファイル, config, 区分, 表示名)
ROWS = [
    ('stage6', 'D_main_diff', '主な結果', 'MiniRocket・1次差分'),
    ('stage6', 'base_vwc_stats_main', '基準線(main)', 'VWC の要約統計 + LogReg'),
    ('stage6', 'base_seed_doy_main', '基準線(main)', '播種日 + LogReg'),
    ('stage6', 'base_irrigated_main', '基準線(main)', '灌水の有無 + LogReg'),
    ('stage7', 'G1_main_raw', '入力の形(main)', 'MiniRocket・生の値'),
    ('stage7', 'G2_main_znorm', '入力の形(main)', 'MiniRocket・z 正規化'),
    ('stage6', 'F3_main_obs_diff', '感度分析', '実測開花日のみ(60件)'),
    ('stage6', 'F3_main_no_kakogawa_diff', '感度分析', '加古川を除く(76件)'),
    ('stage6', 'F3_main_no_irrigation_diff', '感度分析', '灌水なしのみ(70件)'),
    ('stage6', 'E_main_common_diff', '揃え方(共通77件)', '開花起点 0〜+59'),
    ('stage6', 'E_dap_common_diff', '揃え方(共通77件)', 'dap 50–109'),
    ('stage6', 'F4_main_labelyear_diff', 'ラベル', 'label_year(58件)'),
    ('stage7/pooling', 'H_diff_ppv', 'Rocket(main・1次差分)', 'Rocket・PPV(normalise=False)'),
    ('stage7/pooling', 'H_diff_ppvmax', 'Rocket(main・1次差分)', 'Rocket・PPV+max(normalise=False)'),
    ('stage8', 'R1_ppv', 'Rocket(main・1次差分)', 'Rocket・PPV(系列ごとの z 正規化)'),
    ('stage8', 'R2_ppv', 'Rocket(main・1次差分)', 'Rocket・PPV(全体の SD で割る)'),
]
COLS = ['n', 'n_seeds', 'n_perm', 'auc_mean', 'auc_seed_min', 'auc_seed_max', 'auc_null_q95', 'auc_p',
        'bacc_mean', 'bacc_null_q95', 'bacc_p']


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def load_table():
    cache = {}
    rows = []
    for src, cfg, group, name in ROWS:
        if src not in cache:
            cache[src] = pd.read_csv(os.path.join(OUTPUTS, src, 'summary.csv'), encoding='utf-8-sig')
        r = cache[src].set_index('config').loc[cfg]
        rows.append({'group': group, 'name': name, 'source': src, 'config': cfg, **{c: r.get(c, np.nan) for c in COLS}})
    t = pd.DataFrame(rows)
    for m in ['auc', 'bacc']:
        t[f'{m}_above_q95'] = t[f'{m}_mean'] > t[f'{m}_null_q95']
    return t


def plot_null(t):
    null = np.load(os.path.join(OUTPUTS, 'stage6', 'null_D_main_diff.npy'))
    d = t.set_index('config').loc['D_main_diff']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (j, m, lab) in zip(axs, [(1, 'auc', 'AUC'), (0, 'bacc', 'balanced accuracy')]):
        ax.hist(null[:, j], bins=np.arange(0.40, 0.66, 0.0075), color=C_BASE, edgecolor='white', lw=0.6)
        q95, obs, p = d[f'{m}_null_q95'], d[f'{m}_mean'], d[f'{m}_p']
        ax.axvline(q95, color=MUTED, ls='--', lw=1.2)
        ax.axvline(obs, color=C_MR, lw=2.2)
        ymax = ax.get_ylim()[1] * 1.45  # 文字を棒の上の余白に置く
        ax.set_ylim(0, ymax)
        box = dict(facecolor='white', edgecolor='none', pad=1.5)
        right = obs > q95  # 観測値の文字は外側、95%点の文字は内側に出す
        ax.text(q95, ymax * 0.97, f' 95%点 {q95:.3f} ', color=MUTED, fontsize=9, va='top',
                ha='right' if right else 'left', bbox=box)
        ax.text(obs, ymax * 0.88, f' 観測値 {obs:.3f}\n p = {p:.3f} ', color=C_MR, fontsize=10, va='top',
                ha='left' if right else 'right', bbox=box)
        ax.set_title(lab, fontsize=11, color=INK, loc='left')
        ax.set_xlabel(f'{lab}(置換1000回、ラベルを市町村×年の中で並べ替え)', fontsize=9, color=MUTED)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        style(ax)
    axs[0].set_ylabel('回数', color=MUTED)
    fig.suptitle('主な結果: main(82件)・1次差分・MiniRocket と、偶然で出る値の分布', fontsize=12, color=INK,
                 x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_null_main.png'), dpi=150)
    plt.close(fig)


def plot_dots(t, configs, title, fname, names=None, show_base=True):
    """設定ごとの平均(点)、乱数10通りの範囲(横線)、置換の95%点(縦の短い線)。"""
    d = t.set_index('config').loc[configs].reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.55 * len(d) + 1.8), sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, m, lab in zip(axs, ['auc', 'bacc'], ['AUC', 'balanced accuracy']):
        ax.axvline(0.5, color=MUTED, lw=0.8)
        for yi, (_, r) in zip(y, d.iterrows()):
            mr = r.group != '基準線(main)'
            c = C_MR if mr else C_BASE
            if m == 'auc' and mr and not np.isnan(r.auc_seed_min):
                ax.plot([r.auc_seed_min, r.auc_seed_max], [yi, yi], color=c, lw=3, alpha=0.4, solid_capstyle='round')
            ax.plot(r[f'{m}_null_q95'], yi, marker='|', ms=16, mew=2, color=C_NULL)
            ax.plot(r[f'{m}_mean'], yi, 'o', ms=8, color=c, mec='white', mew=1.2)
            ax.text(r[f'{m}_mean'], yi + 0.28, f'{r[f"{m}_mean"]:.3f}', ha='center', fontsize=8, color=INK)
        ax.set_xlim(0.38, 0.68)
        ax.set_title(lab, fontsize=11, color=INK, loc='left')
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        style(ax)
    names = names or [f'{n}(基準線)' if g.startswith('基準線') else n for n, g in zip(d.name, d.group)]
    axs[0].set_yticks(y, names, fontsize=9, color=INK)
    axs[0].set_ylim(-0.7, len(d) - 0.4)
    handles = [Line2D([], [], marker='o', color=C_MR, ls='', ms=8), Line2D([], [], marker='o', color=C_BASE, ls='', ms=8),
               Line2D([], [], marker='|', color=C_NULL, ls='', ms=14, mew=2)]
    labels = ['MiniRocket(太い線 = 乱数10通りの範囲、AUC のみ)', '非時系列の基準線', '置換検定の95%点(これより右なら偶然の水準を超える)']
    keep = [0, 1, 2] if show_base else [0, 2]
    fig.legend([handles[i] for i in keep], [labels[i] for i in keep], loc='lower center', ncol=3, frameon=False,
               fontsize=9)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t = load_table()
    t.to_csv(os.path.join(OUT_DIR, 'report_table.csv'), index=False, encoding='utf-8-sig')
    plot_null(t)
    plot_dots(t, ['base_vwc_stats_main', 'base_seed_doy_main', 'base_irrigated_main',
                  'G1_main_raw', 'G2_main_znorm', 'D_main_diff'],
              'main(82件)での比較: 非時系列の基準線と、MiniRocket の入力の形', 'fig2_main_comparison.png')
    plot_dots(t, ['D_main_diff', 'F3_main_obs_diff', 'F3_main_no_kakogawa_diff', 'F3_main_no_irrigation_diff'],
              '感度分析(MiniRocket・1次差分)', 'fig3_sensitivity.png',
              names=['main(82件、主な結果)', '実測開花日のみ(60件)', '加古川を除く(76件)', '灌水なしのみ(70件)'],
              show_base=False)

    paired = pd.read_csv(os.path.join(OUTPUTS, 'stage7', 'paired_diff_summary.csv'), encoding='utf-8-sig')
    paired_e = pd.read_csv(os.path.join(OUTPUTS, 'stage6', 'expE_paired_summary.csv'), encoding='utf-8-sig')
    pd.set_option('display.width', 250)
    print(t[['group', 'name', 'n', 'n_perm', 'auc_mean', 'auc_null_q95', 'auc_p', 'bacc_mean', 'bacc_null_q95',
             'bacc_p']].round(3).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    print(paired_e.round(3).to_string(index=False))
    print(f'出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
