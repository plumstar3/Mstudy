"""
calc_gdd.py
============================================================
大豆の有効積算温度（GDD: Growing Degree Days）算出スクリプト

【計算方式】
  修正平均法（Modified Average Method）
  - 日最高気温・最低気温をそれぞれ生理学的上下限でクランプした後、
    その平均から基準温度を引いて日次GDDを求める。
  - 負値は 0 に切り上げ（下限フロア）
  - 時系列順の累積和で累積GDDを算出

【大豆の生理学的パラメータ】
  T_base  : 10.0 ℃  （発育零点 — これを下回ると発育が停止）
  T_upper : 30.0 ℃  （上限温度 — これを超えても発育速度は増加しない）

【forループ不使用】
  pandasのclip / np.maximum でベクトル演算のみ使用
"""

import numpy as np
import pandas as pd

# ── 生理学的定数 ───────────────────────────────────────────────────────────────
T_BASE  = 10.0   # 発育零点 [℃]
T_UPPER = 30.0   # 上限温度（キャップ） [℃]


# ── GDD 算出関数 ───────────────────────────────────────────────────────────────

def calc_gdd(df: pd.DataFrame,
             t_max_col: str = 'T_max',
             t_min_col: str = 'T_min',
             t_base: float  = T_BASE,
             t_upper: float = T_UPPER) -> pd.DataFrame:
    """修正平均法（Modified Average Method）で日次・累積GDDを算出する。

    forループを使わず、pandas/numpy のベクトル演算のみで実装。

    Args:
        df        : 入力 DataFrame。Date, T_max, T_min 列を含むこと。
        t_max_col : 日最高気温の列名（デフォルト: 'T_max'）
        t_min_col : 日最低気温の列名（デフォルト: 'T_min'）
        t_base    : 基準温度 [℃]（発育零点）
        t_upper   : 上限温度 [℃]（キャップ）

    Returns:
        計算過程の全列を含む新しい DataFrame
    """
    result = df.copy()

    # ── Step 1: 日最高気温（T_max）の補正 ──────────────────────────────────────
    # ・T_upper を超えた場合 → T_upper に丸める（発育速度の飽和）
    # ・T_base を下回った場合 → T_base に引き上げる（発育ゼロの保証）
    result['補正後T_max'] = result[t_max_col].clip(lower=t_base, upper=t_upper)

    # ── Step 2: 日最低気温（T_min）の補正 ──────────────────────────────────────
    # ・T_base を下回った場合 → T_base に引き上げる
    # ・T_min には上限キャップを設けない（修正平均法の仕様）
    result['補正後T_min'] = result[t_min_col].clip(lower=t_base)

    # ── Step 3: 日次GDDの算出 ──────────────────────────────────────────────────
    # 日次GDD = (補正後T_max + 補正後T_min) / 2 − T_base
    result['日次GDD'] = (result['補正後T_max'] + result['補正後T_min']) / 2.0 - t_base

    # ── Step 4: 下限フロア（負値を 0 にクランプ） ──────────────────────────────
    # 補正後でも平均がT_baseを下回るケースを除外（理論上は起きないが念のため）
    result['日次GDD'] = result['日次GDD'].clip(lower=0.0)

    # ── Step 5: 累積GDDの算出（時系列順 cumsum）───────────────────────────────
    # Date 列で昇順ソートしてから累積和を取る（入力順が保証されている場合も明示的に実施）
    result = result.sort_values('Date').reset_index(drop=True)
    result['累積GDD'] = result['日次GDD'].cumsum()

    return result


# ── サンプルデータ ─────────────────────────────────────────────────────────────

def make_sample_data() -> pd.DataFrame:
    """検証用サンプルデータを返す。

    各行のシナリオ:
      2026-06-01: 理想的な日（T_max=26℃, T_min=14℃）
      2026-06-02: 朝晩が冷え込む日（T_min=6℃ → T_base未満）
      2026-06-03: 日中猛暑日（T_max=36℃ → T_upper超過）
      2026-06-04: 生育不可能な冬日（T_max=8℃, T_min=2℃ → 両方T_base未満）
      2026-06-05: 朝晩冷涼・日中普通（T_min=5℃ → T_base未満）
    """
    return pd.DataFrame({
        'Date' : pd.to_datetime(['2026-06-01', '2026-06-02', '2026-06-03',
                                  '2026-06-04', '2026-06-05']),
        'T_max': [26.0, 22.0, 36.0,  8.0, 15.0],
        'T_min': [14.0,  6.0, 24.0,  2.0,  5.0],
    })


# ── 結果表示ヘルパー ───────────────────────────────────────────────────────────

def print_result(result: pd.DataFrame) -> None:
    """計算結果を整形して標準出力に表示する。"""
    display_cols = ['Date', 'T_max', 'T_min',
                    '補正後T_max', '補正後T_min', '日次GDD', '累積GDD']

    print('=' * 72)
    print('  有効積算温度（GDD）計算結果（修正平均法）')
    print(f'  T_base={T_BASE}℃  /  T_upper={T_UPPER}℃')
    print('=' * 72)

    # 表示用に日付を文字列化（幅を揃えるため）
    disp = result[display_cols].copy()
    disp['Date'] = disp['Date'].dt.strftime('%Y-%m-%d')

    # 数値列を小数点1桁に整形
    num_cols = ['T_max', 'T_min', '補正後T_max', '補正後T_min', '日次GDD', '累積GDD']
    disp[num_cols] = disp[num_cols].round(1)

    print(disp.to_string(index=False))
    print('=' * 72)

    # ── 行ごとの補正ポイントを注釈表示 ─────────────────────────────────────
    print('\n【補正ポイントの解説】')
    scenarios = [
        ('2026-06-01', '理想的な日: 補正なし。GDD = (26+14)/2 - 10 = 10.0'),
        ('2026-06-02', '冷え込む日: T_min=6 → 10 に引き上げ。GDD = (22+10)/2 - 10 = 6.0'),
        ('2026-06-03', '猛暑日    : T_max=36 → 30 に丸め。GDD = (30+24)/2 - 10 = 17.0'),
        ('2026-06-04', '冬日      : T_max=8, T_min=2 → 両方10に引き上げ。GDD = (10+10)/2 - 10 = 0.0'),
        ('2026-06-05', '朝晩冷涼  : T_min=5 → 10 に引き上げ。GDD = (15+10)/2 - 10 = 2.5'),
    ]
    for date_str, note in scenarios:
        print(f'  {date_str}: {note}')
    print()


# ── メインエントリ ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # サンプルデータ生成
    df_sample = make_sample_data()

    print('【入力データ】')
    print(df_sample.to_string(index=False))
    print()

    # GDD 算出
    df_result = calc_gdd(df_sample)

    # 結果表示
    print_result(df_result)

    # ── 追加検証: 累積GDDの最終値と平均日次GDDを表示 ────────────────────────
    total_gdd  = df_result['累積GDD'].iloc[-1]
    mean_daily = df_result['日次GDD'].mean()
    print(f'サマリー:')
    print(f'  累積GDD（5日間合計） : {total_gdd:.1f} degC-day')
    print(f'  平均日次GDD          : {mean_daily:.2f} degC-day/日')
