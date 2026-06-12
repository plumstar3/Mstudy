"""
map_yield.py
============
FieldData_fieldid.db の Questionaire テーブルから lat / lon / yield を読み込み、
各地点の収量を地図上にマッピングした図を生成する。

出力:
  outputs/yield_map_all.png       - 全年度まとめ
  outputs/yield_map_<year>.png    - 年度別

依存:
  pip install matplotlib cartopy pandas
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

# ── Cartopy（地図描画）があれば使う、なければ scatter のみ ──────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("[INFO] cartopy が見つかりません。シンプルな散布図モードで描画します。")
    print("       pip install cartopy  でインストールできます。")

# ─────────────────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH  = os.path.join(os.path.dirname(__file__), "data", "processed", "FieldData_fieldid.db")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs", "yield_maps")
os.makedirs(OUT_DIR, exist_ok=True)

COLORMAP  = "RdYlGn"   # 低収量=赤 / 高収量=緑
POINT_SIZE = 60
ALPHA      = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────────────────────
def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT lat, lon, yield, year, place
        FROM   Questionaire
        WHERE  lat   IS NOT NULL
          AND  lon   IS NOT NULL
          AND  yield IS NOT NULL
        ORDER BY year, place
        """,
        conn,
    )
    conn.close()
    print(f"[INFO] ロードしたレコード数: {len(df)}  (年度: {sorted(df['year'].unique())})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 地図描画ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────
def _japan_extent(df: pd.DataFrame, margin: float = 0.5):
    """データ範囲から地図の表示範囲を計算（日本に自動フィット）。"""
    lon_min = df["lon"].min() - margin
    lon_max = df["lon"].max() + margin
    lat_min = df["lat"].min() - margin
    lat_max = df["lat"].max() + margin
    return lon_min, lon_max, lat_min, lat_max


def _norm_and_cmap(yield_series: pd.Series, vmin=None, vmax=None):
    vmin = vmin if vmin is not None else yield_series.min()
    vmax = vmax if vmax is not None else yield_series.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps[COLORMAP]
    return norm, cmap


def _add_colorbar(fig, ax, norm, cmap, label="収量 (kg/10a)"):
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02, shrink=0.85)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    return cbar


def draw_map_cartopy(df: pd.DataFrame, title: str, save_path: str,
                     vmin=None, vmax=None):
    """Cartopy を使った地図描画（推奨）。"""
    norm, cmap = _norm_and_cmap(df["yield"], vmin, vmax)
    lon_min, lon_max, lat_min, lat_max = _japan_extent(df)

    fig = plt.figure(figsize=(10, 8), facecolor="#1a1a2e")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor("#0f3460")

    # 地図フィーチャー
    ax.add_feature(cfeature.LAND,       facecolor="#16213e", edgecolor="none")
    ax.add_feature(cfeature.OCEAN,      facecolor="#0f3460")
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6, edgecolor="#a0a0c0")
    ax.add_feature(cfeature.BORDERS,    linewidth=0.4, edgecolor="#606080")
    ax.add_feature(cfeature.LAKES,      facecolor="#0f3460", edgecolor="none", alpha=0.6)
    ax.add_feature(cfeature.RIVERS,     edgecolor="#2060a0", linewidth=0.3, alpha=0.5)

    # グリッドライン
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": "#c0c0d0", "size": 8}
    gl.ylabel_style = {"color": "#c0c0d0", "size": 8}

    # 収量散布図
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["yield"], cmap=cmap, norm=norm,
        s=POINT_SIZE, alpha=ALPHA,
        transform=ccrs.PlateCarree(),
        edgecolors="white", linewidths=0.4, zorder=5
    )

    _add_colorbar(fig, ax, norm, cmap)

    ax.set_title(title, fontsize=15, color="white", pad=14,
                 fontweight="bold")
    fig.text(0.5, 0.01,
             f"N={len(df)}  |  収量範囲: {df['yield'].min():.0f}〜{df['yield'].max():.0f} kg/10a"
             f"  |  平均: {df['yield'].mean():.1f}  |  中央値: {df['yield'].median():.1f}",
             ha="center", color="#a0a0c0", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def draw_map_simple(df: pd.DataFrame, title: str, save_path: str,
                    vmin=None, vmax=None):
    """Cartopy なしのシンプル散布図（フォールバック）。"""
    norm, cmap = _norm_and_cmap(df["yield"], vmin, vmax)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#0f3460")

    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["yield"], cmap=cmap, norm=norm,
        s=POINT_SIZE, alpha=ALPHA,
        edgecolors="white", linewidths=0.4
    )

    _add_colorbar(fig, ax, norm, cmap)

    # 軸スタイル
    ax.set_xlabel("経度 (°E)", color="#c0c0d0", fontsize=11)
    ax.set_ylabel("緯度 (°N)", color="#c0c0d0", fontsize=11)
    ax.tick_params(colors="#c0c0d0")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f°"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f°"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#4a4a6a")
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    ax.set_title(title, fontsize=15, color="white", pad=14, fontweight="bold")
    fig.text(0.5, 0.01,
             f"N={len(df)}  |  収量範囲: {df['yield'].min():.0f}〜{df['yield'].max():.0f} kg/10a"
             f"  |  平均: {df['yield'].mean():.1f}  |  中央値: {df['yield'].median():.1f}",
             ha="center", color="#a0a0c0", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


def draw_map(df, title, save_path, vmin=None, vmax=None):
    """Cartopy の有無で描画関数を自動切り替え。"""
    if HAS_CARTOPY:
        draw_map_cartopy(df, title, save_path, vmin, vmax)
    else:
        draw_map_simple(df, title, save_path, vmin, vmax)


# ─────────────────────────────────────────────────────────────────────────────
# 年度別パネル図（1枚にまとめる）
# ─────────────────────────────────────────────────────────────────────────────
def draw_panel(df: pd.DataFrame, save_path: str):
    """全年度を2×2パネルに並べた比較図。"""
    years = sorted(df["year"].unique())
    vmin  = df["yield"].min()
    vmax  = df["yield"].max()
    norm, cmap = _norm_and_cmap(df["yield"], vmin, vmax)

    ncols = 2
    nrows = int(np.ceil(len(years) / ncols))

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(14, 5 * nrows),
            subplot_kw={"projection": proj},
            facecolor="#1a1a2e"
        )
    else:
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, 5 * nrows),
                                 facecolor="#1a1a2e")

    axes = axes.flatten()
    lon_min, lon_max, lat_min, lat_max = _japan_extent(df)

    for i, year in enumerate(years):
        ax = axes[i]
        sub = df[df["year"] == year]

        if HAS_CARTOPY:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                          crs=ccrs.PlateCarree())
            ax.set_facecolor("#0f3460")
            ax.add_feature(cfeature.LAND,      facecolor="#16213e", edgecolor="none")
            ax.add_feature(cfeature.OCEAN,     facecolor="#0f3460")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#a0a0c0")
            ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#606080")
            gl = ax.gridlines(draw_labels=True, linewidth=0.2,
                              color="gray", alpha=0.4, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"color": "#c0c0d0", "size": 7}
            gl.ylabel_style = {"color": "#c0c0d0", "size": 7}
            ax.scatter(sub["lon"], sub["lat"],
                       c=sub["yield"], cmap=cmap, norm=norm,
                       s=40, alpha=ALPHA,
                       transform=ccrs.PlateCarree(),
                       edgecolors="white", linewidths=0.3, zorder=5)
        else:
            ax.set_facecolor("#0f3460")
            ax.scatter(sub["lon"], sub["lat"],
                       c=sub["yield"], cmap=cmap, norm=norm,
                       s=40, alpha=ALPHA,
                       edgecolors="white", linewidths=0.3)
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
            ax.set_xlabel("経度", color="#c0c0d0", fontsize=8)
            ax.set_ylabel("緯度", color="#c0c0d0", fontsize=8)
            ax.tick_params(colors="#c0c0d0", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#4a4a6a")
            ax.grid(True, linewidth=0.2, color="gray", alpha=0.3, linestyle="--")

        ax.set_title(f"{year}年  (N={len(sub)}, 平均={sub['yield'].mean():.1f} kg/10a)",
                     color="white", fontsize=12, fontweight="bold", pad=8)

    # 余ったパネルを非表示
    for j in range(len(years), len(axes)):
        axes[j].set_visible(False)

    # 右端にカラーバー専用の axes を作成（マップと重ならないよう余白を確保）
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.78])  # [left, bottom, width, height]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("収量 (kg/10a)", fontsize=12, color="white", labelpad=10)
    cbar.ax.tick_params(labelsize=9, colors="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar_ax.set_facecolor("#1a1a2e")

    fig.suptitle("大豆収量マップ（年度別比較）", fontsize=18,
                 color="white", fontweight="bold", y=1.01)
    fig.text(0.44, -0.01,
             f"全データ: N={len(df)}, 収量範囲: {vmin:.0f}〜{vmax:.0f} kg/10a",
             ha="center", color="#a0a0c0", fontsize=10)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 日本語フォント設定（Windows / Linux / macOS）
    import matplotlib
    for font in ["Yu Gothic", "Meiryo", "IPAexGothic", "Hiragino Sans", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = font
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    df = load_data(DB_PATH)
    vmin = df["yield"].quantile(0.02)  # 外れ値を除いたカラースケール
    vmax = df["yield"].quantile(0.98)

    # ① 全年度まとめマップ
    draw_map(df, "大豆収量マップ（全年度: 2015〜2018）",
             os.path.join(OUT_DIR, "yield_map_all.png"), vmin, vmax)

    # ② 年度別マップ
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        draw_map(sub, f"大豆収量マップ（{year}年）",
                 os.path.join(OUT_DIR, f"yield_map_{year}.png"), vmin, vmax)

    # ③ 年度別パネル比較図
    draw_panel(df, os.path.join(OUT_DIR, "yield_map_panel.png"))

    print("\n完了！出力先:", OUT_DIR)


if __name__ == "__main__":
    main()
