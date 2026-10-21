# rocket/ — 土壌水分量(VWC)時系列による多収・低収分類

## 研究の目的

土壌水分量(VWC)の時系列を入力として、大豆圃場の多収(収量上位)と低収(収量下位)を分類し、**その精度を上げていく**こと。手法は ROCKET を軸にする。今後の作業はこのディレクトリ内で行う。

進捗: データセット構築・品質確認・可視化・ROCKET デモまで完了。**本番データでの学習・評価は未着手**。

## 実行環境とコマンド

専用の venv **`rocket\venv_rocket`**(Python 3.12.10)を使う。`amd` や anaconda は使わない。パッケージは `rocket/requirements.txt` に固定している(主なもの: sktime 1.2.0, scikit-learn 1.5.1, numpy 2.5, pandas 3.0, numba 0.68, matplotlib 3.11)。sktime と scikit-learn は、それまで使っていた anaconda と同じ版に合わせてある。パッケージを追加したら `pip freeze > rocket\requirements.txt` で更新する。

```powershell
# 環境の再作成
C:\Users\amilu\AppData\Local\Programs\Python\Python312\python -m venv rocket\venv_rocket
rocket\venv_rocket\Scripts\python -m pip install -r rocket\requirements.txt

# データセット再構築(この順で実行)
rocket\venv_rocket\Scripts\python rocket\scripts\build_soilmoisture_dataset.py
rocket\venv_rocket\Scripts\python rocket\scripts\add_yield_labels.py

# 地域×年ごとの VWC 推移の図(--x date で横軸を日付に)
rocket\venv_rocket\Scripts\python rocket\scripts\plot_vwc_by_region.py
```

既存のデータ構築スクリプトは anaconda(pandas 2.2)で作ったもの。venv_rocket(pandas 3.0)では、`rocket_demo.py` 以外はまだ実行して確かめていない。

## ディレクトリの約束

- `scripts/` — 実行する入口。新しい評価・解析スクリプトはここに置く
- `src/` — `scripts/` から import する共通部品(現在は空)。複数スクリプトで使う処理(データ読み込み・系列整形・CV 分割など)はここに切り出す
- `docs/papers/` — 参考論文の PDF(著作権のため Git 管理外)。`ROCKET.pdf` が ROCKET 原論文(Dempster, Petitjean, Webb. *Data Mining and Knowledge Discovery* 34:1454–1495, 2020)。Read ツールでは開けないので `pdftotext -f <開始頁> -l <終了頁> rocket/docs/papers/ROCKET.pdf -` で本文を読む。手法の仕様(カーネルのサンプリング、PPV、系列長の扱い)はコードより論文を根拠にする
- `docs/survey/` — ユーザーが Deep Research で作った調査報告(.docx)。`ROCKET系列アルゴリズムの比較分析.docx`(MiniRocket / MultiRocket / Hydra などの比較)と `大豆収量分類の文献調査.docx`(土壌水分による収量分類の先行研究)の2つ。Read ツールでは開けないので、zip として `word/document.xml` の `<w:t>` 要素からテキストを取り出す。**AI が作った二次資料なので、引用には一次文献で確かめてから使う**(出典にブログや Wiki も混ざっている)
- `docs/experiments.md` — 実験記録。**実験を回したら、その場で EXP-XXX を追記し、一覧表も更新する**(書式はファイル内のテンプレート)
- `docs/decisions.md` — 方針の決定とその理由(D-XXX)。ユーザーと方針を決めたらここに追記し、結論だけを本ファイルに反映する
- `docs/sessions/`, `docs/meetings/` — 作業セッションのまとめと打ち合わせメモ。`_template.md` をコピーして `YYYY-MM-DD_*.md` で保存する
- `external/rocket/` — ROCKET 公式実装(GPL-3.0)。入れ子 Git を避けるため `.git` を削除して取り込んだもの。**編集しない**
- `data/processed/`, `outputs/` — 生成物。スクリプト内のパスは `__file__` 基準で解決する(既存スクリプトの `ROCKET_DIR` / `ROOT_DIR` の書き方に合わせる)
- 元データはリポジトリルートの `data/processed/FieldData_fieldid.db`(`Questionaire`, `SolidMoisture` テーブル)と `outputs/reverse_geocode/field_addresses.csv`。これらは読むだけで書き換えない

## データセット(`data/processed/`)

| ファイル | 1行の単位 | 行数 | 主な列 |
|---|---|---|---|
| `soilmoisture_timeseries.csv` | 圃場×年×観測日 | 30,040 | `field_id, year, date, dap, VWC` |
| `field_year_static.csv` | 圃場×年 | 304 | `field_id, year, muniCd, yield, seed_date, harvest_date, breed, irrigation, lat, lon, depth, between_lines, between_stocks, n_obs` |
| `field_year_labeled.csv` | 圃場×年 | 304 | 上記 + `label_muniyear`, `label_year` |

- 結合キーは `(field_id, year)`。`muniCd` は圃場ごとに1つなので `field_id` だけで結合する
- CSV は `utf-8-sig`。`muniCd` は `dtype=str` で読む
- `dap` = 観測日 − 播種日
- VWC があるのは **2016〜2018年のみ**。圃場×年は 2016=143 / 2017=115 / 2018=46
- 系列長は 32〜151 日(中央値 102)で**不揃い**。観測間隔はほぼ毎日で、欠けや重複はない

## ラベル(0=低収, 1=多収, 空欄=中間 or 対象外)

グループ内 n 件のとき、上位と下位から k = round(n/3) 件ずつ取る。

| 列 | 分割単位 | 件数 | 備考 |
|---|---|---|---|
| `label_muniyear` | 同一 `muniCd`×年 | 94(47/47) | **主ラベル**。4圃場以上のグループのみ。地域×年グループ106個の大半は2圃場以下。2018年は6件のみ |
| `label_year` | 同一年 | 202(101/101) | 比較用。地域差を除けない |

両方のラベルが付いた圃場での一致率は約0.90。両ラベルで同じモデルを学習して精度を比べ、地域差ではなく土壌水分の効果を拾えているかを確かめる想定。

## 評価方針(決定済み)

- **CV は `muniCd` 単位の GroupKFold**。同じ地域の圃場は気象や土壌が似ていて、分けると精度が過大になる(データ漏洩)ため。ランダム分割は使わない。`label_muniyear` が付く地域は21で、1地域あたり2〜10件。fold 数は少なめになる
- 基本構成は `Rocket`(変換器) → `StandardScaler(with_mean=False)` → `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`。`RocketClassifier` の内部構成と同じ。**ただし sktime 1.2.0 の `Rocket` 変換器も、実際には不揃いな系列長を扱えない**(タグ `capability:unequal_length` は True なのに、nested_univ / pd-multiindex / df-list のどの形式でも ValueError になる。EXP-001)。長さが揃っていなくても動くのは、`MiniRocketMultivariateVariable` と、公式実装を1系列ずつ適用する方法だけ。公式の `external/rocket/code/rocket_functions.py` は NumPy 2 で削除された `np.NINF` を使っているので、**import した時点で失敗する**。代わりに `src/rocket_kernels.py` を使う。ここには `-np.inf` に直した公式実装の写しと、前処理(`znorm`, `pad_with_noise`, `resample`, `apply_kernels_variable`)がある。ROCKET は入力が z 正規化されている前提なので、系列は本来の長さで z 正規化してから長さを揃える。ArrowHead での比較(EXP-002)では、時間軸の位置を揃えたまま長さを合わせる方法が一番よかった。補間とノイズパディングは精度が下がった。VWC での対応方法は**未決定**(decisions.md の D-005 を参照)
- 手法は当面 ROCKET(`rocket_transform="rocket"`)のみ。sktime 1.2.0 には `MiniRocketClassifier` / `MultiRocketClassifier` が無い(使う場合は `from sktime.transformations.rocket import Rocket, MiniRocket, MultiRocket` の変換器か。`sktime.transformations.panel` は非推奨 `RocketClassifier(rocket_transform=...)`)
- ROCKET の特徴量は各カーネルの「最大値」と「PPV(正の値の割合)」で、2 × num_kernels 次元(10000本なら20000次元)。94件に対して非常に高次元なので過学習に注意する

## データの注意点(`outputs/quality_check.csv` に圃場×年ごとのフラグあり)

- 除外済み: 観測数30未満の2件(`field_id=5` 2016年, `field_id=580` 2018年)。基準は `build_soilmoisture_dataset.py` の `MIN_OBS = 30`
- 除外していないが要注意:
  - 栽培期間に対する観測期間のカバー率が50%未満: 28件
  - 観測開始が播種後30日超: 約3割
  - 栽培期間の記録が異常(収穫日が翌年1/5): `field_id` 568, 571
  - 収量の外れ値: `field_id` 525, 248, 566(実際に高収の可能性もある)
  - VWC の負値は残してある
  - 兵庫県加古川市(2016年)の一部は終盤に VWC が0へ落ちる(観測欠落の疑い)
- 可視化からわかったこと:
  - 多収と低収の分かれ方は地域によって違う
  - VWC の水準の高低と収量の関係は地域によって逆向きになる
  - 同じ地域・年の圃場は同じ日に降雨に反応する
  - センサごとの水準差が大きい

  このため、生の水準より**変動パターン**が手がかりになる可能性がある。

## 未決定事項(作業前にユーザーと相談する)

1. 系列の揃え方(日付基準か dap 基準か、共通に使う期間をどこにするか)。D-005 の選び直しと一体で決める。EXP-002 では時間軸の位置を保って揃える方法が最良。候補として**積算 GDD(生育ステージの代わり)で揃える**案もある(`outputs/gdd/gdd_daily.csv` に日ごとの積算 GDD があり、94件すべてそろっている。文献調査では R5〜R6 期の水分ストレスの影響が大きいとされる)
2. 除外基準(カバー率が低い圃場、`field_id` 568/571)
3. 入力の形(生の VWC、圃場ごとの標準化、差分)。差分はセンサーごとの水準差を消せる(MultiRocket も1次差分を使う)
4. VWC=0 への落ち込みを欠測として扱うか
5. ラベルが94件と少ない問題への対処
6. 年の偏りと、相対ラベル(地域×年内の上下1/3)の扱い
7. `Questionaire.depth`(94件で 5〜28.5)の意味。センサーの深さか作土の深さか、ユーザーに確認中。センサーの深さなら、VWC の水準差の原因かもしれない

## 次のステップ

1. 実データで、dap 基準 / 積算 GDD 基準の共通区間ごとに、区間の長さと残る件数(94件中)の関係を集計する。その結果を見て、未決定事項1と D-005 をユーザーと決める
2. `scripts/` に本番の評価スクリプトを作る。`src/rocket_kernels.py` の公式 ROCKET(z 正規化してから長さを揃える) → StandardScaler(with_mean=False) → RidgeClassifierCV、GroupKFold(groups=muniCd)、入力は `label_muniyear` が付いた94件。シードを変えて繰り返し、平均±標準偏差で報告する
3. (優先度低)`scripts/rocket_demo.py` を公式分割に直す。公式分割での値(0.806、論文値 0.8143)は EXP-002 の基準ですでに確認済み
