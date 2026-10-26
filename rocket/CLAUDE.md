# rocket/ — 土壌水分量(VWC)時系列による多収・低収分類

## 研究の目的

土壌水分量(VWC)の時系列を入力として、大豆圃場の多収(収量上位)と低収(収量下位)を分類し、**その精度を上げていく**こと。手法は ROCKET を軸にする。今後の作業はこのディレクトリ内で行う。

進捗: データセット構築・品質確認・可視化・ROCKET デモ、段階1(データの棚卸し)、段階2(非時系列の基準線、EXP-003)、段階3(揃え方の起点と窓の集計、EXP-004, 005)、段階4(入力データ、EXP-006)、段階5(MiniRocket による初めての評価、EXP-007)まで完了。

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

- `scripts/` — 実行する入口(データ構築・可視化など)
- `analysis/stageN/` — 段階ごとの解析スクリプトと `report.md`。結果の CSV・図は `outputs/stageN/` に出す
- `src/` — 共通部品。`sys.path` に rocket/ を入れて `from src.xxx import ...` で使う。`rocket_kernels.py`(公式 ROCKET の写しと前処理)、`evaluation.py`(評価の仕組み。下の「評価方針」を参照)、`phenology.py`(日長・開花日の推定)、`windows.py`(窓の切り出しと入力の形の変換)がある。複数スクリプトで使う処理はここに切り出す
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
- 系列長は 32〜151 日(中央値 102)で**不揃い**。観測は毎日で重複はない。日の欠けがあるのは全304件中 `field_id=90`(2016年、11日)の1件だけで、`label_muniyear` 付き94件には欠けが無い
- 開花日は `Questionaire.flower_date` にあり(`field_year_static.csv` には入れていない)、`label_muniyear` 付き94件中66件に値がある。欠測28件のうち10件は加古川(28210)で、この地域は全件が欠測。播種から開花までは35〜65日(中央値45.5)

## ラベル(0=低収, 1=多収, 空欄=中間 or 対象外)

グループ内 n 件のとき、収量の上位と下位から k = round(n/3) 件ずつ取る(中央値で分けるのではない。中間は空欄、同順位は field_id 昇順)。`label_muniyear` は**市町村×年の中での相対評価**で、各集団で多収と低収が同数になる(94件では29集団すべてで同数)。このため地域と年だけではラベルを当てられない。ただし窓で絞ると片方のクラスだけが落ちる集団が出て、釣り合いが崩れる(EXP-003)

| 列 | 分割単位 | 件数 | 備考 |
|---|---|---|---|
| `label_muniyear` | 同一 `muniCd`×年 | 94(47/47) | **主ラベル**。4圃場以上のグループのみ。地域×年グループ106個の大半は2圃場以下。2018年は6件のみ |
| `label_year` | 同一年 | 202(101/101) | 比較用。地域差を除けない |

両方のラベルが付いた圃場での一致率は約0.90。両ラベルで同じモデルを学習して精度を比べ、地域差ではなく土壌水分の効果を拾えているかを確かめる想定。

## 評価方針(決定済み)

- **評価の仕組みは `src/evaluation.py`**(段階2で作成、EXP-003)。`StratifiedGroupKFold(n_splits=5, shuffle=True, groups=muniCd)` をシード 0〜19 で20回繰り返し、balanced accuracy と ROC-AUC を fold ごとに出す。前処理は Pipeline に入れて学習データだけで fit する。置換検定(200回)はラベルを市町村×年の中で並べ替える。ROCKET は 0.5 ではなく、同じ分割での置換検定の帰無分布の95%点と比べる
- **系列の揃え方(決定、D-005、2026-10-23)**: 主設定は**開花起点(開花日 = 0日目)の 0〜+59 日、60点**(82件・21地域)。開花日は実測を優先し、無い件は (ii)「開花 dap = 97.70 − 0.2889 × 播種日の通日」(実測66件で学習)で推定する。比較は dap 50–109(60点、81件)。感度分析は、実測開花日の件だけ(60件)と加古川を除く(76件)。開花前を含む窓と、開花後に観測が始まる10件は対象外。理由: GDD に揃えても開花はそろわず(EXP-004)、80件以上残る窓では開花起点と dap が60日で最長、開花起点は地域と集団の釣り合いが最もよく残る(EXP-004, 005)
- **入力データは `data/processed/stage4/`**(EXP-006)。`<名前>_X.npy` が (件数, 1, 60) の生の VWC、`<名前>_meta.csv` が同じ順のメタデータ。名前は main / dap / main_common / dap_common(両方を覆う77件)/ main_obs / main_no_kakogawa。作り方は `src/windows.py`(開花日の付与、窓の切り出し、変換 `to_raw` / `znorm` / `first_diff`)。収穫日より後の観測は使わない
- 非時系列の基準線は StandardScaler → LogisticRegression(L2, **C=1 固定**)。結果は `analysis/stage2/report.md`。どの特徴量も偶然水準程度で、Holm 補正後に有意なものは無い
- **CV は `muniCd` 単位の GroupKFold**。同じ地域の圃場は気象や土壌が似ていて、分けると精度が過大になる(データ漏洩)ため。ランダム分割は使わない。`label_muniyear` が付く地域は21で、1地域あたり2〜10件。fold 数は少なめになる
- 基本構成は `Rocket`(変換器) → `StandardScaler(with_mean=False)` → `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`。`RocketClassifier` の内部構成と同じ。**ただし sktime 1.2.0 の `Rocket` 変換器も、実際には不揃いな系列長を扱えない**(タグ `capability:unequal_length` は True なのに、nested_univ / pd-multiindex / df-list のどの形式でも ValueError になる。EXP-001)。長さが揃っていなくても動くのは、`MiniRocketMultivariateVariable` と、公式実装を1系列ずつ適用する方法だけ。公式の `external/rocket/code/rocket_functions.py` は NumPy 2 で削除された `np.NINF` を使っているので、**import した時点で失敗する**。代わりに `src/rocket_kernels.py` を使う。ここには `-np.inf` に直した公式実装の写しと、前処理(`znorm`, `pad_with_noise`, `resample`, `apply_kernels_variable`)がある。ROCKET は入力が z 正規化されている前提なので、系列は本来の長さで z 正規化してから長さを揃える。ArrowHead での比較(EXP-002)では、時間軸の位置を揃えたまま長さを合わせる方法が一番よかった。補間とノイズパディングは精度が下がった。VWC では、系列を揃えて固定長にする(D-005、下の「系列の揃え方」)
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

1. ~~系列の揃え方~~ → 決定(上の「評価方針」の D-005)。経緯は EXP-004(`analysis/stage3/report.md`)と EXP-005(`analysis/stage3b/report.md`)
2. 除外基準(カバー率が低い圃場、`field_id` 568/571)
3. 入力の形(生の VWC、圃場ごとの標準化、差分)。差分はセンサーごとの水準差を消せる(MultiRocket も1次差分を使う)
4. VWC=0 への落ち込みを欠測として扱うか
5. ラベルが94件と少ない問題への対処
6. 年の偏りと、相対ラベル(地域×年内の上下1/3)の扱い
7. `Questionaire.depth`(94件で 5〜28.5、欠損2)は**作付けの深さ**(ユーザー回答 2026-10-21)。センサーの深さではないので、VWC の水準差の説明には使えない。値が播種深度としては大きく、小数(11.97 など)は複数地点の平均らしいので、単位と測り方(播種深度か耕深か)は確認中。多収/低収で平均 16.3 / 15.5 とほぼ差がない
8. 灌水(`Questionaire.irrigation`)は生育時期の区分の文字列で、日付も回数も無い(EXP-006)。区分を窓の何日目に当てるか、灌水ありの件をどう扱うか(段階4では灌水の有無と区分のフラグだけを作った)

## 次のステップ

1. (済)系列の揃え方を決め、入力データを作った(EXP-006)
2. (済)段階5(EXP-007、`analysis/stage5/report.md`): sktime 1.2.0 の MiniRocket で評価した(aeon は venv_rocket の版が変わるので使っていない)。1次差分の入力だけが帰無分布の95%点を境界で上回った(main diff: bacc 0.590 / AUC 0.627、p=0.050)。部分集合では揺れる。Ridge の alpha は上限1000に張り付いている
3. 段階5のレポートにある追加実験の提案(diff での揃え方の比較、alpha の範囲、置換回数、カーネルの乱数、raw+diff の2チャンネル、label_year)から、ユーザーと次の実験を決める
4. (優先度低)`scripts/rocket_demo.py` を公式分割に直す。公式分割での値(0.806、論文値 0.8143)は EXP-002 の基準ですでに確認済み
