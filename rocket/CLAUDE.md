# rocket/ — 土壌水分量(VWC)時系列による多収・低収分類

## 研究の目的

土壌水分量(VWC)の時系列を入力として、大豆圃場の多収(収量上位)と低収(収量下位)を分類し、**その精度を上げていく**こと。手法は ROCKET を軸にする。今後の作業はこのディレクトリ内で行う。

進捗: データセット構築・品質確認・可視化・ROCKET デモ、段階1(データの棚卸し)、段階2(非時系列の基準線、EXP-003)、段階3(揃え方の起点と窓の集計、EXP-004, 005)、段階4(入力データ、EXP-006)、段階5(MiniRocket による初めての評価、EXP-007)、段階6(確認用の解析、EXP-008)、段階7(入力の形の比較 EXP-009、プーリングの比較 EXP-010)、段階8(Rocket の入力の大きさ EXP-011、alpha の張り付きの観察 EXP-012)、開花日予測(EXP-013〜016)、段階9(開花日のずれへの感度 EXP-017)、段階10(1次差分の情報の切り分け EXP-018、時間の順番の置換検定 EXP-019)、段階11(label_year での評価 EXP-020)、段階12(地域内/地域間の分解 EXP-021)、段階13(中央値分割のラベル3種での評価と分解 EXP-022)、段階14(label_muniyear_median の確認と交絡の基準線 EXP-023)、段階15(品種が集団の中で同じ集団だけでの評価 EXP-024)まで完了。**主ラベルは `label_muniyear_median`(D-006、2026-11-08)**。

## 実行環境とコマンド

専用の venv **`rocket\venv_rocket`**(Python 3.12.10)を使う。`amd` や anaconda は使わない。パッケージは `rocket/requirements.txt` に固定している(主なもの: sktime 1.2.0, scikit-learn 1.5.1, numpy 2.5, pandas 3.0, numba 0.68, matplotlib 3.11)。sktime と scikit-learn は、それまで使っていた anaconda と同じ版に合わせてある。パッケージを追加したら `pip freeze > rocket\requirements.txt` で更新する。

```powershell
# 環境の再作成
C:\Users\amilu\AppData\Local\Programs\Python\Python312\python -m venv rocket\venv_rocket
rocket\venv_rocket\Scripts\python -m pip install -r rocket\requirements.txt

# データセット再構築(この順で実行)
rocket\venv_rocket\Scripts\python rocket\scripts\build_soilmoisture_dataset.py
rocket\venv_rocket\Scripts\python rocket\scripts\add_yield_labels.py
rocket\venv_rocket\Scripts\python rocket\scripts\add_flowering_dates.py   # 開花日の列(要 flowering_pred_64.csv)

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
- `docs/data_issues.md` — 元データの問題点(日付の補完・異常値など)と対応の記録
- 元データは **`rocket/data/raw/FieldData_fieldid.db`**(2026-11-05 からこちらを使う。`Questionaire`, `SolidMoisture` テーブル)と、リポジトリルートの `outputs/reverse_geocode/field_addresses.csv`。読むだけで書き換えない(列の追加などが必要なときはユーザーに確認し、`.bak` を作ってから)。ルートの `data/processed/FieldData_fieldid.db` との違いは、`Questionaire.date_flag` 列があること。**既存のスクリプト(`build_soilmoisture_dataset.py`, `src/windows.py` など)はまだルートの DB を読んでいる**
- **開花日推定には `Questionaire.seed_date_raw` / `harvest_date_raw`(補完前の播種日・収穫日)を使う**。`seed_date` / `harvest_date` には近傍圃場からの補完値(播種日26件・収穫日93件)が混ざっている。`date_flag` が付いた圃場×年(A4・2016 / O7・O8・2016 / D1〜D4・2015)は日付に要確認の点がある。詳細は `docs/data_issues.md`

## データセット(`data/processed/`)

| ファイル | 1行の単位 | 行数 | 主な列 |
|---|---|---|---|
| `soilmoisture_timeseries.csv` | 圃場×年×観測日 | 30,040 | `field_id, year, date, dap, VWC` |
| `field_year_static.csv` | 圃場×年 | 304 | `field_id, year, muniCd, yield, seed_date, harvest_date, breed, irrigation, lat, lon, depth, between_lines, between_stocks, n_obs` |
| `field_year_labeled.csv` | 圃場×年 | 304 | 上記 + `label_muniyear`, `label_year`, `label_muniyear_median`, `label_year_median`, `label_muniyear_median_min2` + `flower_date_obs`, `flower_date_pred`, `flower_date`, `flower_source` |

- **開花日は `field_year_labeled.csv` の `flower_date` を使う**(2026-11-05 追加、`scripts/add_flowering_dates.py`)。実測(`Questionaire.flower_date`、240件)を優先し、実測の無い64件は EXP-016 の予測(`flowering_pred_64.csv`)で埋めてある。`flower_source` が `obs` / `pred`。`label_muniyear` 94件では実測66 / 予測28。`src/windows.py` と `stage4/` の入力(EXP-006〜011)はまだ古い式 (ii) で開花日を補っているので、切り替えは次の実験で行う

- 結合キーは `(field_id, year)`。`muniCd` は圃場ごとに1つなので `field_id` だけで結合する
- CSV は `utf-8-sig`。`muniCd` は `dtype=str` で読む
- `dap` = 観測日 − 播種日
- VWC があるのは **2016〜2018年のみ**。圃場×年は 2016=143 / 2017=115 / 2018=46
- 系列長は 32〜151 日(中央値 102)で**不揃い**。観測は毎日で重複はない。日の欠けがあるのは全304件中 `field_id=90`(2016年、11日)の1件だけで、`label_muniyear` 付き94件には欠けが無い
- 開花日は `Questionaire.flower_date` にあり(`field_year_static.csv` には入れていない)、`label_muniyear` 付き94件中66件に値がある。欠測28件のうち10件は加古川(28210)で、この地域は全件が欠測。播種から開花までは35〜65日(中央値45.5)

## ラベル(0=低収, 1=多収, 空欄=中間 or 対象外)

**主ラベルは `label_muniyear_median`(D-006、2026-11-08。それまでは `label_muniyear`)**。主の結果は段階13(EXP-022)の全123件で、M(MiniRocket・1次差分)の AUC 0.607(p=0.026)。中身は主に地域間(B 0.640)で、地域内(W 0.536)は確認できていない。品種の交絡の確認として、品種が集団の中で同じ部分集合(19集団・76件)で学習し直すと AUC 0.678(p=0.011)、品種 + 播種日の基準線 B3 は偶然水準(0.472)だった(EXP-024)。`label_muniyear` は感度分析として残す。下の段落は `label_muniyear` の付け方

グループ内 n 件のとき、収量の上位と下位から k = round(n/3) 件ずつ取る(中央値で分けるのではない。中間は空欄、同順位は field_id 昇順)。`label_muniyear` は**市町村×年の中での相対評価**で、各集団で多収と低収が同数になる(94件では29集団すべてで同数)。このため地域と年だけではラベルを当てられない。ただし窓で絞ると片方のクラスだけが落ちる集団が出て、釣り合いが崩れる(EXP-003)

| 列 | 分割単位 | 件数 | 備考 |
|---|---|---|---|
| `label_muniyear` | 同一 `muniCd`×年 | 94(47/47) | 感度分析(2026-11-08 まで主ラベル、D-006)。4圃場以上のグループのみ。94件はすべて同じラベルで `label_muniyear_median` に含まれる。地域×年グループ106個の大半は2圃場以下。2018年は6件のみ |
| `label_year` | 同一年 | 202(101/101) | 比較用。地域差を除けない |
| `label_muniyear_median` | 同一 `muniCd`×年 | 146(73/73) | 中央値分割(各集団で下位 floor(n/2) 件 = 0、上位 floor(n/2) 件 = 1。n が奇数なら中央の1件は空欄)。4圃場以上のグループのみ。**主ラベル**(D-006)。窓0〜+59日で123件(64/59)・21地域・28集団、地域内のペア147組(`data/processed/stage13/label_muniyear_median_*`)。2026-10-31 追加 |
| `label_muniyear_median_min2` | 同一 `muniCd`×年 | 282(141/141) | `label_muniyear_median` と同じ分け方を2圃場以上のグループに付けたもの(2圃場は1位と2位、3圃場は中央の1件が空欄)。4圃場以上の件は `label_muniyear_median` と同じ。59地域。2026-10-31 追加 |
| `label_year_median` | 同一年 | 302(151/151) | 中央値分割(同上)。2016・2017 は件数が奇数なので中央の1件ずつが空欄 |

両方のラベルが付いた圃場での一致率は約0.90。両ラベルで同じモデルを学習して精度を比べ、地域差ではなく土壌水分の効果を拾えているかを確かめる想定。

## 評価方針(決定済み)

- **評価の仕組みは `src/evaluation.py`**(段階2で作成、EXP-003)。`StratifiedGroupKFold(n_splits=5, shuffle=True, groups=muniCd)` をシード 0〜19 で20回繰り返し、balanced accuracy と ROC-AUC を fold ごとに出す。前処理は Pipeline に入れて学習データだけで fit する。置換検定(200回)はラベルを市町村×年の中で並べ替える。ROCKET は 0.5 ではなく、同じ分割での置換検定の帰無分布の95%点と比べる
- **系列の揃え方(決定、D-005、2026-10-23)**: 主設定は**開花起点(開花日 = 0日目)の 0〜+59 日、60点**(82件・21地域)。開花日は実測を優先し、無い件は (ii)「開花 dap = 97.70 − 0.2889 × 播種日の通日」(実測66件で学習)で推定する。比較は dap 50–109(60点、81件)。感度分析は、実測開花日の件だけ(60件)と加古川を除く(76件)。開花前を含む窓と、開花後に観測が始まる10件は対象外。理由: GDD に揃えても開花はそろわず(EXP-004)、80件以上残る窓では開花起点と dap が60日で最長、開花起点は地域と集団の釣り合いが最もよく残る(EXP-004, 005)
- **入力データは `data/processed/stage4/`**(EXP-006)。`<名前>_X.npy` が (件数, 1, 60) の生の VWC、`<名前>_meta.csv` が同じ順のメタデータ。名前は main / dap / main_common / dap_common(両方を覆う77件)/ main_obs / main_no_kakogawa。作り方は `src/windows.py`(開花日の付与、窓の切り出し、変換 `to_raw` / `znorm` / `first_diff`)。収穫日より後の観測は使わない
- 非時系列の基準線は StandardScaler → LogisticRegression(L2, **C=1 固定**)。結果は `analysis/stage2/report.md`。どの特徴量も偶然水準程度で、Holm 補正後に有意なものは無い
- **CV は `muniCd` 単位の GroupKFold**。同じ地域の圃場は気象や土壌が似ていて、分けると精度が過大になる(データ漏洩)ため。ランダム分割は使わない。`label_muniyear` が付く地域は21で、1地域あたり2〜10件。fold 数は少なめになる
- **段階6以降の分類器は `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10) × n_train)`**(n_train は fold の学習件数)。StandardScaler(with_mean=False) と組み合わせると、MiniRocket 原論文の `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)` と等価(EXP-008 の事前確認)。変換は sktime 1.2.0 の MiniRocket(aeon は venv_rocket の版が変わるので使わない)。sktime の `Rocket` は `normalise=True` が既定なので、使うときは明示する
- 基本構成は `Rocket`(変換器) → `StandardScaler(with_mean=False)` → `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`。`RocketClassifier` の内部構成と同じ。**ただし sktime 1.2.0 の `Rocket` 変換器も、実際には不揃いな系列長を扱えない**(タグ `capability:unequal_length` は True なのに、nested_univ / pd-multiindex / df-list のどの形式でも ValueError になる。EXP-001)。長さが揃っていなくても動くのは、`MiniRocketMultivariateVariable` と、公式実装を1系列ずつ適用する方法だけ。公式の `external/rocket/code/rocket_functions.py` は NumPy 2 で削除された `np.NINF` を使っているので、**import した時点で失敗する**。代わりに `src/rocket_kernels.py` を使う。ここには `-np.inf` に直した公式実装の写しと、前処理(`znorm`, `pad_with_noise`, `resample`, `apply_kernels_variable`)がある。ROCKET は入力が z 正規化されている前提なので、系列は本来の長さで z 正規化してから長さを揃える。ArrowHead での比較(EXP-002)では、時間軸の位置を揃えたまま長さを合わせる方法が一番よかった。補間とノイズパディングは精度が下がった。VWC では、系列を揃えて固定長にする(D-005、下の「系列の揃え方」)
- 手法は当面 ROCKET(`rocket_transform="rocket"`)のみ。sktime 1.2.0 には `MiniRocketClassifier` / `MultiRocketClassifier` が無い(使う場合は `from sktime.transformations.rocket import Rocket, MiniRocket, MultiRocket` の変換器か。`sktime.transformations.panel` は非推奨 `RocketClassifier(rocket_transform=...)`)
- ROCKET の特徴量は各カーネルの「最大値」と「PPV(正の値の割合)」で、2 × num_kernels 次元(10000本なら20000次元)。94件に対して非常に高次元なので過学習に注意する

## データの注意点(`outputs/quality_check.csv` に圃場×年ごとのフラグあり)

- 除外済み: 観測数30未満の2件(`field_id=5` 2016年, `field_id=580` 2018年)。基準は `build_soilmoisture_dataset.py` の `MIN_OBS = 30`
- 除外していないが要注意:
  - 栽培期間に対する観測期間のカバー率が50%未満: 28件
  - 観測開始が播種後30日超: 約3割
  - 栽培期間の記録が異常(収穫日が翌年1/5): `field_id` 568, 571
  - VWC の系列が127日すべて同じ: `field_id` 472, 473(2018年)。`docs/data_issues.md` の3
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
3. (済)段階6(EXP-008、`analysis/stage6/report.md`): 確認用の解析(main・diff・MiniRocket、乱数10通り、置換1000回)で、AUC 0.605 が帰無分布の95%点 0.595 を上回った(p=0.035、基準線 0.483)。bacc 0.544 は95%点 0.562 未満。1次差分でも揃え方の差は検出できなかった(AUC +0.055、繰り返しの85%が正)。alpha は原論文の尺度でも上限に張り付く
4. (済)段階7(EXP-009、`analysis/stage7/report.md`): 同じ条件で、1次差分(AUC 0.605)は生の値(0.492)・z 正規化(0.463)より明らかに高い(差を検出)。sktime の MiniRocket はゼロパディングのため平行移動に不変でない。プーリングの比較(EXP-010): sktime Rocket(normalise=False)で max を加えても差は検出できず、Rocket はどの設定も偶然水準
5. (済)段階8(EXP-011、`analysis/stage8/report.md`): Rocket に前提どおりの大きさの入力(系列ごとの z 正規化 / 全体の標準偏差で割る)を入れても偶然水準で、normalise=False(段階7)より下がった。主な設定(MiniRocket・1次差分)は変えない。alpha の観察(EXP-012、`analysis/stage8/alpha_diag.py`): 候補を広げても GCV はほぼ単調に下がり、切片だけのモデルに近づくので上限に張り付く。テスト AUC は alpha が小さいと 0.630 だが、置換の95%点も上がる
6. (済)開花日予測(EXP-013〜016、`scripts/fit_flowering_*.py`, `eval_flowering_raw_seed.py`, `predict_flowering_targets.py`): GDD・光熱モデル・RF はどれも播種日の通日による回帰に勝てなかった(leave-one-region-out の MAE 4.43 日)。補完前の播種日で評価し直し、実測の無い64件を播種日回帰で予測した(`flowering_pred_64.csv` → `field_year_labeled.csv` の `flower_date`)
7. (済)段階9(EXP-017、`analysis/stage9/report.md`): 実測開花日の60件で窓の起点を予測値に変えると AUC −0.035(0.620 → 0.585、95% 基準では差を検出せず)。±3 日のずれは ±0.015 以内。60件では基準も置換の95%点未満で、予測開花日で精度が保てるとは言えない
8. (済)段階10(EXP-018・019、`analysis/stage10/report.md`): 1次差分の要約統計だけ(S1)は偶然水準。日の順番をシャッフルした分布(200通り)の中で D(0.605)は上位約20%(p ≈ 0.20〜0.23)で、**時間の並び方の効果は確認できなかった**。シャッフルしても AUC 平均 0.56〜0.57 で、値の分布の情報を拾っている可能性がある(探索的)
9. (済)段階11(EXP-020、`analysis/stage11/report.md`): label_year(162件、48地域、開花日は CSV の `flower_date`)で M(MiniRocket・1次差分)は AUC 0.635(p=0.005)・bacc 0.593 で当たっている。実測開花日のみ(132件)でも 0.628。場所と時期だけの基準線 B2 は 0.403(偶然水準未満)で M との差は大きいが割り引いて読む。VWC の要約統計 B1(0.558)との AUC の差は検出できず。地域差を含むので土壌水分の効果とは言えない。alpha は上限に張り付かなかった
10. (済)段階12(EXP-021、`analysis/stage12/report.md`): 段階11のテスト件のペアを、地域内(同じ市町村×年)・地域間(同じ年・違う市町村)・年をまたぐ、に分けた。M は地域間 0.646(p=0.006)で当たっているが、地域内は 0.526(p=0.40、95%点 0.649)で当たっているとは言えない。地域内のペアは17集団・41組だけで検出力が低い(80%で検出するには約0.72が必要)。段階11の AUC は主に地域間の差。B2 は地域内・地域間・年をまたぐのどれも0.5未満
11. (済)段階13(EXP-022、`analysis/stage13/report.md`): 中央値分割のラベル3種で評価と地域内/地域間の分解を行った。主の `label_muniyear_median_min2`(228件)で M の地域内は 0.508(p=0.44、95%点 0.579、80%で検出には約0.62が必要)で、**地域内の多収・低収を当てられるとは言えない**(全体も 0.510)。`label_muniyear_median` は全体 0.607(p=0.026)で当たるが中身は地域間(0.640)で、地域内は 0.536。label_year で B2 の地域内が0.5を下回るのは、地域間で学んだ播種日の向きが地域内で逆に当たるためと見られる
12. (済)段階14(EXP-023、`analysis/stage14/report.md`): label_muniyear_median(123件)の確認。地域間と地域内の差 B − W は 0.101(市町村単位のブートストラップの95%区間 [0.027, 0.175])で差あり。多収と低収が同数の集団どうしの B は 0.621(p=0.006)で、集団のクラスの偏りではない。品種 + 播種日だけの基準線 B3 が AUC 0.575(p=0.047)で、M との差(+0.032)を検出できず、品種との交絡を否定できなかった。主ラベルを `label_muniyear_median` に変えた(D-006)
13. (済)段階15(EXP-024、`analysis/stage15/report.md`): **品種の交絡の確認**。品種が集団の中で1つだけ(欠測なし)の19集団・76件で学習し直すと、M は AUC 0.678(p=0.011、95%点 0.631)で当たり、B3 は 0.472 まで下がった(M − B3 +0.206、M − B1 +0.128 で差あり)。M の結果は品種では説明できない。ただし全件モデルを部分集合で集計しても 0.665 で、0.607 から上がったのは除いた集団で当たりにくかったためと推測。地域内 W は 0.632(p=0.072)で届かない
14. 段階6〜15のレポートにある追加実験の提案(除いた9集団で当たりにくい理由、median の地域間が地域内より高い理由、同じ市町村×年の平均推移を引いた系列、値の分布の情報、しきい値、MultiRocket)から、ユーザーと次の実験を決める
15. (優先度低)`scripts/rocket_demo.py` を公式分割に直す。公式分割での値(0.806、論文値 0.8143)は EXP-002 の基準ですでに確認済み
