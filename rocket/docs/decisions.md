# 決定記録

研究の方針として決めたことを、理由と一緒に残す。結論だけを `rocket/CLAUDE.md` に反映する。
方針を変えたときは古い項目を消さず、状態を「廃止」にして、新しい項目の番号を書く。

## 一覧

| ID | 日付 | 決定 | 状態 |
|---|---|---|---|
| D-001 | 2026-10 | 結合キーは `(field_id, year)` | 有効 |
| D-002 | 2026-10 | 観測数30未満の圃場×年を除外 | 有効 |
| D-003 | 2026-10 | 主ラベルは地域×年内の上下1/3 (`label_muniyear`) | 有効 |
| D-004 | 2026-10 | CV は `muniCd` 単位の GroupKFold | 有効 |
| D-005 | 2026-10 | 手法は当面 ROCKET のみ。`Rocket` 変換器 + Ridge で組む | **要再検討**(不揃いな長さに非対応と判明) |

---

<!-- ここから下にコピーして使う
## D-XXX: <決定を一文で>

- 日付: YYYY-MM-DD
- 状態: 有効 / 廃止(→ D-YYY)
- 決定: <何をどうするか>
- 理由: <根拠になったデータ・図・実験(EXP-XXX)・論文>
- 検討した他の案: <案と、採らなかった理由>
- 影響: <変更が必要なスクリプトやデータ>
-->

## D-001: 結合キーは `(field_id, year)`

- 日付: 2026-10
- 状態: 有効
- 決定: `Questionaire` と `SolidMoisture` を `(field_id, year)` で結合する。`muniCd` は `field_id` だけで結合する
- 理由: 収量と土壌水分は同じ圃場でも年ごとに値が違う。`muniCd` は圃場ごとに1つしかない
- 影響: `field_id` が NULL の行は結合できないので除外する(`Questionaire` は702行中611行が使える)

## D-002: 観測数30未満の圃場×年を除外

- 日付: 2026-10
- 状態: 有効
- 決定: `build_soilmoisture_dataset.py` の `MIN_OBS = 30`。除外したのは `field_id=5`(2016年、3日)と `field_id=580`(2018年、12件)
- 理由: 系列が短すぎて、特徴を取り出せない
- 検討した他の案: カバー率50%未満の28件や、栽培期間が異常な `field_id` 568・571 の除外 → **未決定**

## D-003: 主ラベルは地域×年内の上下1/3 (`label_muniyear`)

- 日付: 2026-10
- 状態: 有効
- 決定: 同じ `muniCd`×年の中で収量の上位・下位を k = round(n/3) 件ずつ取り、1(多収)・0(低収)とする。4圃場以上のグループだけに付ける(94件)。同じ年の中だけで付ける `label_year`(202件)は比較用
- 理由: 地域による収量差を除き、土壌水分の効果だけを見るため。地域×年グループ106個の大半は2圃場以下なので、最小サイズを4にした
- 検討した他の案: 最小サイズを3以下にする → グループ内の順位の意味が薄れる
- 影響: 2つのラベルで同じモデルの精度を比べ、地域差を拾っていないか確かめる

## D-004: CV は `muniCd` 単位の GroupKFold

- 日付: 2026-10
- 状態: 有効
- 決定: `GroupKFold(groups=muniCd)` を使う。ランダム分割は使わない
- 理由: 同じ地域の圃場は気象や土壌が似ている。学習と検証に分けると情報が漏れ、精度が過大になる
- 影響: `label_muniyear` が付く地域は21で、1地域あたり2〜10件。fold 数は少なくなる

## D-005: 手法は当面 ROCKET のみ。`Rocket` 変換器 + Ridge で組む

- 日付: 2026-10
- 状態: 有効
- 決定: `Rocket` → `StandardScaler(with_mean=False)` → `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`
- 理由: 系列長が32〜151日と揃っていない。sktime 1.2.0 の `RocketClassifier` は不揃いな長さに対応しないが、`Rocket` 変換器は対応する。この構成は `RocketClassifier` のデフォルトの内部構成と同じ
- 検討した他の案: MiniRocket / MultiRocket → 当面は ROCKET 単体で基準を作る
- **問題(2026-10-20 に判明)**: sktime 1.2.0 の `Rocket` は `capability:unequal_length=True` のタグを持つが、長さ32・100・151の系列を `pd-multiindex` で渡すと `ValueError: cannot reshape array of size 283 into shape (3,151,1)` になる。venv_rocket(pandas 3.0)と anaconda(pandas 2.2)の両方で同じ結果。長さの不揃いを扱えたのは `MiniRocketMultivariateVariable` だけだった。「理由」に書いた前提が成り立たないので、次の案から選び直す
  - A. 系列を揃えて固定長にする(共通の dap 区間で切り出す、補間・パディングなど)。未決定事項1「系列の揃え方」と一緒に決められる
  - B. 公式実装 `external/rocket/code/rocket_functions.py` のカーネルを、系列ごとに長さのまま適用する(ROCKET の論文上は長さが違っても扱える)
  - C. `MiniRocketMultivariateVariable` を使う(ROCKET のみという方針から外れる)
