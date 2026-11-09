# データの問題点と対応

元データ `rocket/data/raw/FieldData_fieldid.db` について、確認した問題とその対応を記録する。

## 1. 播種日・収穫日の補完(2026-11-05 確認・対応)

### 経緯

リポジトリルートの `data/processed/FieldData_fieldid.db` は、`data/raw/FieldData.db` に field_id を付けたうえで、次の修正を加えたもの。

| 修正 | スクリプト | 件数 |
|---|---|---|
| 播種日の欠損を近傍圃場から補完 | `impute_growing_dates.py --write-db` | 26 |
| 収穫日の欠損を近傍圃場から補完 | `impute_growing_dates.py --write-db`、`fix_field5_harvest.py` | 92 + 1 |
| A4・2016 の収穫日(2016-07-25)を開花日の誤入力として `flower_date` に移す | `fix_field5_harvest.py` | 1 |
| 株間 `between_stocks` の文字化けを 12.5 に修正 | 不明 | 44 |

補完は、同じ年の100 km 以内の近傍3圃場の通日の中央値。開花日推定で補完値を使うと、推定の入力に推定値が混ざる。

### 対応

- ルートの `add_raw_growing_dates.py` で、`Questionaire` に **`seed_date_raw` / `harvest_date_raw`** を追加した。中身は元の `FieldData.db` の値(補完前)。既存の `seed_date` / `harvest_date`(補完済み)は残している。作業前のバックアップは `data/processed/FieldData_fieldid.db.bak_before_rawdates`
- A4・2016 は開花日への移し替えを残し、`harvest_date_raw` を NULL にした
- 確認: A4・2016 以外は2列とも元の `FieldData.db` と一致。値がある行は既存列とも一致
- このDBを `rocket/data/raw/FieldData_fieldid.db` にコピーし、以後こちらを使う

| 列 | NULL | うち補完済み列には値がある |
|---|---|---|
| `seed_date_raw` | 117 | 26 |
| `harvest_date_raw` | 183 | 93 |

**開花日推定には `seed_date_raw` / `harvest_date_raw` を使う。**

参考: `data/processed/FieldData_fieldid_raw.db` は元の `FieldData.db` に field_id を付けただけのもの(ルートの `build_fielddata_fieldid_raw.py`)。A4・2016 も元のまま(収穫日が 7/25)なので注意。

## 2. 元データの播種日・開花日・収穫日の確認(2026-11-05)

元の `FieldData.db`(702件)を確認した。

- 書式の誤りなし。日付の順序の逆転(播種 → 開花 → 収穫)なし
- 欠損: 播種日117、開花日238、収穫日182。3つそろうのは413件、全部欠けるのは110件
- 間隔: 播種→開花 平均46日(26〜75)、開花→収穫 平均93日(59〜127)、播種→収穫 平均139日(60〜193)

外れ値(IQR の1.5倍)と、年の不一致を調べた結果、要確認の点が次のとおり。

| 圃場・年 | field_id | 内容 | 対応 | `date_flag` |
|---|---|---|---|---|
| A4・2016 | 5 | 収穫日が 2016-07-25(播種から60日)。近隣の A 圃場の開花日(7/15〜7/22)と合う | 開花日の誤入力として `flower_date` に移した。`harvest_date_raw` は NULL | `harvest_was_flower` |
| O7・O8・2016 | 568, 571 | 収穫日が翌年の 2017-01-05。播種(7/25・7/28)→収穫が164日で、同じ年の他の O 圃場(137〜142日程度)より長い | 値は変えない(遅播きなので実際の値の可能性もある) | `harvest_next_year` |
| D1・D2・2015 | 80, 86 | 播種→収穫193日(全体で最長)、開花→収穫121日。同じ品種・同じ播種日の D3・D4 は 11/05 収穫なのに、こちらは 12/09 | 値は変えない | `long_season` |
| D1〜D4・2015 | 80, 86, 89, 92 | 播種→開花72〜75日(外れ値)。4圃場とも同じ品種・同じ播種日 | 値は変えない(品種の特性の可能性)。開花日推定の学習データに入れるかは要判断 | `long_seed_to_flower` |

### date_flag 列

`rocket/scripts/add_date_flags.py` で `rocket/data/raw/FieldData_fieldid.db` の `Questionaire` に `date_flag`(TEXT)を追加した。値は変えていない。複数該当するときは `;` でつなぐ(D1・D2 は `long_season;long_seed_to_flower`)。該当しない行は NULL。作業前のバックアップは `rocket/data/raw/FieldData_fieldid.db.bak_before_date_flag`。

ルートの `data/processed/FieldData_fieldid.db` には `date_flag` を付けていない。

### 未解決

- O7・O8・2016 と D1・D2・2015 の収穫日が正しいか(元の調査票で確かめられるか)
- D1〜D4・2015 を開花日推定の学習データに入れるか

## 3. 圃場472・473(2018年)の VWC が同じ(2026-11-06 確認、段階13)

| 圃場・年 | field_id | 内容 | 対応 |
|---|---|---|---|
| M02・M03・2018 | 472, 473 | `SolidMoisture` の VWC が127日(2018-07-23〜11-26)すべて同じ値。`soilmoisture_timeseries.csv` でも同じ | 値は変えない。除外もしていない |

- 2件は市町村 25210、緯度・経度・品種・播種日・開花日・収穫日・畝間・株間も同じ。収量は 272.0 / 222.7 kg/10a で違う。`depth` は 17.5 / NULL
- 同じセンサーのデータを2圃場に入れた可能性がある(推測)。どちらが正しいか、両方とも同じ圃場の値かは分からない
- 全圃場×年で、VWC の系列が完全に同じ組はこの1組だけ
- 影響(段階13、EXP-022):
  - 2件とも入るラベルは `label_year_median` だけ(どちらも低収)。観測ラベルではペアにならないので、評価値には影響しない
  - 置換で多収と低収に分かれるとスコアが同点になり、高速版の Ridge が sklearn で学習し直すので計算時間が延びる(結果は同じ)
  - 同じ系列に違うラベルが付くと、地域内(W)のペアの AUC は0.5になる

### 未解決

- 元の調査票・センサーの記録で、どちらの圃場の値かを確かめられるか
- 確かめられない場合、片方(または両方)を除外するか
