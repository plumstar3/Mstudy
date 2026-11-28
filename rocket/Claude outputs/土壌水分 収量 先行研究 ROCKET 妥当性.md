# 現地センサー土壌水分で大豆収量を分ける研究の空白

**結論の要約(BLUF)**: 今回の調査範囲では、**多数の農家圃場それぞれに置いた現地センサー(in-situ)の連続土壌水分時系列を、そのまま入力にして大豆(または他作物)の収量を分類・回帰した査読論文は見つからなかった**。近い研究は4種類に分かれる。土壌水分を推定値で扱うもの(熊谷ら 2024)、収量を圃場×年の中で正規化しつつ土壌水分は地形指数で代用するもの(Maestrini & Basso 2018)、排水性と転換畑大豆の収量を単発の測定で結ぶもの(北海道中央農試 2005)、センサー時系列はあるが収量を扱わないもの(Yang et al. 2024、Figueroa & Pope 2017)である。降雨後の乾き方(drydown)を収量の特徴量に使った研究も、衛星研究を含めて確認できなかった。このため、「多数圃場の in-situ VWC 時系列 × 地域×年の中の相対収量 × 交絡の統制 × 降雨後の減少への分解」という組み合わせは新規性の候補になる。ただし検索は Web 検索が中心で網羅的ではなく、J-STAGE・ASABE の本文も読めていないので、「世界初」「前例がない」とは書けない。RQ2 については、**ROCKET 系の原論文は「小標本で過学習しにくい」とは言っていない**。原論文が言っているのは、特徴量の数が件数を大きく上回る(p ≫ n)設定でも、L2 正則化した ridge を交差検証で調整すれば、学習数十〜数千件の UCR データで最先端並みの精度が出るということである。また 2024 年の大規模比較では、**学習 1〜99 件のデータ群で1位は HIVE-COTE 2(HC2)で、ROCKET 系の最上位(MultiRocket-Hydra)は2位**だった。修論では「過学習しにくい」を「学習するのは線形層の重みだけで、正則化の強さを交差検証で選べる。小〜中規模の UCR データで実績がある」と書き直し、catch22・1NN-DTW などの基準線を加えると、主張が原論文と合う。

---

## RQ1: 多数圃場の in-situ 土壌水分時系列から収量を分類した先行例は見つからなかった

### 調査の範囲と、「見つからない」の意味

調査は Web 検索・取得で、トピックごとに 10〜18 回程度行った。J-STAGE・ASABE eLibrary・ResearchGate は本文を取得できず、Google Scholar・Web of Science での網羅的な検索もしていない。下の表で「タイトルのみ確認」とした文献は、中身を確かめていない。**「見つからなかった」は「存在しない」の証明にはならない**。修論に書く前に、Google Scholar で `"soil moisture sensor" AND yield AND (soybean OR "on-farm")`、CiNii・J-STAGE で「土壌水分センサー 大豆 収量」「転換畑 ダイズ 地下水位 収量」などを検索して補うことを勧める。特に、ユーザーのデータ源(多数の圃場にセンサーを入れた事業)そのものの報告があれば、それが最も近い先行研究になりうる。農研機構のスマート農業実証の資料が検索で出たが、中身は確認していない([NARO SA1-204J2](https://www.naro.go.jp/brain/smart-nogyo/results/files/SA1-204J2.pdf))。

### 直接関連する研究の一覧(土壌水分 × 収量、または in-situ 土壌水分時系列)

見つかった研究は、(a) 土壌水分を推定値や1回きりの測定で扱う、(b) センサー時系列はあるが収量と結び付けない、(c) 収量マップと代理変数(地形指数など)で水分の効果を推測する、のいずれかだった。

| 研究 | 対象作物 | データ種類と時間解像度 | 件数 | 手法 | 目的変数 | 交絡統制 | 解釈の有無 |
|---|---|---|---|---|---|---|---|
| **本研究(比較の基準)** | 大豆 | **in-situ センサー VWC**、1時間 → 日平均、開花起点 0〜59 日 | 約300圃場×年、主ラベル約120件 | MiniRocket → Ridge | 市町村×年の中の中央値より上/下 | 市町村単位 GroupKFold、市町村×年内の置換検定、品種が集団内で同じ部分集合、品種+播種日の基準線 | あり(入力の成分への分解: 降雨直後の減少など) |
| 熊谷・湊・高橋 2024 日作紀 | 丹波黒大豆 | **推定**土壌水分(大豆灌水支援システム)、旬別。実測は検証用の2地点・2年のみ | 定点4地点 × 16年 | LASSO 回帰 | 精子実重(年次の変動) | 単一品種・単一産地で固定 | あり(10月上旬の土壌水分が最大の正の効果) |
| Maestrini & Basso 2018 Sci Rep | トウモロコシ・大豆・綿・小麦 | **土壌水分は未測定**。TWI(DEM)、Daymet 日降水量、SSURGO | 338圃場・収量マップ1,625枚(30 m セル) | 線形混合モデル | 圃場×年の中で正規化した収量の安定/不安定ゾーン | 圃場×年内の正規化、年を1つずつ除く確認 | あり(不安定ゾーンは TWI が高く、グライ化と関連) |
| 北海道中央農試 2005(成績概要書) | 転換畑大豆 | 土壌水分の時系列なし。現場透水試験(Ib 値)など単発 | 空知の26圃場 | しきい値による整理 | 収量 | なし(記述) | あり(Ib < 100 mm/h で収量 300 kg/10a 未満) |
| 石塚ら 2024 福岡農林試研報 | 大豆 | pF による土壌水分(測定法の詳細は未確認) | 処理比較の試験 | 処理区の比較 | 収量・収量構成要素 | 処理の設計による | あり(排水で過湿軽減、乾燥期の灌漑で増収) |
| Wraith et al. 2005 Comput Electron Agric | 小麦(収量あり)、ペパーミント(収量なし) | 車載 TDR の**1回きり**の格子調査。別圃場で固定 TDR(6時間ごと約100日) | 2圃場(n=70・75) | 空間解析 | 収量(相関係数は報告なし) | なし | 考察のみ |
| Yang et al. 2024 Vadose Zone J | 農家圃場(米国中西部) | in-situ 長期ステーション + 2回の dry-down の密なサンプリング | 3圃場 | 空間構造の記述 | **収量なし**(土壌水分の空間構造) | — | あり(時間で変わらない空間構造) |
| Figueroa & Pope 2017 Sensors | 果樹(アボカド・キウイ・ネクタリン) | in-situ FDR、**15分ごと** | 5圃場 | SAX による時系列パターン検出 | **収量なし**(根の吸水パターン) | — | あり(パターンそのもの) |
| Kinoshita et al. 2022 Agron J(誌名は推定) | トウモロコシ | 近接センシング(ECa・光学・pH)、土壌水分は重量法サンプル | 26圃場 | random forest | PCA で標準化した収量パターン | 収量の標準化、地域ごとのモデル、独立圃場で検証 | あり |

出典: 熊谷ら([農研機構プレスリリース](https://www.naro.go.jp/publicity_report/press/laboratory/niaes/168580.html))、Maestrini & Basso([Springer](https://link.springer.com/10.1038/s41598-018-32779-3))、北海道中央農試([道総研](https://www.hro.or.jp/agricultural/center/result/kenkyuseika/gaiyosho/h17gaiyo/f3/2005301.htm))、石塚ら([J-STAGE](https://www.jstage.jst.go.jp/article/fafrc/10/0/10_1/_article/-char/ja))、Wraith et al.([USDA ARS PDF](https://ars.usda.gov/arsuserfiles/20360500/pdf_pubs/P1908.pdf))、Yang et al.([DOAJ](https://doaj.org/article/255d81b6e7a545fb9f9c2190dd1cc080))、Figueroa & Pope([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5492835/))、Kinoshita et al.([USQ リポジトリ PDF](https://research.usq.edu.au/download/5cfb3cc71cea50bd8eb2a94982ec7452d93c595af25c411716d05f0a11dad3de/2493684/2022_Kinoshita.pdf))。

熊谷ら (2024) は、大豆・時期別の土壌水分・機械学習・どの時期が効くかの解釈、の4点で最も近い。LASSO で選ばれた変数のうち **10月上旬(子実肥大前半)の土壌水分が最大の正の効果**を持ち、R² は 0.8 だった。ただし土壌水分は推定値で、問いは4地点の「年次変動」であり、ユーザーの「同じ市町村×年の中の圃場差」とは違う([農研機構](https://www.naro.go.jp/publicity_report/press/laboratory/niaes/168580.html))。DOI は取得資料に記載がなく未確認。Maestrini & Basso (2018) は、**収量を圃場×年の中で正規化して相対評価する**という、ユーザーの相対ラベルと同じ発想の前例として引用できる。天水圃場では、不安定ゾーンは TWI が高く(12.9 対 12.4/12.7、p<0.05)、不安定さはグライ化(周期的な湛水の痕跡)の確率と正の関連があった。ただし土壌水分は測っていない。本文には降水との相関の符号の記述に不整合がある、とノートは指摘している([Springer](https://link.springer.com/10.1038/s41598-018-32779-3))。

**タイトルのみ確認で、中身は未確認**の文献もある。Green & Erskine (2004) *Hydrological Processes*, DOI 10.1002/hyp.1422 は空間的な収量と土壌水分を扱う([zendy](https://zendy.io/title/10.1002/hyp.1422))。Paz ほか (1998) *Transactions of the ASAE* 41(5) は大豆の水ストレスと収量の空間変動を扱い、作物モデル研究と記憶されているが未確認である([ASABE](https://elibrary.asabe.org/abstract.asp?JID=3&AID=17284&CID=t1998&v=41&i=5&T=1))。"Relationship between soybean yield from high and low yielding field sites and selected soil characteristics" は多収/低収の圃場を比べる点でユーザーのラベルに近い可能性があるが、土壌水分を測ったかは不明である([AGRIS](https://agris.fao.org/search/en/records/65e01b2e4c5aef494fe853d5))。大豆栽培での LSTM による土壌水分の**予測**(収量予測ではない)の研究もある(*Sensors* 2023, [DOI 10.3390/s23041976](https://doi.org/10.3390/s23041976)、タイトルのみ確認)。

### 交絡(地域・年・品種)を統制した研究: 前例は相対収量の考え方だけ

交絡を明示的に統制していると確認できたのは、圃場×年の中で収量を正規化し、混合モデルと年を1つずつ除く確認を行った Maestrini & Basso (2018)([Springer](https://link.springer.com/10.1038/s41598-018-32779-3))と、PCA で収量を標準化し、地域(MLRA)ごとにモデルを作り独立圃場で検証した Kinoshita et al. (2022)([USQ PDF](https://research.usq.edu.au/download/5cfb3cc71cea50bd8eb2a94982ec7452d93c595af25c411716d05f0a11dad3de/2493684/2022_Kinoshita.pdf))だけだった。熊谷ら (2024) は品種と産地を固定した設計で、地域間の交絡は設計上ないが、地点は4つしかない([農研機構](https://www.naro.go.jp/publicity_report/press/laboratory/niaes/168580.html))。**品種の交絡をデータで直接確かめた例、センサー実測の土壌水分と収量を混合モデルや grouped CV で結んだ例は見つからなかった**。ユーザーの設計(市町村×年の中の中央値分割、市町村単位の GroupKFold、市町村×年内の置換検定、品種が集団内で同じ部分集合での再評価)は、確認できた範囲の先行研究より交絡の扱いが厳しい。

ただし、設計が厳しいことと、土壌水分の効果を「切り分けた」ことは別である。ユーザー自身の段階13〜14では、主ラベルの全体 AUC 0.607 の中身は主に地域間のペア(B 0.640)で、地域内(W 0.536)は確認できておらず、品種+播種日の基準線 B3(0.575)との差も検出できなかった。品種が集団の中で同じ部分集合(76件)で AUC 0.678 になり B3 が偶然水準に下がったことは**品種では説明できない**という根拠になるが、地域内の判別は件数不足で検出できていない(D-007)。修論では「交絡を統制した評価の下で、全体として偶然以上に見分けられる」と書き、「同じ地域の中で土壌水分の効果を切り分けた」とは書かないほうがよい。

### どの時期・特徴が効くか: 時期の感受性は確立、drydown と収量の関係は未開拓

**大豆の生育ステージ別の水ストレス感受性**は一次文献で定量的に確立している。干ばつでは、Eck, Mathers & Musick (1987) *Field Crops Research* 17(1):1–16 が、R1/R2〜R3 のストレスで 9〜13% 減、R1/R2〜R4.5 で 46% 減、**R5 から成熟まで(5週間)で 45%(ストレスの強い年は 88%)減**と報告した([AGRIS](https://agris.fao.org/search/en/records/65dee5040f3e94b9e5d27576))。短い期間より、莢形成期をまたぐ長い期間や子実肥大期(seed fill)のストレスが効く。過湿(waterlogging)では、54 論文をまとめた Ploschuk, Miralles & Striker (2022) *Plant and Soil* 475:237–252 によると、**R1〜R3 の過湿で平均 57% 減**、最大 80% 減で、主因は1株あたり莢数の減少だった([Springer](https://link.springer.com/doi/10.1007/s11104-022-05364-x))。Adegoye et al. (2023) *Agriculture* 13(4):844 では、R1 で16日間の過湿により 50% 超の減収、莢数 −79% だった([MDPI](https://mdpi.com/2077-0472/13/4/844/html))。ユーザーの窓(開花 0〜59 日、おおよそ R1〜R5/R6)は、過湿に敏感な R1〜R3 と干ばつに敏感な R3〜R6 の両方を含む。ただし、開花後の日数と R ステージの対応は文献で確認しておらず、品種と播種日で変わる。

**降雨後の乾き方(drydown)**は水文学で研究が多いが、衛星(SMAP)を使った地域規模のものが中心である。McColl et al. (2017) *GRL*(DOI 10.1002/2017GL072819、本文未取得)は、乾燥度が高く砂が多い地域ほど時定数が短いが、土性の効果は比較的小さく、説明できない分散が大きいと報告した(Shellito et al. 2018 による要約、[HESS](https://hess.copernicus.org/articles/22/1649/2018/))。Shellito, Small & Livneh (2018) *HESS* 22:1649–1663 では、乾燥速度は雨からの経過日数とともに下がり、SMAP では土性の役割は小さく、**植生が増えると蒸発効率が2〜3倍下がった**([HESS](https://hess.copernicus.org/articles/22/1649/2018/))。McColl et al. (2018) *J. Hydrometeorology* 19(5):871–889 は、土壌水分の水準ごとの負の変化の平均として損失関数(loss function)を推定したが、表層約 5 cm の衛星データでは排水と蒸発散を定量的に分けていない([MIT DSpace](https://dspace.mit.edu/handle/1721.1/119399?show=full))。**どの研究も drydown を収量予測の特徴量には使っておらず、現地センサーの drydown 時定数と収量を直接結んだ査読論文も見つからなかった**。

ユーザーの「雨の翌日〜3日の減少(N_post)だけが情報を持ち、無降雨日の減少(N_dry)は偶然水準」という結果は、排水(drainage)の側の文献につなげやすい。暗渠(tile drainage)で大豆が観測で **+13.66%(+0.498 t/ha)増収**し、その仕組みは湿潤な春の酸素ストレスの軽減と根の発達で、年 ET は暗渠の有無でほぼ同じだったという報告がある(Ma et al. 2025 *HESS* 29:6393–6417、[HESS](https://hess.copernicus.org/articles/29/6393/2025/hess-29-6393-2025.pdf))。日本では大豆の大半が水田転換畑で作られ、排水不良による湿害・夏の干ばつ・可給態窒素の低下が主な減収要因とされる(松本 2026『土壌の物理性』162:15–25、[J-STAGE](https://www.jstage.jst.go.jp/article/jssoilphysics/162/0/162_15/_article/-char/ja))。北海道の26圃場では、透水性(Ib 値)の低い圃場で収量が低かった([道総研](https://www.hro.or.jp/agricultural/center/result/kenkyuseika/gaiyosho/h17gaiyo/f3/2005301.htm))。一方、**雨後の数日は土が湿っていて蒸発散も大きいので、N_post は排水だけを表すとは言えない**。衛星研究で土性の効果が小さく植生の効果が大きいという結果(Shellito et al. 2018)も、「土の性質」という解釈への反論として扱うべきである。ただし衛星は数十 km の空間平均で土性の差がならされるので、圃場スケールにそのまま当てはめられない(推測)。また、N_post の結果は旧日平均(123件)のもので、新データ(`soilmoisture_daily_v2.csv`、121件)ではまだ確認していない。

「k = 3 日」の根拠に圃場容水量の古典的な定義(十分に湿らせた後 2〜3 日)を引く場合、その原典とされる Veihmeyer & Hendrickson (1931) と FAO-56 の該当記述は**未確認**である([Wikipedia](https://en.wikipedia.org/wiki/Field_capacity) に記載があるとされるが二次資料)。現代の教科書は、排水に明確な終点はなく、圃場容水量は土壌固有の値ではないと注意している([OSU オープン教科書](https://open.library.okstate.edu/rainorshine/chapter/8-3-field-capacity/))。修論では「k = 3 日は慣例の範囲内の便宜的な目安であり、k = 2・5 でも AUC 0.651〜0.667 で結果は変わらなかった」と、頑健性の側から書くのが安全である。

### 参考: 衛星・作物モデル・気象のみの研究との違いはスケールと特徴の種類

衛星研究は郡単位の年々変動を、水分や植生の「量」で説明する。Mateo-Sanchis et al. (2019) *Remote Sensing of Environment* 234:111460 は、SMAP の L-band VOD の日時系列と MODIS EVI をカーネルリッジ回帰に入れ、**米国の郡単位の大豆収量で R² 0.90**(VOD のみで 0.86)を得たが、使ったのは土壌水分ではなく VOD で、drydown の動態は使っていない([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6876658))。圃場内の大豆収量を Sentinel-2 と土壌データで予測した研究もある(*Remote Sensing* 14(9):2256, 2022、[DOI 10.3390/rs14092256](https://doi.org/10.3390/rs14092256)、タイトルのみ確認)。気象のみの深層学習では、Shook et al. (2021) *PLoS ONE* 16(6):e0252402 が LSTM と時間方向の attention で品種試験の大豆収量を予測し、**attention は 8〜9月(莢形成・子実肥大期)を重視した**([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8211294))。作物モデル(CROPGRO-Soybean / DSSAT、APSIM)と、気象のみの ML の代表例(Khaki & Wang 2019 など)は、今回一次資料を確認していないので、修論では別途確かめて引用する。対比の軸は、**スケール(郡・地域 対 圃場)、目的変数(年々変動の絶対収量 対 地域×年の中の相対収量)、特徴(水分の量・水準 対 雨への応答の形)**の3点にまとめられる。

### 最も近い研究トップ5と本研究の違い

同じ設計の研究が無いので、「近さ」の観点ごとに5件を選んだ。

| 順位 | 研究 | 近い点 | 本研究との違い |
|---|---|---|---|
| 1 | 熊谷・湊・高橋 (2024)「近年の気候変動と土壌水分の変化が丹波黒大豆主要産地の収量に及ぼした多面的影響分析」日本作物学会紀事 93(4):278–293([農研機構](https://www.naro.go.jp/publicity_report/press/laboratory/niaes/168580.html)、DOI 未確認) | 大豆、時期別の土壌水分の効果、機械学習、効く時期の解釈 | 土壌水分は**推定値**。4地点 × 16年の**年次変動の回帰**で、圃場間の相対収量の分類ではない。気象も説明変数 |
| 2 | Maestrini, B., Basso, B. (2018) "Drivers of within-field spatial and temporal variability of crop yield across the US Midwest" *Scientific Reports* 8:14833, DOI 10.1038/s41598-018-32779-3([Springer](https://link.springer.com/10.1038/s41598-018-32779-3)) | 圃場×年の中で正規化した相対収量、大豆を含む、湿りやすさと収量の不安定性 | **土壌水分は未測定**(TWI・グライ化で代用)。単位は圃場内 30 m セル。時系列入力なし |
| 3 | 北海道立中央農業試験場 (2005)「転換畑における土壌物理性に起因した大豆生産阻害要因の解明と改善指標」(成績概要書、[道総研](https://www.hro.or.jp/agricultural/center/result/kenkyuseika/gaiyosho/h17gaiyo/f3/2005301.htm)) | 日本の転換畑大豆、26の農家圃場、排水性(透水性)と収量 | 土壌水分の**時系列がない**(単発の透水試験)。しきい値による整理。査読論文ではない |
| 4 | Yang, Peng, Guan, Pan, Franz, Cosh, Bernacchi (2024) "Within-field soil moisture variability and time-invariant spatial structures of agricultural fields in the US Midwest" *Vadose Zone Journal* 23(4), DOI 10.1002/vzj2.20337([DOAJ](https://doaj.org/article/255d81b6e7a545fb9f9c2190dd1cc080)) | 農家圃場の in-situ 長期観測、dry-down に注目 | **収量を扱わない**。3圃場のみ。空間構造の記述が目的 |
| 5 | Figueroa, M., Pope, C. (2017) "Root System Water Consumption Pattern Identification on Time Series Data" *Sensors* 17(6):1410, DOI 10.3390/s17061410([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5492835/)) | in-situ 土壌水分の高頻度時系列に時系列データマイニング(SAX)を当て、乾き方のパターンを識別 | **収量を扱わない**。果樹5圃場。評価は precision |

次点は Kinoshita et al. (2022)(26圃場、近接センシング + random forest による収量パターンの要因の解釈、土壌水分時系列なし、[USQ PDF](https://research.usq.edu.au/download/5cfb3cc71cea50bd8eb2a94982ec7452d93c595af25c411716d05f0a11dad3de/2493684/2022_Kinoshita.pdf))。本文を確認していない Green & Erskine (2004)、Paz ほか (1998) は比較から外した。修論の関連研究の章は、(a) 推定土壌水分 × 時期別効果、(b) 相対収量 × 地形水分指標、(c) 排水性 × 転換畑大豆、(d) in-situ の drydown 観測、の4つの流れを整理し、本研究をその交点に置くとまとめやすい。

---

## RQ2-1 ROCKET を採用する根拠: 「過学習しにくい」ではなく「線形層だけを強い L2 正則化で学ぶ」

ROCKET 原論文(Dempster, Petitjean, Webb 2020 *Data Mining and Knowledge Discovery* 34:1454–1495, DOI 10.1007/s10618-020-00701-z、[Springer](https://doi.org/10.1007/s10618-020-00701-z))を全文で確認すると、小標本について言っているのは次の2点だけである。1つ目は、10,000 カーネルで 20,000 特徴量になり、UCR のどのデータでも特徴量の数が件数より多くなりうるが、それでも線形分類器で高精度だったこと(§3.2)。もう1つは §2.4 の「小さいデータでは『良い』カーネルを学習するのが難しいかもしれず、ランダムカーネルが有利かもしれない」で、これは Jarrett et al. 2009 と Yosinski et al. 2014 の引用だけに基づく**検証されていない仮説**である。むしろ原論文は「特徴量の数が学習件数を大きく上回るときは正則化が決定的に重要(critically important)」(§3.3)とし、大きいデータでは正則化の重要性が下がると書いている(§3.4.2)。つまり、**小データで過学習を抑えているのはランダムカーネルではなく ridge の L2 正則化**というのが原論文の立場である。学習件数が特徴量の数より少なければ ridge(GCV で正則化の強さを速く選べる)、多ければロジスティック回帰、という推奨もここから来ている(§3.3)。

MiniRocket(Dempster, Schmidt, Webb 2021 *Proc. KDD '21* pp. 248–257, DOI 10.1145/3447548.3467231、[ACM](https://doi.org/10.1145/3447548.3467231))は、約 10,000 件より少なければ ridge を使い、**学習 50 件の DucksAndGeese でも ridge を使った**(§4.2.2)。109 データ × 30 回の resample で、HIVE-COTE/TDE・TS-CHIEF・InceptionTime・ROCKET と有意差はなかった。乱数を使うのはバイアスを選ぶときの学習例の抽出だけで、完全に決定的な版でも精度はほぼ同じだった(W/D/L 42/11/56)。MultiRocket(Tan, Dempster, Bergmeir, Webb 2022 *DMKD* 36:1623–1646, DOI 10.1007/s10618-022-00844-1、[Springer](https://doi.org/10.1007/s10618-022-00844-1))は、**元の系列に1次差分を加えると判別力が上がる**ことを示しており(§4.3.1–4.3.2)、ユーザーが1次差分を主の入力にしたことの根拠として引用できる。Hydra(Dempster, Schmidt, Webb 2023 *DMKD*, DOI 10.1007/s10618-023-00939-3、[Springer](https://link.springer.com/10.1007/s10618-023-00939-3)、巻号・頁は未確認)の abstract は、小データ・過学習に触れていない。QUANT(Dempster, Schmidt, Webb 2024 *DMKD*, DOI 10.1007/s10618-024-01036-9、[arXiv:2308.00928](https://arxiv.org/pdf/2308.00928))は分類器が extremely randomised trees で ridge ではない。感度分析に使ったのは学習 100 件以上の 50 データだけで、**小データでの挙動は直接評価していない**。

原論文群のどれにも、学習件数別(例: n < 100)の層別の精度は見当たらなかった(ROCKET は全文、MiniRocket・MultiRocket は該当箇所を確認。Hydra は本文未確認)。UCR には学習数十件のデータが多く、ROCKET 論文の Table 1 には学習 20 件の BeetleFly(ROCKET 0.900、HIVE-COTE 0.950)なども含まれる([ROCKET 論文](https://doi.org/10.1007/s10618-020-00701-z)、件数は [Detach-ROCKET](https://arxiv.org/html/2309.14518v3) の Table 2 で確認)。ただし **UCR の多くは正解率 0.8〜1.0 の強い信号の問題**で、ユーザーの AUC 約 0.6 のような弱い信号の問題に実績をそのまま当てはめることはできない。UCR は事前に決まった分割で、group によるデータ漏洩の統制もない。

ユーザーの観察とのつながりも書いておく。ridge の alpha が上限に張り付くのは、GCV が「切片だけに近いモデル」を選んでいる状態であり、**正則化が過学習を抑えているというより、特徴量からの信号が弱いことの表れ**と読むべきである(推測)。修論では「ROCKET 系の設計どおりに ridge を使ったが、正則化はほぼ最大の強さが選ばれた」と正直に書くのがよい。

**修論での書き方の例**: 「ROCKET 系は学習件数が数十〜数百の UCR データで最先端並みの精度を示し(Dempster et al. 2020, 2021; Middlehurst et al. 2024)、調整するハイパーパラメータが少なく、学習するのは線形分類器の重みだけで、L2 正則化の強さを交差検証で選べるので、小標本に適用しやすい。ただし小データに限ると HC2 が上位であり(Middlehurst et al. 2024)、小標本で過学習しにくいことを直接示した研究は確認できなかった」。

---

## RQ2-2 代替手法との比較: 小データの1位は HC2、ROCKET 系は計算時間込みで上位

2024 年の大規模比較 Bake off redux(Middlehurst, Schäfer, Bagnall 2024 *DMKD*, DOI 10.1007/s10618-024-01022-1、[Springer](https://link.springer.com/doi/10.1007/s10618-024-01022-1) / [arXiv:2304.13029v3](https://arxiv.org/html/2304.13029v3)、巻号・頁 38:2958–3031 は未確認)は、MultiROCKET+Hydra(MR-Hydra)と HIVE-COTE 2(HC2)がほかの手法より有意に良いと結論した。下の数値は WebFetch の要約を通して得たもので、**引用前に原表で確かめる必要がある**。

| カテゴリ | 手法 | 112 UCR データの平均正解率(Table 14 ほか) |
|---|---|---|
| ハイブリッド | HC2 | 0.891 |
| 畳み込み(ROCKET 系) | MR-Hydra | 0.884 |
| shapelet | RDST | 0.876 |
| 深層学習 | H-InceptionTime / InceptionTime / ResNet | 0.874〜0.876 / 0.874 / 0.833 |
| 辞書 | WEASEL 2.0 | 0.874 |
| 区間 | QUANT | 0.867 |
| 特徴量 | FreshPRINCE / TSFresh / catch22 | 0.855 / 0.799 / 0.795 |
| 距離 | 1NN-DTW | 0.756 |

**学習件数別の平均順位(Table 16、142 データ)**では、学習 1〜99 件(42 データ)で HC2 2.70(1位)、MR-Hydra 3.68(2位)、RDST 4.33(3位)、H-InceptionTime 5.20(6位)だった。100〜299 件では HC2 2.81、MR-Hydra 3.39、RDST 3.45 で、300 件以上では MR-Hydra が1位になる。著者は HC2 を小さい学習集合に、MR-Hydra を学習件数が多い場合に向くとまとめ、事前情報が無ければこの2つが出発点として妥当としている([arXiv](https://arxiv.org/html/2304.13029v3))。ユーザーの各 fold の学習件数(約95件)は、ちょうど「1〜99件」と「100〜299件」の境目にあたる。**MiniRocket 単体・catch22・1NN-DTW の層別の順位は取得できておらず**、MiniRocket 単体は MR-Hydra より下位と考えられるが未確認である。したがって「ROCKET 系は小データで最良」とは書けず、書けるのは「最良の手法に近い精度を、桁違いに短い計算時間で得られる」までである。総学習時間(142 問題)は MR-Hydra 1.33 時間に対して HC2 263.89 時間で([arXiv](https://arxiv.org/html/2304.13029v3))、grouped CV × 20 回の繰り返し × 置換検定という評価計画に HC2 を組み込むのは現実的でない可能性が高い(推測)。これは MiniRocket を選ぶ実務上の正当な理由になる。

個別の代替手法について。**catch22**(Lubba et al. 2019 *DMKD* 33(6):1821–1852)は、ROCKET 原論文の比較で「最も速いが最も精度が低い」とされ、単一の決定木では精度が低すぎたので random forest に替えて比べた(Appendix E、[ROCKET 論文](https://doi.org/10.1007/s10618-020-00701-z))。ただし解釈しやすく、弱い信号の小標本では ROCKET 系との差が縮む可能性があるので、基準線として外せない。**WEASEL 2.0**(Schäfer & Leser 2023 *Machine Learning*, DOI 10.1007/s10994-023-06395-w、[Springer](https://link.springer.com/doi/10.1007/s10994-023-06395-w))は平均で ROCKET 系に近いが、小データについての記述は確認していない。**shapelet(RDST)**は小データで3位だったが、小データでの挙動を述べた一次資料は確認していない。**深層学習(InceptionTime・ResNet)**は、Ismail Fawaz et al. (2019) *DMKD* 33(4):917–963, DOI 10.1007/s10618-019-00619-1 がパラメータの多さと UCR の小さい学習集合による過学習のリスク、初期値の乱数への感度を指摘している([arXiv:1809.04356](https://ar5iv.arxiv.org/html/1809.04356))。ROCKET 論文では DiatomSizeReduction で ResNet 0.3013 に対し ROCKET 0.9699 のように深層学習が崩れた例が載っている([ROCKET 論文](https://doi.org/10.1007/s10618-020-00701-z))。ただしこれらのデータの学習件数が小さいことは記憶によるもので未確認であり、一般化はほどほどにする。**ロジスティック回帰**については、ROCKET 原論文が、小データでロジスティック回帰を ridge と同じ精度まで調整するのは「かなり難しく時間がかかる」と書いている(§3.3)。一方、要約統計 + ロジスティック回帰は基準線としての役割がある(ユーザーの段階2で実施済み、偶然水準)。

**農業・環境データへの応用例**はほとんど見つからなかった。作物収量・土壌水分・水文で ROCKET 系を使った査読論文は今回の検索では確認できず、衛星時系列の作物分類も Transformer・CNN・RF が中心だった(例: Illarionova et al. 2025 *Sensors* 25(18):5746 は大豆を含む作物分類で RF のみ、[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473680))。確認できた農業系の応用は、子牛30頭の首輪型加速度計による行動分類の**プレプリント**だけで、balanced accuracy は ROCKET 0.70、catch22 0.69、手作り特徴量 0.65、最良は ROCKET + RidgeClassifierCV の 0.77 だった(Dissanayake et al. 2024, [arXiv:2404.18159](https://arxiv.org/abs/2404.18159))。ただしサンプルは3秒窓単位で数が多く、ユーザーの「圃場×年 約120件・grouped CV」と同じ小データの前例とは言えない。回帰では、ROCKET を回帰に改変したものが 19 データの TSER アーカイブで全体として最高精度だった(Tan, Bergmeir, Petitjean, Webb, "Time Series Extrinsic Regression", [arXiv:2006.12672](https://arxiv.org/abs/2006.12672)。掲載誌と、環境系データが含まれるかは未確認)。「農業での前例」は、汎用ベンチマークでの強さ + TSER + 家畜センサーのプレプリント、という形で控えめに書くのが安全で、**土壌水分や作物収量への ROCKET 系の適用自体が新しい可能性**として書ける。

---

## RQ2-3 限界と修論で示すべき基準線

### 主な限界: 高次元、乱数、解釈、時刻の情報

1つ目は**高次元への依存**である。原論文は、多数のカーネルが必要なので分類器が非常に多くの特徴量を扱えるもの(ridge・logistic)に限られると書いている(§3.6、[ROCKET 論文](https://doi.org/10.1007/s10618-020-00701-z))。Detach-ROCKET(Uribarri et al. 2024 *DMKD* 38:3922–3947, DOI 10.1007/s10618-024-01062-7、[Springer](https://link.springer.com/article/10.1007/s10618-024-01062-7))は、ROCKET の特徴量の多くが冗長か無関係で過学習の原因になりうるという仮説を検証した。2値の UCR 42 データで特徴量を 10% に絞ると平均 +0.2% だったが、**MiniRocket では平均 −0.38% と下がった**([arXiv:2309.14518](https://arxiv.org/html/2309.14518v3))。特徴量を絞っても大きな改善は期待しにくい。著者の順序は arXiv 版と出版版で異なって表示されたので、引用時に出版版で確かめる。

2つ目は**乱数による揺れ**である。原論文の正解率の標準偏差の中央値は k = 10,000 で 0.0037 と小さいが(§4.3.1)、これは UCR の大きなテストセットで測った値である。約120件の grouped CV で測る AUC のばらつきはこれより大きいのが自然で(推測)、カーネルと分割の乱数を複数通り回して平均と幅を報告するユーザーの方法は正当である。小標本・弱い信号の条件で揺れを体系的に調べた査読論文は見つからなかった。

3つ目は**解釈しにくさと時刻の情報の欠如**である。MiniRocket の PPV プーリングは波形が「いつ」出たかを捨てる。明示的な時刻符号化を加える HDC-MiniROCKET([arXiv:2202.08055](https://arxiv.org/pdf/2202.08055))と、その後続の DMKD 論文(DOI 10.1007/s10618-025-01162-y、[Springer](https://link.springer.com/article/10.1007/s10618-025-01162-y)、著者・巻頁は未取得)があること自体が、標準の MiniRocket が時刻の位置を持たないことの傍証になる。これはユーザーの結果と整合する。時間の順番の置換検定(段階10・25)で並び方の効果が確認できず、20日ごとの部分窓(段階22)で有意な期間が無かったのは、データの性質だけでなく、**位置をほぼ捨てる特徴量の設計にも由来する可能性がある**(推測)。「開花後のどの時期が効くか」という問いに、標準の MiniRocket の特徴量だけで答えるのは難しい。ROCKET 専用の査読付きの解釈法は Detach-ROCKET だけで、それも「残ったカーネルの dilation や活性化の位置で時間尺度・時期がわかるかもしれない」と提案するだけで、実例は示していない([arXiv](https://arxiv.org/html/2309.14518v3))。時系列の saliency 法は重要な時点をしばしば正しく特定できないというベンチマーク結果もある(Ismail et al., NeurIPS 2020, [arXiv:2010.13924](https://arxiv.org/abs/2010.13924))。摂動の種類と分類器を変えて説明法を評価する枠組み AMEE も**プレプリント**の段階である([arXiv:2306.05501](https://arxiv.org/abs/2306.05501v1))。このため、ユーザーの「入力を成分に分けて仮説を検定する(N_post / N_dry / R)」アブレーション(ablation)を主な証拠にし、事後的な重要度マップは補助にする構成が安全である(推測)。

4つ目は**入力の前提**である。ROCKET のカーネルの重みとバイアスの分布は、入力が平均0・標準偏差1に z 正規化されている前提で決めてある(§3.1、[ROCKET 論文](https://doi.org/10.1007/s10618-020-00701-z))。1次差分の入力はこの前提とは別の形になる。sktime の MiniRocket のゼロパディングで平行移動に不変でない点を論じた文献は見つからず、ユーザーの EXP-009 の観察が唯一の根拠である。

### 修論で示すべき基準線(baseline)

ROCKET 系の採用を正当化する最小のセットは次のとおりである。ユーザーがすでに実施したものと、未実施のものを分けて示す。

| 基準線 | 目的 | 状況 |
|---|---|---|
| 置換検定の帰無分布の95%点(AUC 0.5 ではなく同じ分割で) | 偶然の水準 | 実施済み |
| 系列の要約統計 + L2 ロジスティック回帰 | 時系列の形を使う意味があるか | 実施済み(段階2、B1) |
| 地域・播種日・品種だけのモデル | 交絡で説明できるか | 実施済み(B2・B3) |
| 降雨後の減少率(drydown・loss function の傾き)の手作り特徴量 + 線形モデル | N_post の情報が ROCKET でなければ取れないか | 未実施(段階30の提案、McColl et al. 2018 の損失関数が参考になる) |
| catch22 + ロジスティック回帰(または RF) | 標準の特徴量ベースの基準線(ROCKET 論文・bake off redux で使用) | 未実施 |
| 1NN-DTW(または 1NN-Euclid) | TSC の古典的な基準線 | 未実施 |
| (計算が許せば)MR-Hydra、または QUANT | ROCKET 系・区間法の中での上限の目安 | 一部(MultiRocket は段階17で実施) |

特に、**降雨後の減少率の手作り特徴量が MiniRocket と同程度の AUC を出すなら、修論の主張は「ROCKET が必要」から「雨の後の減少という特徴が効く。それを ROCKET 系で見つけた」に変わる**。これはむしろ、解釈と新規性の両方で強い主張になりうる。

---

## 修論で主張できそうな新規性: 言える範囲と言えない範囲

| 区分 | 主張の候補 | 根拠と条件 |
|---|---|---|
| **言える** | 多数の農家圃場の in-situ VWC 日次時系列を、収量の分類に直接使った | 調査範囲で同じ設計の研究は見つからなかった。ただし「確認した範囲では」と限定し、検索方法(データベース・検索語)を明記する |
| **言える** | 地域×年の中の相対ラベル + 市町村単位の grouped CV + 集団内置換検定 + 品種の部分集合による交絡の確認 | 相対収量の発想は Maestrini & Basso (2018) に前例があるので引用し、それを圃場間の分類と土壌水分時系列に組み合わせた点を新規性とする |
| **言える** | ROCKET 系(MiniRocket)を土壌水分・作物収量に当てた | 農業・環境への ROCKET 系の適用は、確認できた範囲では家畜センサーのプレプリントだけ |
| **言える(条件付き)** | 降雨直後(翌日〜3日)の減少に収量の情報があり、無降雨日の減少や雨の日の位置には無い | drydown を収量の特徴量に使った例は見つからなかった。**ただし結果は旧日平均のもので、新データ(121件)での再確認と、判定を固定した再確認・年をまたぐ確認が必要** |
| **言える(記述として)** | 交絡を統制した評価の下で、全体として偶然以上に見分けられる(AUC 約 0.61) | D-007 の判定(全体 AUC)に沿う。品種が集団内で同じ部分集合で 0.678、品種+播種日の基準線は偶然水準 |
| **言えない** | 「世界初」「前例がない」 | 検索は網羅的でなく、J-STAGE・ASABE の本文を読めていない |
| **言えない** | 同じ地域の中で土壌水分の効果を切り分けた | 地域内 W は確認できていない(主に地域間の判別)。件数が足りず検出力が低い |
| **言えない** | N_post は土の排水性(土の性質)を表す | 雨後数日は蒸発散も大きく分離できない。衛星研究では植生の効果が大きいという反対の結果もある。「排水を反映する可能性」までにとどめる |
| **言えない** | ROCKET は小標本で過学習しにくいので選んだ | 原論文は明言していない。小データの1位は HC2。「線形層だけを L2 正則化で学ぶので小標本に適用しやすく、計算が軽いので繰り返しの評価と置換検定ができる」と書き直す |
| **言えない** | 開花後の特定の時期が効く | 部分窓で有意な期間なし。MiniRocket の特徴量は時刻の位置をほぼ捨てる。文献上の感受性の時期(R1〜R3 の過湿、R5 以降の干ばつ)は「矛盾しない」程度にとどめる |
| **言えない** | k = 3 日は圃場容水量の定義に基づく | 原典(Veihmeyer & Hendrickson 1931、FAO-56)の記述は未確認。便宜的な目安とし、k を変えても結果が変わらないことで支える |

---

## Conclusion

調査で見えてきたのは、本研究の新しさが「手法」よりも「データと問いの組み合わせ」にあるということである。土壌水分と収量の文献は、推定値・地形の代理変数・単発の透水試験・収量を扱わないセンサー観測にばらばらに分かれていて、多数の農家圃場のセンサー時系列を地域×年の中で比べた研究はその隙間に位置する。drydown は水文学で詳しく研究されているのに、収量の説明に使われていない。この点で、雨の直後の減少に情報があるという結果は、水文学(drydown・損失関数)と作物学(転換畑の排水と湿害)をつなぐ論点として、修論の考察の中心にできる。そのためには、新データでの再確認と、降雨後の減少率の手作り特徴量という単純な基準線が、追加の ROCKET の実験より優先度が高い。

ROCKET 系の妥当性は、「小標本で過学習しにくい」という性質の主張ではなく、「弱い信号の小標本で、交絡を統制した厳しい評価(grouped CV × 繰り返し × 置換検定)を現実的な計算時間で回せる、最先端に近い時系列分類器」という実務上の選択として示すのが筋がよい。alpha の上限への張り付き、時刻の位置を捨てる特徴量、小データでは HC2 が上位という事実は、隠すより限界として明記したほうが審査で強い。そのうえで catch22 と 1NN-DTW を加えれば、採用の根拠は文献と実験の両方で閉じる。

---

## 参考文献一覧(本文で引用したもの)

確認の程度を併記する。「確認済み」= 本文または公式要旨を取得、「一部確認」= 二次的な要約や一部のみ、「タイトルのみ」= 書誌だけ、「未確認」= 記憶または二次資料。

**RQ1(土壌水分 × 収量、時期・drydown、排水)**
- 熊谷悦史・湊政徳・高橋智紀 (2024)「近年の気候変動と土壌水分の変化が丹波黒大豆主要産地の収量に及ぼした多面的影響分析」日本作物学会紀事 93(4):278–293。DOI 未確認。[農研機構プレスリリース](https://www.naro.go.jp/publicity_report/press/laboratory/niaes/168580.html)(プレスリリースで確認)
- Maestrini, B., Basso, B. (2018) "Drivers of within-field spatial and temporal variability of crop yield across the US Midwest." *Scientific Reports* 8:14833. DOI 10.1038/s41598-018-32779-3(確認済み)
- 北海道立中央農業試験場 (2005)「転換畑における土壌物理性に起因した大豆生産阻害要因の解明と改善指標」成績概要書。[道総研](https://www.hro.or.jp/agricultural/center/result/kenkyuseika/gaiyosho/h17gaiyo/f3/2005301.htm)(確認済み、査読論文ではない)
- 石塚明子・荒木雅登・大野礼成・岩渕哲也 (2024)「明渠と弾丸暗渠を活用した土壌水分管理が大豆の生産性に及ぼす影響」福岡県農林業総合試験場研究報告 10:1–6. DOI 10.60399/fafrc.10.0_1(要旨で確認)
- Wraith, J. M., Robinson, D. A., Jones, S. B., Long, D. S. (2005) "Spatially characterizing apparent electrical conductivity and water content of surface soils with time domain reflectometry." *Computers and Electronics in Agriculture* 46:239–261. DOI 10.1016/j.compag.2004.11.009(確認済み)
- Yang, Peng, Guan, Pan, Franz, Cosh, Bernacchi (2024) "Within-field soil moisture variability and time-invariant spatial structures of agricultural fields in the US Midwest." *Vadose Zone Journal* 23(4). DOI 10.1002/vzj2.20337(要旨で確認)
- Figueroa, M., Pope, C. (2017) "Root System Water Consumption Pattern Identification on Time Series Data." *Sensors* 17(6):1410. DOI 10.3390/s17061410(確認済み)
- Kinoshita, R., Tani, M., Sherpa, S., Ghahramani, A., van Es, H. M. (2022) "Soil Sensing and Machine Learning Reveal Factors Affecting Maize Yield in the Mid-Atlantic USA." DOI 10.1002/agj2.21223(誌名 *Agronomy Journal* は DOI からの推定。受理原稿で確認)
- Green, T. R., Erskine, R. H. (2004) "Measurement, scaling, and topographic analyses of spatial crop yield and soil water content." *Hydrological Processes*. DOI 10.1002/hyp.1422(タイトルのみ)
- Paz, J. O. ほか (1998) "Analysis of water stress effects causing spatial yield variability in soybeans." *Transactions of the ASAE* 41(5)([ASABE](https://elibrary.asabe.org/abstract.asp?JID=3&AID=17284&CID=t1998&v=41&i=5&T=1)、タイトルのみ)
- Eck, H. V., Mathers, A. C., Musick, J. T. (1987) "Plant water stress at various growth stages and growth and yield of soybeans." *Field Crops Research* 17(1):1–16([AGRIS](https://agris.fao.org/search/en/records/65dee5040f3e94b9e5d27576)、抄録で確認)
- Ploschuk, R. A., Miralles, D. J., Striker, G. G. (2022) "A quantitative review of soybean responses to waterlogging." *Plant and Soil* 475:237–252. DOI 10.1007/s11104-022-05364-x(確認済み)
- Adegoye, G. A. ほか (2023) "Waterlogging Effects on Soybean Physiology and Hyperspectral Reflectance during the Reproductive Stage." *Agriculture* 13(4):844. DOI 10.3390/agriculture13040844(確認済み)
- McColl, K. A. ほか (2017) "Global characterization of surface soil moisture drydowns." *Geophysical Research Letters*. DOI 10.1002/2017GL072819(本文未取得、Shellito et al. 2018 による要約で確認)
- McColl, K. A., Haghighi, E., Salvucci, G. D., Akbar, R., Gianotti, D. J. S., Entekhabi, D. (2018) "Estimation of Landscape Soil Water Losses from Satellite Observations of Soil Moisture." *Journal of Hydrometeorology* 19(5):871–889. DOI 10.1175/JHM-D-17-0200.1(確認済み)
- Shellito, P. J., Small, E. E., Livneh, B. (2018) "Controls on surface soil drying rates observed by SMAP and simulated by the Noah land surface model." *HESS* 22:1649–1663. DOI 10.5194/hess-22-1649-2018(確認済み)
- Ma ほか (2025) "Soil oxygen dynamics: a key mediator of tile drainage impacts on coupled hydrological, biogeochemical, and crop systems." *HESS* 29:6393–6417. DOI 10.5194/hess-29-6393-2025(確認済み)
- 松本 (2026)「ダイズ種子の出芽に悪影響を及ぼす粘土質転換畑の砕土不良な作土層における乾燥過程の水分動態」土壌の物理性 162:15–25. DOI 10.34467/jssoilphysics.162.0_15(要旨で確認)
- Mateo-Sanchis, A. ほか (2019) "Synergistic integration of optical and microwave satellite data for crop yield estimation." *Remote Sensing of Environment* 234:111460. DOI 10.1016/j.rse.2019.111460(確認済み)
- Shook, J. ほか (2021) "Crop yield prediction integrating genotype and weather variables using deep learning." *PLoS ONE* 16(6):e0252402. DOI 10.1371/journal.pone.0252402(確認済み)
- "Soya Yield Prediction on a Within-Field Scale Using Machine Learning Models Trained on Sentinel-2 and Soil Data." *Remote Sensing* 14(9):2256 (2022). DOI 10.3390/rs14092256(タイトルのみ)
- Veihmeyer, F. J., Hendrickson, A. H. (1931) *Soil Science* 32:181–193(未確認)

**RQ2(ROCKET 系と代替手法)**
- Dempster, A., Petitjean, F., Webb, G. I. (2020) "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels." *Data Mining and Knowledge Discovery* 34:1454–1495. DOI 10.1007/s10618-020-00701-z(全文確認)
- Dempster, A., Schmidt, D. F., Webb, G. I. (2021) "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." *Proc. KDD '21*, pp. 248–257. DOI 10.1145/3447548.3467231(該当箇所を確認)
- Tan, C. W., Dempster, A., Bergmeir, C., Webb, G. I. (2022) "MultiRocket: multiple pooling operators and transformations for fast and effective time series classification." *DMKD* 36:1623–1646. DOI 10.1007/s10618-022-00844-1(該当箇所を確認)
- Dempster, A., Schmidt, D. F., Webb, G. I. (2023) "Hydra: competing convolutional kernels for fast and accurate time series classification." *DMKD*. DOI 10.1007/s10618-023-00939-3(abstract のみ。巻号・頁は未確認)
- Dempster, A., Schmidt, D. F., Webb, G. I. (2024) "QUANT: A Minimalist Interval Method for Time Series Classification." *DMKD*. DOI 10.1007/s10618-024-01036-9(arXiv 版で確認)
- Middlehurst, M., Schäfer, P., Bagnall, A. (2024) "Bake off redux: a review and experimental evaluation of recent time series classification algorithms." *DMKD*. DOI 10.1007/s10618-024-01022-1(arXiv 版を要約経由で確認。巻号・頁と Table 16 の数値は原表で要確認)
- Bagnall, A., Lines, J., Bostrom, A., Large, J., Keogh, E. (2017) "The great time series classification bake off." *DMKD* 31(3):606–660(書誌は ROCKET 論文の参考文献で確認、DOI は未確認)
- Lubba, C. H. ほか (2019) "catch22: CAnonical Time-series CHaracteristics." *DMKD* 33(6):1821–1852(書誌は ROCKET 論文の参考文献で確認)
- Schäfer, P., Leser, U. (2023) "WEASEL 2.0: a random dilated dictionary transform for fast, accurate and memory constrained time series classification." *Machine Learning*. DOI 10.1007/s10994-023-06395-w(本文の小データの記述は未確認)
- Ismail Fawaz, H., Forestier, G., Weber, J., Idoumghar, L., Muller, P.-A. (2019) "Deep learning for time series classification: a review." *DMKD* 33(4):917–963. DOI 10.1007/s10618-019-00619-1(arXiv 版を要約経由で確認)
- Uribarri, G. ほか (2024) "Detach-ROCKET: sequential feature selection for time series classification with random convolutional kernels." *DMKD* 38:3922–3947. DOI 10.1007/s10618-024-01062-7(arXiv 版で確認。著者の順序は出版版で要確認)
- Tan, C. W., Bergmeir, C., Petitjean, F., Webb, G. I. "Time Series Extrinsic Regression." arXiv:2006.12672(掲載誌は未確認)
- Dissanayake, O. ほか (2024) "Evaluating ROCKET and Catch22 features for calf behaviour classification from accelerometer data using Machine Learning models." arXiv:2404.18159(**プレプリント**)
- Illarionova ほか (2025) *Sensors* 25(18):5746. DOI 10.3390/s25185746(確認済み、ROCKET は不使用)
- HDC-MiniROCKET, arXiv:2202.08055(プレプリント)、後続 *DMKD* DOI 10.1007/s10618-025-01162-y(著者・巻頁は未取得)
- Ismail, A. A., Gunady, M., Corrada Bravo, H., Feizi, S. (2020) "Benchmarking Deep Learning Interpretability in Time Series Predictions." NeurIPS 2020. arXiv:2010.13924
- Nguyen, T. T., Le Nguyen, T., Ifrim, G. "AMEE: A Robust Framework for Explanation Evaluation in Time Series Classification." arXiv:2306.05501(**プレプリント**)
