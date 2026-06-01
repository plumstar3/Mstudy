# -*- coding: utf-8 -*-
#
# 茨城県つくば市館野における日平均気温の折れ線グラフを作成します。
#	OHNO, Hiroyuki 2012.08.20   

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as md
import AMD_Tools4 as AMD


# 計算の領域と期間の指定
element = "TMP_mea"
lalodomain = [ 36.0566, 36.0566, 140.125, 140.125]	#つくば(舘野)
p = datetime.today()				#今日の時刻オブジェクト
db = p - timedelta(days=64)		#期間の始まり、今日-days
de = p + timedelta(days=64)		#期間の終わり、今日+days
date_begin = db.strftime("%Y-%m-%d")
date_end   = de.strftime("%Y-%m-%d")
timedomain = [ date_begin, date_end ]

# メッシュデータの取得
T0, tim, lat, lon = AMD.GetMetData(element, timedomain, lalodomain, cli=True)
T1, tim, lat, lon, nam, uni = AMD.GetMetData(element, timedomain, lalodomain, namuni=True)
T0 = T0[:,0,0]
T1 = T1[:,0,0]
#tim = np.array([datetime(x.year,x.month,x.day) for x in tim]) #エラーが表示される場合はこの行を有効にしてください。

# 時系列グラフの描画
#－－－－－－－－－－－
D1D1 = T1
D1D2 = T0
title = "N"+str(lalodomain[0])+", E"+str(lalodomain[2])+' ('+p.strftime("%Y/%m/%d")+')'
fig = plt.figure(num=None, figsize=(12, 4))
# ・目盛の作成
ax = plt.axes()
xmajoPos = md.DayLocator(bymonthday=[1])
xmajoFmt = md.DateFormatter('%m/%d')
ax.xaxis.set_major_locator(xmajoPos)
ax.xaxis.set_major_formatter(xmajoFmt)
xminoPos = md.DayLocator()
ax.xaxis.set_minor_locator(xminoPos)
# ・データのプロット
ax.fill_between(tim,D1D1,D1D2,where=D1D1>D1D2,facecolor='orange',alpha=0.5,interpolate=True)	#細線より高い部分を橙色に塗る
ax.fill_between(tim,D1D2,D1D1,where=D1D1<D1D2,facecolor='skyblue',alpha=0.5,interpolate=True)	#細線より低い部分を水色に塗る
ax.plot(tim, D1D1, 'k')						#太線
ax.plot(tim, D1D2, 'k', linewidth=0.3)	#細線
# ・「今日」印を付ける
p = datetime.today()							      			#今日の時刻オブジェクト
today = tim == datetime(p.year,p.month,p.day,0,0,0)		#今日の配列要素番号
plt.plot(tim[today], D1D1[today], "ro")						#今日に赤点を打つ
# ・ラベル、タイトルの付加
plt.xlabel('Date')
plt.ylabel(nam+' [' + uni + ']')
plt.title(title)
plt.savefig('result'+'.png', dpi=600) #この文をコメントアウトすると、図のpngファイルは作られません。
plt.show()
plt.clf()
#－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
# データの保存
Table = np.array([T0,T1])
AMD.PutCSV_MT(Table, tim, header='Date,Normal,Obs.')			#単純時系列のCSVファイル出力

