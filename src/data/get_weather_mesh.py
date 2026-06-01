import pandas as pd
import sqlite3
import numpy as np
import sys, os

import AMD_Tools4_2 as amd

# 変数に取得条件を設定
nani = 'APCPRA'                         #気象要素の指定。TMP_meaは日平均気温を意味します。
itsu = ['2010-06-01', '2010-06-03']         #期間の設定。
doko = [40.62268, 40.62268, 140.52278, 140.52278]    #領域の設定。室戸岬の先端あたりです。

# 設定に基づき気象データを取得
data, tim, lat, lon = amd.GetMetData(nani, itsu, doko)

print(data)