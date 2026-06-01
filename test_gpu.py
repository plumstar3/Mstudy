import cupy as cp

# GPUを使って配列を作成
x = cp.array([1, 2, 3, 4, 5])

print("配列のデータ:", x)
print("使用しているデバイス:", x.device)