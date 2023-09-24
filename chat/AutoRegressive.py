# AR (AutoRegressive) モデルは、過去のラグされた（過去の時間ステップの）データが現在のデータに影響を与えるモデルです。単純なAR(1)モデルは、1つ前の時点のデータの影響を考慮するものです。以下に、Pythonコードと具体例を示します。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータを生成
np.random.seed(0)
n = 100  # データポイントの数
noise = np.random.normal(0, 1, n)  # ノイズ
data = np.zeros(n)

# AR(1)モデルのパラメータ
phi = 0.7  # 自己回帰係数

# AR(1)モデルのデータ生成
for i in range(1, n):
    data[i] = phi * data[i - 1] + noise[i]

# データフレームに格納
df = pd.DataFrame({'Data': data})

# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'])
plt.title('AR(1) Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.grid(True)
plt.show()

# このコードでは、AR(1)モデルを使って過去のデータが現在のデータに影響を与える人工的な時系列データを生成しています。
# 過去のデータに自己回帰係数 phi をかけた値にランダムなノイズを加えています。
# 最終的に、生成されたデータをプロットしています。このモデルは1つ前の時点のデータが現在のデータに影響を与えることを示しています。
