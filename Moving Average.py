# MA (Moving Average) モデルは、過去の観測値の移動平均を使用して未来のデータを予測するためのモデルです。単純なMA(1)モデルは、直近の1つまたは複数の過去の観測値の平均を使用します。以下に、Pythonコードと具体例を示します。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータを生成
np.random.seed(0)
n = 100  # データポイントの数
noise = np.random.normal(0, 1, n)  # ノイズ
data = np.zeros(n)

# MA(1)モデルのパラメータ
theta = 0.6  # 移動平均係数

# MA(1)モデルのデータ生成
for i in range(1, n):
    data[i] = noise[i] + theta * noise[i - 1]

# データフレームに格納
df = pd.DataFrame({'Data': data})

# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'])
plt.title('MA(1) Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.grid(True)
plt.show()

# このコードでは、MA(1)モデルを使ってランダムなノイズに移動平均係数 theta をかけた値を生成しています。この値は未来のデータを予測するための観測値です。
# 生成されたデータはプロットされています。MA(1)モデルは直近の過去のノイズ成分の平均値が現在のデータを決定することを示しています。
