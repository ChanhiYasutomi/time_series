# ARIMA（AutoRegressive Integrated Moving Average）モデルは、時系列データの予測とモデリングに使用される統計的な手法です。
# ARIMAモデルは、自己回帰（AR：AutoRegressive）成分、差分（I：Integrated）成分、移動平均（MA：Moving Average）成分から構成されます。
# 以下に、Pythonのコードを使用してARIMAモデルを作成し、具体的な例を説明します。
# まず、必要なライブラリをインポートします。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 次に、サンプルの時系列データを作成します。ここでは、サンプルデータをランダムに生成します。
# サンプルデータを生成
np.random.seed(0)
n = 200  # データポイントの数
noise = np.random.normal(0, 1, n)  # ノイズ

# ARMA(2, 1)モデルのパラメータ
phi1, phi2 = 0.5, -0.2  # AR成分の係数
theta1 = 0.8  # MA成分の係数

# ARMA(2, 1)モデルのデータ生成
data = np.zeros(n)
for i in range(2, n):
    data[i] = phi1 * data[i - 1] + phi2 * data[i - 2] + noise[i] + theta1 * noise[i - 1]

# データフレームに格納
df = pd.DataFrame({'Data': data})

# 次に、ARIMAモデルを作成し、データにフィットさせます。
# ARIMAモデルのフィッティング
model = sm.tsa.ARIMA(df['Data'], order=(2, 1, 1))  # (p, d, q) = (2, 1, 1)
result = model.fit()

# ここで、(2, 1, 1) はARIMAモデルの次数を指定しています。p はAR次数、d は差分次数、q はMA次数です。
# この例では、ARIMA(2, 1, 1)モデルを作成しています。

# 最後に、データとモデルの適合結果をプロットして表示します。

# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], label='Data')
plt.plot(result.fittedvalues, color='red', label='Fitted Values (ARIMA)')
plt.title('ARIMA(2, 1, 1) Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)
plt.show()

# このコードにより、ARIMAモデルを使用して時系列データをモデリングし、適合結果を視覚化できます。 ARIMAモデルは、時系列データのトレンドや季節性を考慮して予測や分析を行うのに役立ちます。
