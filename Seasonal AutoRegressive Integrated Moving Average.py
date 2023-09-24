# SARIMA（Seasonal AutoRegressive Integrated Moving Average）モデルは、ARIMAモデルを季節成分を考慮したバージョンです。
# 季節性のある時系列データをモデル化するために使用されます。以下に、Pythonのコードを使用してSARIMAモデルを作成し、具体的な例を説明します。
# まず、必要なライブラリをインポートします。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 次に、サンプルの季節性を持つ時系列データを作成します。ここでは、サンプルデータをランダムに生成します。
# サンプルデータを生成
np.random.seed(0)
n = 200  # データポイントの数
seasonality = 10 * np.sin(np.arange(n) * (2 * np.pi) / 12)  # 季節性成分
noise = np.random.normal(0, 1, n)  # ノイズ

# SARIMA(1, 1, 1)(1, 1, 1, 12)モデルのデータ生成
data = seasonality + noise

# データフレームに格納
df = pd.DataFrame({'Data': data})

# 次に、SARIMAモデルを作成し、データにフィットさせます。
# SARIMAモデルのフィッティング
model = sm.tsa.SARIMAX(df['Data'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # (p, d, q) = (1, 1, 1), (P, D, Q, s) = (1, 1, 1, 12)
result = model.fit()

# ここで、(1, 1, 1) はARIMAモデルの次数を指定し、(1, 1, 1, 12) は季節性の次数を指定しています。p はAR次数、d は差分次数、q はMA次数、(P, D, Q, s) は季節性成分の次数です。この例では、SARIMA(1, 1, 1)(1, 1, 1, 12)モデルを作成しています。
# 最後に、データとモデルの適合結果をプロットして表示します。
# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], label='Data')
plt.plot(result.fittedvalues, color='red', label='Fitted Values (SARIMA)')
plt.title('SARIMA(1, 1, 1)(1, 1, 1, 12) Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)
plt.show()

# このコードにより、SARIMAモデルを使用して季節性を考慮した時系列データをモデリングし、適合結果を視覚化できます。 SARIMAモデルは、季節性のあるデータに対する高度な予測と分析に役立ちます。
