# SARIMAX（Seasonal AutoRegressive Integrated Moving Average with eXogenous factors）モデルは、季節性成分を考慮したARIMAモデルに、外部の説明変数（exogenous factors）を組み合わせたモデルです。
# これにより、季節性データと外部要因の影響を同時にモデル化できます。以下に、Pythonのコードを使用してSARIMAXモデルを作成し、具体的な例を説明します。
# まず、必要なライブラリをインポートします。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 次に、サンプルの季節性データと外部要因（exogenous factors）を生成します。
# サンプルデータを生成
np.random.seed(0)
n = 200  # データポイントの数
seasonality = 10 * np.sin(np.arange(n) * (2 * np.pi) / 12)  # 季節性成分
noise = np.random.normal(0, 1, n)  # ノイズ

# 外部要因データを生成
exogenous_factor = np.random.normal(0, 1, n)

# SARIMAX(1, 1, 1)(1, 1, 1, 12)モデルのデータ生成
data = seasonality + noise

# データフレームに格納
df = pd.DataFrame({'Data': data, 'ExogenousFactor': exogenous_factor})
# 次に、SARIMAXモデルを作成し、データにフィットさせます。

# SARIMAXモデルのフィッティング
model = sm.tsa.SARIMAX(df['Data'], exog=df['ExogenousFactor'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()
ここで、(1, 1, 1) はARIMAモデルの次数を指定し、(1, 1, 1, 12) は季節性の次数を指定します。また、exog パラメータに外部要因のデータを指定しています。

# 最後に、データとモデルの適合結果をプロットして表示します。
# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], label='Data')
plt.plot(result.fittedvalues, color='red', label='Fitted Values (SARIMAX)')
plt.title('SARIMAX(1, 1, 1)(1, 1, 1, 12) Model with Exogenous Factor')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)
plt.show()

# このコードにより、SARIMAXモデルを使用して季節性データと外部要因を同時にモデル化し、適合結果を視覚化できます。
# SARIMAXモデルは、外部要因が時系列データに与える影響を考慮して高度な予測と分析を行うのに役立ちます。
