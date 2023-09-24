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
display(result.summary())
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], label='Data')
plt.plot(result.fittedvalues, color='red', label='Fitted Values (SARIMAX)')
plt.title('SARIMAX(1, 1, 1)(1, 1, 1, 12) Model with Exogenous Factor')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)
plt.show()



# result.summary()は、Pythonの統計モデリングライブラリであるStatsmodelsを使用して作成した統計モデルの要約結果を表示するためのコードです。
# Statsmodelsを使用して統計モデルをフィットさせた後、result.summary() メソッドを呼び出すと、その統計モデルに関する詳細な情報がまとまった要約テーブルとして表示されます。このテーブルには、以下のような情報が含まれています。
# モデルの要約情報: モデルの名前、使用したデータ数、パラメータ数などの基本的な情報が含まれます。
# モデル係数の統計情報: モデル内の各パラメータ（係数）に関する統計的な情報が表示されます。これには、係数の値、標準誤差、t-統計量、p-値などが含まれます。
# モデルの当てはまり度合い: モデルの適合度を示す指標として、尤度、AIC、BICなどが表示されます。
# 残差の統計情報: モデルの残差に関する統計的な情報が含まれます。残差の平均、標準偏差、最小値、最大値、25パーセンタイル、50パーセンタイル（中央値）などが表示されます。
# この要約テーブルは、統計モデルの適切性を評価し、モデルの係数や統計的な有意性を確認するのに役立ちます。特に、p-値は係数の統計的有意性を示し、小さな値ほどその係数が有意であることを示します。統計モデリングにおいて結果を評価するために重要な情報源です。

# このコードにより、SARIMAXモデルを使用して季節性データと外部要因を同時にモデル化し、適合結果を視覚化できます。
# SARIMAXモデルは、外部要因が時系列データに与える影響を考慮して高度な予測と分析を行うのに役立ちます。
