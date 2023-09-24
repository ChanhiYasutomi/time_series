# ARMA (AutoRegressive Moving Average) モデルは、自己回帰 (AR) モデルと移動平均 (MA) モデルを組み合わせた統計モデルで、時系列データの予測や分析に使用されます。
# ARMAモデルは、過去の観測値の自己回帰成分と過去の誤差項の移動平均成分を組み合わせて、未来のデータを予測します。
# 以下に、Pythonコードと具体例を示します。この例では、ARMA(2, 1) モデルを使用してサンプルデータを生成し、プロットしています。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

# ARMAモデルのフィッティング
model = sm.tsa.ARIMA(df['Data'], order=(2, 0, 1))
result = model.fit()

# プロット
plt.figure(figsize=(10, 6))
plt.plot(df['Data'], label='Data')
plt.plot(result.fittedvalues, color='red', label='Fitted Values (ARMA)')
plt.title('ARMA(2, 1) Model')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend()
plt.grid(True)
plt.show()

# このコードは、ARMA(2, 1) モデルを使用してサンプルデータをフィッティングし、データとフィットされたモデルをプロットするものです。以下は各部分の説明です：

# サンプルデータの生成:
# np.random.seed(0) を使用して乱数のシードを固定します。これにより再現性が確保されます。
# n は生成するデータポイントの数です。
# noise は平均0、標準偏差1の正規分布から抽出されたノイズです。

# ARMA(2, 1) モデルのパラメータ:
# phi1 と phi2 は AR 成分の係数です。
# theta1 は MA 成分の係数です。

# ARMA(2, 1) モデルのデータ生成:
# data 配列は、ARMA(2, 1) モデルに従って生成されたデータです。ループを使用して、AR 成分、MA 成分、およびノイズを考慮してデータを生成します。

# データをデータフレームに格納:
# pd.DataFrame() を使用してデータをデータフレームに格納します。列名は 'Data' です。

# ARMA モデルのフィッティング:
# sm.tsa.ARIMA() を使用して ARMA モデルを作成し、データにフィットさせます。order パラメータでモデルの次数を指定します。

# プロット:
# plt.plot() を使用して元のデータとフィットされたモデルをプロットします。
# 'red' で指定したカラーでフィットされたモデルをプロットします。
# プロットのタイトルや軸ラベルを設定し、凡例とグリッドを追加します。
# このコードを実行すると、元のデータとARMA(2, 1) モデルによるフィットが同じプロット上に表示されます。モデルがデータにどれだけ適合しているかを視覚的に確認できます。
