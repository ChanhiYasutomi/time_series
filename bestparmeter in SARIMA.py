import warnings
warnings.filterwarnings("ignore") #0で割ったり、logの中が0になった時にwarningsが出ると行数が多くなるから無視する必要がある

best_result = [0, 0, 10000000]

# パラメータの組み合わせを試行
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            # SARIMAモデルの構築とフィット
            mod = SARIMAX(y, order = param, seasonal_order = param_seasonal)
            results = mod.fit()
            print('ARIMA parameter{}, Seasonal variation parameters{} - AIC: {}'.format(param, param_seasonal, results.aic))
            
            if results.aic < best_result[2]:
                # param: ARとMA
                # param_seasonal: sp = sd = sq(季節の周期性)
                # results.aic: 上記の時のAIC
                # 最初の10000000と比較して、results.aicが低ければ更新する(パラメータに対してaicが最小であるのものを更新していく) ->  現在のAICが過去の最小AICより小さい場合、最適なパラメータを更新
                best_result = [param, param_seasonal, results.aic]
        except:
            continue #計算をtryしてできない場合はcontinue(skip)する -> エラーが発生した場合はスキップ

# 最適なモデルのパラメータとAICを表示
print('AIC min model：', best_result)

# このコードは、SARIMAモデルのパラメータ選択を行うための反復処理を行っています。SARIMAモデルは季節性を考慮した時系列データの予測に使用されます。以下はこのコードの詳細な説明です。
# warnings.filterwarnings("ignore") ラインは、ワーニングメッセージを表示しないようにするためのものです。特に、ゼロで割るなどのエラーが発生する可能性がある場合、ワーニングメッセージが多くなるため、これを無視する設定が行われています。
# best_result リストは、最良のモデルのパラメータとそのAIC値を格納するための変数です。最初に大きなAIC値で初期化されています。

# for param in pdq: ループは、ARIMAモデルの次数の候補（pdq）を反復処理します。
# for param_seasonal in seasonal_pdq: ループは、季節性のパラメータの候補（seasonal_pdq）を反復処理します。
# try ブロック内では、指定されたARIMAモデルのパラメータを使用してモデルをフィットさせ、AIC（赤池情報量基準）を計算します。AICはモデルの良さを評価する指標で、小さな値ほど良いモデルを示します。
# モデルのフィットが成功した場合、results.aic でAIC値を取得し、これを表示します。また、もし現在のAIC値が best_result のAIC値よりも小さい場合、best_result を更新します。これにより、現在のモデルがより適している場合に最適なパラメータが更新されます。
# except ブロック内では、モデルのフィットが失敗した場合にエラーをキャッチし、continue ステートメントによって処理をスキップします。モデルのフィットができない場合は、無視して次のパラメータの組み合わせを試行します。
# 最後に、最良のモデルのパラメータとAIC値が表示されます。

# このコードは、異なるパラメータの組み合わせを試行し、AIC値が最小のモデルを見つけるために使用されます。AIC値が小さいモデルは、データに最も適合していると考えられ、良い予測性能を持つ可能性が高いです。
