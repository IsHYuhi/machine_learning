#githubからダウンロードしたコード
#GDPと暮らしへの満足度データを統合して一つのPandasデータフレームにまとめる。
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model #linear_modelとつける

#データのロード
oecd_bli = pd.read_csv("./lifesat/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("./lifesat/gdp_per_capita.csv", thousands=',',delimiter='\t', encoding='latin1', na_values="n/a")
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#データの可視化
country_stats.plot(kind='scatter', x = 'GDP per capita', y='Life satisfaction')
plt.show()

#線形モデルの選択
model = sklearn.linear_model.LinearRegression()

#モデルの訓練
model.fit(X, y)

#キプロスの例から予測を行う
X_new = [[22587]]  #キプロス一人当たりGDP
print(model.predict(X_new))#出力[[5.96242338]]

