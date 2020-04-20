# lets visualize data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt

wine_df = pd.read_csv("wineQualityImputed.csv")

symbols=['bx','r.','gv','cv','g.','rx','b.']

wine_df.rename(columns={
    'fixed acidity':'1',
    'volatile acidity':'2',
    'citric acid':'3',
    'residual sugar':'4',
    'chlorides':'5',
    'free sulfur dioxide':'6',
    'total sulfur dioxide':'7',
    'density':'8',
    'pH':'9',
    'sulphates':'10',
    'alcohol':'11',
    },inplace=True)
for j in range(1,11):
    for i in range(3,10):
        df=wine_df[wine_df['quality']==i]
        plt.plot(df['{}'.format(j)],df['{}'.format(j+1)],'{}'.format(symbols[i-3]))

    plt.show()

