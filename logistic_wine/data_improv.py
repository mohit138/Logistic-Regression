# improvising dataimport pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine_df = pd.read_csv("winequalityN.csv")

def imputeMean(df,feature):
    mean=df[feature].mean()
    df[feature].fillna(mean,inplace=True)

def imputeMode(df,feature):
    mode=df[feature].mode()
    m=mode[0]
    df[feature].fillna(m,inplace=True)    



print("\nbefore impute :\n",wine_df.isnull().sum())


imputeMean(wine_df,'fixed acidity')
imputeMean(wine_df,'volatile acidity')
imputeMean(wine_df,'citric acid')
imputeMean(wine_df,'residual sugar')
imputeMean(wine_df,'chlorides')
imputeMean(wine_df,'pH')
imputeMean(wine_df,'sulphates')

print("\nold data f :\n",wine_df.isnull().sum())

wine_df.to_csv('wineQualityImputed.csv',index=False)
#print("\nnew df:\n",wine_df1.isnull().sum())
