# seperate data for training, validation and testing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import savetxt
op_cases= 7 
iris_df = pd.read_csv("wineQualityImputed.csv")

def string_to_num(x):
    if x=='white':
        return 1
    if x=='red':
        return 0

def normalise(x):
    Max = np.max(x)
    Min = np.min(x)
    x = x / (Max-Min)
    return x

iris_df['type']=iris_df['type'].apply(string_to_num) 
data=iris_df.to_numpy(dtype='float')
X=data[:,0:12]

y=np.zeros((X.shape[0],op_cases))
for i in range(X.shape[0]):
    if data[i,12:13]==3:
        y[i,:]= np.identity(op_cases)[0]
    elif data[i,12:13]==4:
        y[i,:]= np.identity(op_cases)[1]        
    elif data[i,12:13]==5:
        y[i,:]= np.identity(op_cases)[2]
    elif data[i,12:13]==6:
        y[i,:]= np.identity(op_cases)[3]
    elif data[i,12:13]==7:
        y[i,:]= np.identity(op_cases)[4]
    elif data[i,12:13]==8:
        y[i,:]= np.identity(op_cases)[5]
    elif data[i,12:13]==9:
        y[i,:]= np.identity(op_cases)[6]
      
        
print(y[0:14,:])
one=np.ones((X.shape[0],1))
X= np.append(one,X,axis=1)


#normalising every parameter.
for i in range(1,12):
    X[:,i:i+1]=normalise(X[:,i:i+1])


# make train, val and test sets
# seperate out test set
x_1,x_test,y_1,y_test=train_test_split(X,y,test_size=0.2)
#split remining into train and val set
x_train,x_val,y_train,y_val=train_test_split(x_1,y_1,test_size=0.2)

Train = np.append(x_train,y_train,axis=1) 
Val = np.append(x_val,y_val,axis=1)
Test = np.append(x_test,y_test,axis=1)




columns = "x0,type,fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,q3,q4,q5,q6,q7,q8,q9"
savetxt('TrainData.csv',Train,delimiter=',',header=columns)
columns = "x0,type,fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,q3,q4,q5,q6,q7,q8,q9"
savetxt('ValData.csv',Val,delimiter=',',header=columns)
columns = "x0,type,fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,q3,q4,q5,q6,q7,q8,q9"
savetxt('TestData.csv',Test,delimiter=',',header=columns)
