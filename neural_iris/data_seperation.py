# seperate data for training, validation and testing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from numpy import savetxt
op_cases= 3 
iris_df = pd.read_csv("Iris.csv",index_col='Id')

def string_to_num(x):
    if x=='Iris-setosa':
        return 1
    if x=='Iris-versicolor':
        return 2
    if x=='Iris-virginica':
        return 3

iris_df['Species']=iris_df['Species'].apply(string_to_num) 
data=iris_df.to_numpy(dtype='float')
X=data[:,0:4]

y=np.zeros((X.shape[0],op_cases))
for i in range(X.shape[0]):
    if data[i,4:5]==1:
        y[i,:]= np.identity(3)[0]
    elif data[i,4:5]==2:
        y[i,:]= np.identity(3)[1]        
    elif data[i,4:5]==3:
        y[i,:]= np.identity(3)[2]
      
        
X = normalize(X)

one=np.ones((X.shape[0],1))
X= np.append(one,X,axis=1)

initial_theta=np.ones((X.shape[1],y.shape[1]))

# make train, val and test sets
# seperate out test set
x_1,x_test,y_1,y_test=train_test_split(X,y,test_size=0.2)
#split remining into train and val set
x_train,x_val,y_train,y_val=train_test_split(x_1,y_1,test_size=.01)

Train = np.append(x_train,y_train,axis=1) 
Val = np.append(x_val,y_val,axis=1)
Test = np.append(x_test,y_test,axis=1)




columns = "X0,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Sp1,Sp2,Sp3"
savetxt('TrainData.csv',Train,delimiter=',',header=columns)
columns = "X0,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Sp1,Sp2,Sp3"
savetxt('ValData.csv',Val,delimiter=',',header=columns)
columns = "X0,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Sp1,Sp2,Sp3"
savetxt('TestData.csv',Test,delimiter=',',header=columns)
