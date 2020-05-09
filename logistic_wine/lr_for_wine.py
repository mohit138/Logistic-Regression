# logistic regression for wine data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 9

#false , again after 8 !!

op_cases= 7 
#iris_df = pd.read_csv("Iris.csv",index_col='Id')
train_df = pd.read_csv("TrainData.csv")
data_t = train_df.to_numpy()
x_train=data_t[:,0:13]
#x_train = data_t[:,0:13]
y_train = data_t[:,13:20]

val_df = pd.read_csv("ValData.csv")
data_v = val_df.to_numpy()
#x_val = data_v[:,0:13]
x_val = data_v[:,0:13]
y_val = data_v[:,13:20]

test_df = pd.read_csv("TestData.csv")
data_te = test_df.to_numpy()
x_test = data_te[:,0:13]
#x_test = data_te[:,0:13]
y_test = data_te[:,13:20]

#pd.plotting.scatter_matrix(iris_df)
#plt.show()

def costFunction(X,y,theta,lam):
    m=X.shape[0]
    prod=np.zeros((X.shape[0],theta.shape[1]))
    
    prod=X.dot(theta)
    #print(prod)
    h=1/(1+np.exp(-prod))
    
    J = (-1/m)*np.sum( y*np.log(h) + (1-y)*np.log(1-h)) + (lam/(2*m))*((np.sum(theta))-np.sum(theta[0,:]));
    return [J,h]


def grad_descent_vec(X,y,theta,lam,max_itr,alpha):
    m=X.shape[0]
    J_hist=np.zeros((max_itr,1))
    for i in range(max_itr):
        [J,h]=costFunction(X,y,theta,lam)
        dif = h-y
        theta = theta - (alpha/m)*((X.T).dot(dif))
        J_hist[i]=J
    return [theta,J_hist]

def prediction(h,y):
    h_out=np.zeros((h.shape[0],h.shape[1]))
    add=0
    for i in range(h.shape[0]):
        result = np.where(h[i:i+1,:] == np.amax(h[i:i+1,:],axis=1))
        list_of_cordinates= list(zip(result[0],result[1]))
        #print(list_of_cordinates)
        cord = list_of_cordinates[0][1]
        h_out[i:i+1,cord]=1
        if np.all(h_out[i:i+1,:]==y[i:i+1,:]):
            add=add+1

    accuracy = add / h.shape[0] *100
    return [h_out,accuracy]

def normalise(x):
    Max = np.max(x)
    Min = np.min(x)
    x = x / (Max-Min)
    return x





#_______CHANGING COLUMNS TO RESPECTIVE OPTIMAL DEGREES_______________
'''
# fixed acidity^5
x_train[:,2:3] = normalise(np.power(x_train[:,2:3],5))
x_val[:,2:3] = normalise(np.power(x_val[:,2:3],5))
x_test[:,2:3] = normalise(np.power(x_test[:,2:3],5))
print("fixed acidity, pow 5")

# volatile acidity^5
x_train[:,3:4] = normalise(np.power(x_train[:,3:4],5))
x_val[:,3:4] = normalise(np.power(x_val[:,3:4],5))
x_test[:,3:4] = normalise(np.power(x_test[:,3:4],5))
print("volatile acidity, pow 5")

# citric acid^5
x_train[:,4:5] = normalise(np.power(x_train[:,4:5],5))
x_val[:,4:5] = normalise(np.power(x_val[:,4:5],5))
x_test[:,4:5] = normalise(np.power(x_test[:,4:5],5))
print("citric acid, pow 5")

# residual ^ 5
x_train[:,5:6] = normalise(np.power(x_train[:,5:6],5))
x_val[:,5:6] = normalise(np.power(x_val[:,5:6],5))
x_test[:,5:6] = normalise(np.power(x_test[:,5:6],5))
print("residual sugar, pow 5")

# chlorides ^ 4
x_train[:,6:7] = normalise(np.power(x_train[:,6:7],4))
x_val[:,6:7] = normalise(np.power(x_val[:,6:7],4))
x_test[:,6:7] = normalise(np.power(x_test[:,6:7],4))
print("chlorides, pow 5")

# free SO2 ^ 4
x_train[:,7:8] = normalise(np.power(x_train[:,7:8],4))
x_val[:,7:8] = normalise(np.power(x_val[:,7:8],4))
x_test[:,7:8] = normalise(np.power(x_test[:,7:8],4))
print("free SO2, pow 5")

# total SO2 ^ 5
x_train[:,8:9] = normalise(np.power(x_train[:,8:9],5))
x_val[:,8:9] = normalise(np.power(x_val[:,8:9],5))
x_test[:,8:9] = normalise(np.power(x_test[:,8:9],5))
print("residual sugar, pow 5")

# density ^ 3
x_train[:,9:10] = normalise(np.power(x_train[:,9:10],3))
x_val[:,9:10] = normalise(np.power(x_val[:,9:10],3))
x_test[:,9:10] = normalise(np.power(x_test[:,9:10],3))
print("density, pow 3")

# pH ^ 5
x_train[:,10:11] = normalise(np.power(x_train[:,10:11],5))
x_val[:,10:11] = normalise(np.power(x_val[:,10:11],5))
x_test[:,10:11] = normalise(np.power(x_test[:,10:11],5))
print("pH, pow 5")

# sulphates ^ 4
x_train[:,11:12] = normalise(np.power(x_train[:,11:12],4))
x_val[:,11:12] = normalise(np.power(x_val[:,11:12],4))
x_test[:,11:12] = normalise(np.power(x_test[:,11:12],4))
print("residual sugar, pow 5")
'''

#___________adding columns with columns till optimal power________


#power=[5,5,5,5,3,4,5,1,5,4]
power=[7,7,7,7,7,7,7,7,7,7]


for i in range(2,12):
    if i!=9:
        for j in range(2,power[i-2] +1):
            x_train = np.append(x_train, normalise(np.power(x_train[:,i:i+1],j)),axis=1)
            x_val = np.append(x_val, normalise(np.power(x_val[:,i:i+1],j)),axis=1)
            x_test = np.append(x_test, normalise(np.power(x_test[:,i:i+1],j)),axis=1)

initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
max_itr = 150
alpha = .01
lam=0

[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,0, max_itr, alpha)
i=np.arange(1,151,1)
plt.plot(i,j_hist)
plt.show()
#print(theta)
[J,h_train]=costFunction(x_train, y_train, theta,0)

h_train,acc=prediction(h_train,y_train)
print("\naccuracy (train):  ",acc)

[J,h_test]=costFunction(x_test, y_test, theta,0)

h_test,acc=prediction(h_test,y_test)
print("\naccuracy:  ",acc)