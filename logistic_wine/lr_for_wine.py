# logistic regression for wine data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import savetxt

wine_df = pd.read_csv("wineQualityImputed.csv")
op_cases = 7

def costFunction(X,y,theta):
    m=X.shape[0]
    n=theta.shape[0]
    #print(m)
    #print(n)
    prod=X.dot(theta)
    #print(prod)
    prod=prod.astype(float)
    h=1/(1+np.exp(-prod))
    #print("\n",h)
    J = (-1/m)*np.sum( y*np.log(h) + (1-y)*np.log(1-h));
    return [J,h]

def grad_descent(X,y,theta,max_itr,alpha):
    m=X.shape[0]
    n=theta.shape[0]
    J_hist=np.zeros((max_itr,1))
    for i in range(max_itr):
        [J,h]=costFunction(X,y,theta)
        dif = h-y
        #print("\n\n\n",dif)
        for j in range(n):
            x = X[:,j:j+1]
            #x=(np.reshape(x,(-1,150))).T
            theta[j]=theta[j] - (alpha/m)*np.sum(dif*x)
        J_hist[i]=J
    return [theta,J_hist]

def prediction(h,y):
    h_out=np.zeros((h.shape[0],h.shape[1]))
    add=0
    for i in range(h.shape[0]):
        result = np.where(h[i:i+1,:] == np.amax(h[i:i+1,:],axis=1))
        list_of_cordinates= list(zip(result[0],result[1]))
        print(list_of_cordinates)
        cord = list_of_cordinates[0][1]
        h_out[i:i+1,cord]=1
        if np.all(h_out[i:i+1,:]==y[i:i+1,:]):
            add=add+1

    accuracy = add / h.shape[0] *100
    return [h_out,accuracy]


print(wine_df.isnull().sum())
data=wine_df.to_numpy()
X=data[:,1:12]
one=np.ones((X.shape[0],1))
X=np.append(one,X,axis=1)
print(X)
y=data[:,12:13]
initial_theta=np.zeros((X.shape[1],1))

alpha =.0001
max_itr=2000
theta = np.zeros((X.shape[1],op_cases))
J_hist = np.zeros((max_itr , op_cases))

for i in range(op_cases):
    [theta[:,i:i+1],J_hist[:,i:i+1]]=grad_descent(X,y==i+3,initial_theta,max_itr,alpha)
    #print(J_hist[:,i:i+1])
print("\n\n",theta.shape)
#print("\n\n", J_hist)

i=np.arange(1,2001,1)
plt.plot(i,J_hist)
plt.show()

columns = "3,4,5,6,7,8,9"
savetxt('theta.csv',theta,delimiter=',',header=columns)
