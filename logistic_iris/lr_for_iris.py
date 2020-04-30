#LR for IRIS data set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
op_cases= 3 
#iris_df = pd.read_csv("Iris.csv",index_col='Id')
train_df = pd.read_csv("TrainData.csv")
data_t = train_df.to_numpy()
x_train = data_t[:,0:5]
y_train = data_t[:,5:8]

val_df = pd.read_csv("ValData.csv")
data_v = val_df.to_numpy()
x_val = data_v[:,0:5]
y_val = data_v[:,5:8]

test_df = pd.read_csv("TestData.csv")
data_te = test_df.to_numpy()
x_test = data_te[:,0:5]
y_test = data_te[:,5:8]

def costFunction(X,y,theta,lam):
    m=X.shape[0]
    prod=np.zeros((X.shape[0],theta.shape[1]))
    
    prod=X.dot(theta)
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

def normalise(x):
    Max = np.max(x)
    Min = np.min(x)
    x = x / (Max-Min)
    return x

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



initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
max_itr = 150
alpha = .1
lam=0


# using degree 3 for sepal data
# seplal width^3
#x_train[:,0:1] = normalise(np.power(x_train[:,2:3],3))
#x_val[:,0:1] = normalise(np.power(x_val[:,2:3],3))
#x_test[:,0:1] = normalise(np.power(x_test[:,2:3],3))

# seplal length^3
#x_train[:,1:2] = normalise(np.power(x_train[:,1:2],3))
#x_val[:,1:2] = normalise(np.power(x_val[:,1:2],3))
#x_test[:,1:2] = normalise(np.power(x_test[:,1:2],3))


[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,0, max_itr, alpha)

#[J,h_train]=costFunction(x_train, y_train, theta,0)

#h_train,acc=prediction(h_train,y_train)
#print(h_train,y_train,"\naccuracy:\n",acc)
[J,h_test]=costFunction(x_test, y_test, theta,0)

h_test,acc=prediction(h_test,y_test)
print(h_test,y_test,"\naccuracy:\n",acc)


