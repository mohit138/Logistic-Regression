# data analysis for WINE data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import savetxt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 9

op_cases= 7 
#iris_df = pd.read_csv("Iris.csv",index_col='Id')
train_df = pd.read_csv("TrainData.csv")
data_t = train_df.to_numpy()
x_train = data_t[:,0:13]
y_train = data_t[:,13:20]

val_df = pd.read_csv("ValData.csv")
data_v = val_df.to_numpy()
x_val = data_v[:,0:13]
y_val = data_v[:,13:20]

test_df = pd.read_csv("TestData.csv")
data_te = test_df.to_numpy()
x_test = data_te[:,0:13]
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

def learning_curve(xT,xV,yT,yV,theta,lam):
    ma=yV.shape[0]
    J_t_error = np.zeros((ma-1,1))
    J_v_error = np.zeros((ma-1,1))
    for i in range(2,ma+1):
        [J_t_error[i-2],h] = costFunction(xT[0:i-1,:], yT[0:i-1,:], theta, lam)
        [J_v_error[i-2],h] = costFunction(xV[0:i-1,:], yV[0:i-1,:], theta, lam)
    i=np.arange(1,ma,1)
    #print(J_t_error,J_v_error)
    plt.plot(i,J_t_error)
    plt.plot(i,J_v_error)
    plt.show()



# the function to find degree for best results. 
def lc_degree(d_max,xT,xV,yT,yV,lam,max_itr,alpha,col):
    power=1;
    ma=yV.shape[0]
    J_t_error = np.zeros((d_max,1))
    J_v_error = np.zeros((d_max,1))
    for i in range(d_max):
        initial_theta = np.ones((xT.shape[1],yT.shape[1]))
        [theta,j_hist] = grad_descent_vec(xT, yT, initial_theta,lam, max_itr, alpha)    
        [J_t_error[i],h] = costFunction(xT[0:ma,:], yT[0:ma,:], theta, lam)
        [J_v_error[i],h] = costFunction(xV, yV, theta, lam)
        fact = normalise_fact( np.append(np.power(xT,i+1) , np.power(xV,i+1),axis=0) )
        #fact=1
        #powT = normalise_fact(np.power(xT,i+1))
        #powV = normalise_fact(np.power(xV,i+1))
        #xT = np.append(xT, powT,axis=1)
        #xV = np.append(xV, powV,axis=1)
        xT = np.power(xT,i+1) / fact
        xV = np.power(xV,i+1) / fact
        #print(j_hist)
    power = np.argmin(J_v_error)
    i=np.arange(1,d_max+1,1)
    plt.plot(i,J_t_error,label='Train Error')
    plt.plot(i,J_v_error,label='Validation Error')
    plt.legend(loc='upper left')
    plt.title(col)
    plt.show()
    return power
    

def lc_lambda(limit,xT,xV,yT,yV,lam,max_itr,alpha):
    ma=yV.shape[0]
    J_t_error = np.zeros((limit,1))
    J_v_error = np.zeros((limit,1))
    for i in range(limit):
        lam = .01*pow(2,i)
        initial_theta = np.ones((xT.shape[1],yT.shape[1]))
        [theta,j_hist] = grad_descent_vec(xT, yT, initial_theta,lam, max_itr, alpha)    
        [J_t_error[i],h] = costFunction(xT[0:ma,:], yT[0:ma,:], theta, lam)
        [J_v_error[i],h] = costFunction(xV, yV, theta, lam)
    i=np.arange(1,limit+1,1)
    plt.plot(i,J_t_error,label='Train Error')
    plt.plot(i,J_v_error,label='Validation Error')
    plt.legend(loc='upper left')
    plt.show()

def normalise_fact(x):
    Max = np.max(x)
    Min = np.min(x)
    fact = (Max-Min)
    return fact


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


power = [1, 1, 0, 5, 5, 5, 0, 3, 0, 0, 1]



initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
#[J,h]=costFunction(x_train, y_train, initial_theta,0)
max_itr = 200
alpha = .01
lam=0
columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
#[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,0, max_itr, alpha)

i=np.arange(1,201,1)

#learning_curve(x_train, x_val, y_train, y_val, theta, lam)


#d_max=13
#print(x_train[:,2:3])
#for i in range(1,12):
#    print(x_train[:,i+1:i+2])
#    power[i-1]=lc_degree(d_max, x_train[:,i+1:i+2], x_val[:,i+1:i+2], y_train, y_val, lam, max_itr, alpha,columns[i-1])
    

#power = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
for i in range(2,13):
     if power[i-2] != 0:
         for j in range(2,power[i-2] +2):
             print(j);
             fact = normalise_fact( np.append( np.append( np.power(x_train[:,i:i+1],j) , np.power(x_val[:,i:i+1],j), axis=0) , np.power(x_test[:,i:i+1],j) ,axis=0) )
             #fact=1
             #print(np.append( np.append( np.power(x_train[:,i:i+1],j) , np.power(x_val[:,i:i+1],j), axis=0) , np.power(x_test[:,i:i+1],j) ,axis=0).shape)
             x_train = np.append(x_train, (np.power(x_train[:,i:i+1],j) / fact),axis=1)
             x_val = np.append(x_val, (np.power(x_val[:,i:i+1],j) / fact),axis=1)
             x_test = np.append(x_test, (np.power(x_test[:,i:i+1],j) / fact),axis=1)

initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,0, max_itr, alpha)

i=np.arange(1,201,1)
plt.plot(i,j_hist)
plt.show()

learning_curve(x_train, x_val, y_train, y_val, theta, lam)

[J,h_train]=costFunction(x_train, y_train, theta,0)

h_train,acc=prediction(h_train,y_train)
print("\naccuracy (train):  ",acc)

[J,h_test]=costFunction(x_test, y_test, theta,0)

h_test,acc=prediction(h_test,y_test)
print("\naccuracy:  ",acc)


result = np.append(h_test,np.ones((h_test.shape[0],1)),axis=1)
result = np.append( result ,y_test,axis=1)

savetxt('Result.csv',result,delimiter=',')

#limit = 11
#obtain optimal lambda
#lc_lambda(limit, x_train, x_val, y_train, y_val, lam, max_itr, alpha)

