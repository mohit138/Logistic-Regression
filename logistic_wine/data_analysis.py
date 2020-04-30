# data analysis for WINE data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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
    
    ma=yV.shape[0]
    J_t_error = np.zeros((d_max,1))
    J_v_error = np.zeros((d_max,1))
    for i in range(d_max):
        initial_theta = np.ones((xT.shape[1],yT.shape[1]))
        [theta,j_hist] = grad_descent_vec(xT, yT, initial_theta,lam, max_itr, alpha)    
        [J_t_error[i],h] = costFunction(xT[0:ma,:], yT[0:ma,:], theta, lam)
        [J_v_error[i],h] = costFunction(xV, yV, theta, lam)
        powT = normalise(np.power(xT,i+1))
        powV = normalise(np.power(xV,i+1))
        #xT = np.append(xT, powT,axis=1)
        #xV = np.append(xV, powV,axis=1)
        xT = powT
        xV = powV
        #print(j_hist)
    i=np.arange(1,d_max+1,1)
    plt.plot(i,J_t_error,label='Train Error')
    plt.plot(i,J_v_error,label='Validation Error')
    plt.legend(loc='upper left')
    plt.title(col)
    plt.show()
    

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

def normalise(x):
    Max = np.max(x)
    Min = np.min(x)
    x = x / (Max-Min)
    return x

initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
[J,h]=costFunction(x_train, y_train, initial_theta,0)
max_itr = 150
alpha = .001
lam=0
columns = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,0, max_itr, alpha)

i=np.arange(1,151,1)

#learning_curve(x_train, x_val, y_train, y_val, theta, lam)


d_max=13
print(x_train[:,2:3])
for i in range(1,12):
    print(x_train[:,i+1:i+2])
    lc_degree(d_max, x_train[:,i+1:i+2], x_val[:,i+1:i+2], y_train, y_val, lam, max_itr, alpha,columns[i-1])
    

#limit = 11
# obtain optimal lambda
#lc_lambda(limit, x_train, x_val, y_train, y_val, lam, max_itr, alpha)

