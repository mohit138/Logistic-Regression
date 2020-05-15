# data analysis for IRIS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from pylab import rcParams
from scipy.special import expit
rcParams['figure.figsize'] = 15, 9

op_cases= 3 
#iris_df = pd.read_csv("Iris.csv",index_col='Id')
train_df = pd.read_csv("TrainData.csv")
data_t = train_df.to_numpy()
x_train = data_t[:,1:5]
y_train = data_t[:,5:8]

val_df = pd.read_csv("ValData.csv")
data_v = val_df.to_numpy()
x_val = data_v[:,1:5]
y_val = data_v[:,5:8]

test_df = pd.read_csv("TestData.csv")
data_te = test_df.to_numpy()
x_test = data_te[:,1:5]
y_test = data_te[:,5:8]

#pd.plotting.scatter_matrix(iris_df)
#plt.show()

def costFunction(X,y,theta,lam):
    m=X.shape[0]
    prod=np.zeros((X.shape[0],theta.shape[1]))
    
    prod=X.dot(theta)
    h=1/(1+np.exp(-prod))
    
    J = (-1/m)*np.sum( y*np.log(h + 1e-20) + (1-y)*np.log(1-h + 1e-20)) + (lam/(2*m))*((np.sum(theta))-np.sum(theta[0,:]));
    return [J,h]


def grad_descent_vec(X,y,theta,lam,max_itr,alpha):
    m=X.shape[0]
    J_hist=np.zeros((max_itr,1))
    for i in range(max_itr):
        [J,h]=costFunction(X,y,theta,lam)
        dif = h-y

        tho = lam/m*theta[:,0:1]
        theta = theta - (alpha/m)*((X.T).dot(dif) + lam/m*theta) 
        theta[:,0:1] = theta[:,0:1] - tho
        J_hist[i]=J
    return [theta,J_hist]




# the function to find degree for best results. 
def lc_degree(d_max,xT,xV,yT,yV,lam,max_itr,alpha,col):
    power=1
    alpha =.01
    max_itr = 1000
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
        powT = (np.power(xT,i+1))/fact
        powV = (np.power(xV,i+1))/fact
        xT = np.append(xT, powT,axis=1)
        xV = np.append(xV, powV,axis=1)
        j=np.arange(1,1001,1)
        #plt.plot(j,j_hist)
        #plt.show()
        #xT = np.power(xT,i+1) / fact
        #xV = np.power(xV,i+1) / fact
        #print(j_hist[980:1000])
    power = np.argmin(J_v_error)
    i=np.arange(1,d_max+1,1)
    plt.plot(i,J_t_error,label='Train Error')
    plt.plot(i,J_v_error,label='Validation Error')
    plt.legend(loc='upper left')
    plt.title(col)
    plt.show()
    return power

def normalise_fact(x):
    Max = np.max(x)
    Min = np.min(x)
    fact = (Max-Min)
    return fact


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

def learning_curve(hlSize,numLabel,xT,xV,yT,yV,theta,lam=0):
    ma=yV.shape[0]
    J_t_error = np.zeros((ma-1,1))
    J_v_error = np.zeros((ma-1,1))
    for i in range(2,ma+1):
        [J_t_error[i-2],grad,h] = cost_function_NN1(theta,hlSize,numLabel,xT[0:i-1,:], yT[0:i-1,:], lam)
        [J_v_error[i-2],grad,h] = cost_function_NN1(theta,hlSize,numLabel,xV[0:i-1,:], yV[0:i-1,:],  lam)
    i=np.arange(1,ma,1)
    #print(J_t_error,J_v_error)
    plt.plot(i,J_t_error,label='Train Error')
    plt.plot(i,J_v_error,label='Validation Error')
    plt.legend(loc='upper left')
    plt.show()
    
    
def sigmoidGrad(z):
    return expit(z)*(1-expit(z))

# cost function for 1 hidden layer Neural Network
def cost_function_NN1(theta,hlSize,numLabel,X,y,lam=0):
    m= X.shape[0]
    ipSize = X.shape[1]
    J=0

    t1 = theta[0:hlSize*(ipSize +1)].reshape(hlSize,ipSize+1)
    t2 = theta[hlSize*(ipSize +1) : (hlSize*(ipSize +1)) + (numLabel*(hlSize +1)) +1].reshape(numLabel,hlSize+1)

    tGrad1 = np.zeros(t1.shape)
    tGrad2 = np.zeros(t2.shape)
    #print(X.shape,t1.shape)
    # Forward propogation algorithm
    z2 = np.append(np.ones((m,1)),X,axis =1).dot(t1.T)
    a2 = expit(z2)
    z3 = np.append(np.ones((m,1)),a2,axis=1).dot(t2.T)
    a3 = expit(z3)
    h = a3
    
    J = (-1/m)*np.sum( y*np.log(h + 1e-20) + (1-y)*np.log(1-h + 1e-20)) 
    #Reg = 
    
    # Back propogation algorithm
    D2=np.zeros(t2.shape)
    D1=np.zeros(t1.shape)
    err3 = h - y
    l = t2.shape[1]
    err2 = (err3.dot(t2[:,1:l])) * sigmoidGrad(z2) 
    for i in range(m):
        D2 = D2 + (err3[i:i+1,:].T).dot(np.append(np.ones((1,1)),a2[i:i+1,:],axis=1))
        D1 = D1 + (err2[i:i+1,:].T).dot(np.append(np.ones((1,1)),X[i:i+1,:],axis=1))
    tGrad1 = D1/m
    #tGrad1[:,1:tGrad1.shape[1]] = tGrad1[:,1:tGrad1.shape[1]] + lam/m*t1[:,1:t1.shape[1]]
    tGrad2 = D2/m
    #tGrad2[:,1:tGrad2.shape[1]] = tGrad2[:,1:tGrad2.shape[1]] + lam/m*t2[:,1:t2.shape[1]]

    #print(tGrad1, tGrad2)
    grad = np.concatenate([tGrad1.flat, tGrad2.flat])
    
    #print(grad.shape)
    return [J,grad,h]

def gradient_descent_NN(X,y,theta,lam,max_itr,alpha,hlSize): # without regularisation !
    m=X.shape[0]
    J_hist=np.zeros((max_itr,1))
    
    numLabel = y.shape[1]
    for i in range(max_itr):
        [J,grad,h]=cost_function_NN1(theta, hlSize, numLabel, X, y, lam)
        
        theta = theta - (alpha/m)*(grad) 
        J_hist[i]=J
    return [theta,J_hist]

def gradient_check(theta,hlSize,numLabel,X,y):
    #m= X.shape[0]
    #ipSize = X.shape[1]
    appGrad = np.zeros(theta.shape)

    #t1 = theta[0:hlSize*(ipSize +1)].reshape(hlSize,ipSize+1)
    #t2 = theta[hlSize*(ipSize +1) : (hlSize*(ipSize +1)) + (numLabel*(hlSize +1))+1].reshape(numLabel,hlSize+1)

    #tGrad1 = np.zeros(t1.shape)
    #tGrad2 = np.zeros(t2.shape)
    epsilon =.0001
    
    for i in range(theta.shape[0]):
        posTheta = theta
        posTheta[i:i+1] = posTheta[i:i+1] + epsilon
        posJ,gr,h = cost_function_NN1(posTheta, hlSize, numLabel, X, y)
        negTheta = theta
        negTheta[i:i+1] = negTheta[i:i+1] - epsilon
        negJ,gr,h = cost_function_NN1(negTheta, hlSize, numLabel, X, y)
        appGrad[i]=((posJ-negJ)/(2*epsilon))
    return appGrad

def prediction(h,y):
    
    count =0
    h_out = np.zeros(h.shape)
    
    for i in range(h.shape[0]):
        if h.shape[1] == 1:
            if h[i] >= 0.5:
                h_out[i] = 1
        else:
            h_out[i, np.argmax(h[i,:])] =1
    
    
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if h_out[i,j]!=y[i,j]:
                count = count+1
                break
    acc = (h.shape[0] - count)/h.shape[0] *100
    return [h_out, acc]


def cost_function_NN1_(t1,t2,hlSize,numLabel,X,y,lam=0):
    m= X.shape[0]
    ipSize = X.shape[1]
    J=0

    #t1 = theta[0:hlSize*(ipSize +1)].reshape(hlSize,ipSize+1)
    #t2 = theta[hlSize*(ipSize +1) : (hlSize*(ipSize +1)) + (numLabel*(hlSize +1)) +1].reshape(numLabel,hlSize+1)

    tGrad1 = np.zeros(t1.shape)
    tGrad2 = np.zeros(t2.shape)
    # Forward propogation algorithm
    z2 = np.append(np.ones((m,1)),X,axis =1).dot(t1.T)
    a2 = expit(z2)
    z3 = np.append(np.ones((m,1)),a2,axis=1).dot(t2.T)
    a3 = expit(z3)
    h = a3
    
    J = (-1/m)*np.sum( y*np.log(h + 1e-20) + (1-y)*np.log(1-h + 1e-20)) 
    #Reg = 
    
    # Back propogation algorithm
    D2=np.zeros(t2.shape)
    D1=np.zeros(t1.shape)
    
    err3 = h - y
    l = t2.shape[1]
    err2 = (err3.dot(t2[:,1:l])) * sigmoidGrad(z2) 
    for i in range(m):
        D2 = D2 + (err3[i:i+1,:].T).dot(np.append(np.ones((1,1)),a2[i:i+1,:],axis=1))
        D1 = D1 + (err2[i:i+1,:].T).dot(np.append(np.ones((1,1)),X[i:i+1,:],axis=1))
    
    tGrad1 = D1/m
    #tGrad1[:,1:tGrad1.shape[1]] = tGrad1[:,1:tGrad1.shape[1]] + lam/m*t1[:,1:t1.shape[1]]
    tGrad2 = D2/m
    #tGrad2[:,1:tGrad2.shape[1]] = tGrad2[:,1:tGrad2.shape[1]] + lam/m*t2[:,1:t2.shape[1]]

    #print(tGrad1, tGrad2)
    #grad = np.concatenate([tGrad1.flat, tGrad2.flat])
    
    #print(grad.shape)
    return [J,tGrad1,tGrad2,h]

def gradient_descent_NN_(X,y,t1,t2,lam,max_itr,alpha,hlSize): # without regularisation !
    m=X.shape[0]
    J_hist=np.zeros((max_itr,1))
    
    numLabel = y.shape[1]
    for i in range(max_itr):
        [J,g1,g2,h]=cost_function_NN1_(t1,t2, hlSize, numLabel, X, y, lam)
        
        #t1[:,1:5] = t1[:,1:5] - (alpha/m)*(g1[:,1:5]) 
        #t2[:,1:13] = t2[:,1:13] - (alpha/m)*(g2[:,1:13])
        t1 =t1 - (alpha)*g1
        t2 =t2 - (alpha)*g2
        
        J_hist[i]=J
    return [t1,t2,J_hist]

def gradient_check_(t1,t2,hlSize,numLabel,X,y):
    #m= X.shape[0]
    #ipSize = X.shape[1]
    appGrad1 = np.zeros(t1.shape)
    appGrad2 = np.zeros(t2.shape)
    #t1 = theta[0:hlSize*(ipSize +1)].reshape(hlSize,ipSize+1)
    #t2 = theta[hlSize*(ipSize +1) : (hlSize*(ipSize +1)) + (numLabel*(hlSize +1))+1].reshape(numLabel,hlSize+1)

    #tGrad1 = np.zeros(t1.shape)
    #tGrad2 = np.zeros(t2.shape)
    epsilon =.0001
    
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]):
            posTheta = t1
            posTheta[i:i+1,j:j+1] = posTheta[i:i+1,j:j+1] + epsilon
            posJ,gr1,gr2,h = cost_function_NN1_(posTheta,t2, hlSize, numLabel, X, y)
            negTheta = t1
            negTheta[i:i+1,j:j+1] = negTheta[i:i+1,j:j+1] - epsilon
            negJ,gr1,gr2,h = cost_function_NN1_(negTheta,t2, hlSize, numLabel, X, y)
            appGrad1[i:i+1,j:j+1]=((posJ-negJ)/(2*epsilon))
    for i in range(t2.shape[0]):
        for j in range(t2.shape[1]):
            posTheta = t2
            posTheta[i:i+1,j:j+1] = posTheta[i:i+1,j:j+1] + epsilon
            posJ,gr1,gr2,h = cost_function_NN1_(t1,posTheta, hlSize, numLabel, X, y)
            negTheta = t2
            negTheta[i:i+1,j:j+1] = negTheta[i:i+1,j:j+1] - epsilon
            negJ,gr1,gr2,h = cost_function_NN1_(t1,negTheta, hlSize, numLabel, X, y)
            appGrad2[i:i+1,j:j+1]=((posJ-negJ)/(2*epsilon))
    
    return [appGrad1,appGrad2]


ipSize = x_train.shape[1]
hlSize = 5#1* x_train.shape[1]
numLabel = y_train.shape[1]
E0= 0.01
theta1 = np.random.randint(1,10,(hlSize,ipSize+1))
itheta1 = theta1*2*E0 - E0
theta2 = np.random.randint(1,10,(numLabel,hlSize+1))
itheta2 = theta2*2*E0 - E0
initial_theta = np.concatenate([theta1.flat, theta2.flat])

lam =0
alpha =1
max_itr = 1000

[J,grad1,grad2,h_train]=cost_function_NN1_(itheta1,itheta2, hlSize, numLabel, x_train, y_train)
print(J.shape, grad1.shape,grad2.shape,h_train.shape)


# checking if back propogation is working
J , grad1,grad2,h = cost_function_NN1_(itheta1,itheta2, hlSize, numLabel, x_train, y_train)
approxGrad1,approxGrad2 = gradient_check_(itheta1,itheta2, hlSize, numLabel, x_train, y_train)

for i in range(grad1.shape[0]):
    for j in range(grad1.shape[1]):
        print( "{}      {} " .format(grad1[i:i+1,j:j+1] , approxGrad1[i:i+1,j:j+1]))
        
for i in range(grad2.shape[0]):
    for j in range(grad2.shape[1]):
        print( "{}      {} " .format(grad2[i:i+1,j:j+1] , approxGrad2[i:i+1,j:j+1]))

print(itheta1)
theta1,theta2,J_hist = gradient_descent_NN_(x_train, y_train, itheta1,itheta2, lam, max_itr, alpha, hlSize)
print(theta1)
i=np.arange(1,1001,1) 
plt.plot(i,J_hist)
plt.show()   



#learning_curve(hlSize,numLabel,x_train, x_val, y_train, y_val, theta, lam)

[J,g1,g2,h_train]=cost_function_NN1_(theta1,theta2, hlSize, numLabel, x_train, y_train)
h_train,acc=prediction(h_train,y_train)
print("\naccuracy (train):  ",acc)

[J,g1,g2,h_test]=cost_function_NN1_(theta1,theta2, hlSize, numLabel, x_test, y_test)
print(h_test)
h_test,acc=prediction(h_test,y_test)

print("\naccuracy:  ",acc)    

    
for i in range(y_test.shape[0]):
    print( "{}      {} " .format(h_test[i] , y_test[i]))
    
 
    