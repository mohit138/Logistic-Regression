# data analysis for IRIS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 9

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

'''
# the function to find degree for best results. 
def lc_degree(d_max,xT,xV,yT,yV,lam,max_itr,alpha):
    
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
    plt.show()
'''

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

'''
def string_to_num(x):
    if x=='Iris-setosa':
        return 1
    if x=='Iris-versicolor':
        return 2
    if x=='Iris-virginica':
        return 3

iris_df['Species']=iris_df['Species'].apply(string_to_num) 
data=iris_df.to_numpy(dtype='float')
X=data[:,1:4]
y=np.zeros((X.shape[0],op_cases))
for i in range(X.shape[0]):
    if data[i,4:5]==1:
        y[i,:]= np.identity(3)[0]
    elif data[i,4:5]==2:
        y[i,:]= np.identity(3)[1]        
    elif data[i,4:5]==3:
        y[i,:]= np.identity(3)[2]
      
        

one=np.ones((X.shape[0],1))
X= np.append(one,X,axis=1)
initial_theta=np.ones((X.shape[1],y.shape[1]))

# make train, val and test sets
# seperate out test set
x_1,x_test,y_1,y_test=train_test_split(X,y,test_size=0.2)
#split remining into train and val set
x_train,x_val,y_train,y_val=train_test_split(x_1,y_1,test_size=0.2)
'''

initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
[J,h]=costFunction(x_train, y_train, initial_theta,0)
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

power = [5,8,5,8]
# Adding polynomial terms to the first 8 features 
for i in range(1,5):
     if power[i-1] != 0:
         for j in range(2,power[i-1]+1):
             #print(j);
             fact = normalise_fact( np.append( np.append( np.power(x_train[:,i:i+1],j) , np.power(x_val[:,i:i+1],j), axis=0) , np.power(x_test[:,i:i+1],j) ,axis=0) )
             #fact=1
             #print(np.append( np.append( np.power(x_train[:,i:i+1],j) , np.power(x_val[:,i:i+1],j), axis=0) , np.power(x_test[:,i:i+1],j) ,axis=0).shape)
             x_train = np.append(x_train, (np.power(x_train[:,i:i+1],j) / fact),axis=1)
             x_val = np.append(x_val, (np.power(x_val[:,i:i+1],j) / fact),axis=1)
             x_test = np.append(x_test, (np.power(x_test[:,i:i+1],j) / fact),axis=1)


initial_theta=np.ones((x_train.shape[1],y_train.shape[1]))
[theta,j_hist] = grad_descent_vec(x_train, y_train, initial_theta,lam, max_itr, alpha)

i=np.arange(1,151,1)

learning_curve(x_train, x_val, y_train, y_val, theta, 0)


col = ["sl","sw","pl","pw"]
# find the optimal degree for each function for sepal length feature.
d_max=13

#print(x_train)
#for i in range(4):
#    lc_degree(d_max, x_train[:,i+1:i+2], x_val[:,i+1:i+2], y_train, y_val, lam, max_itr, alpha,col[i])


limit = 11
# obtain optimal lambda
lc_lambda(limit, x_train, x_val, y_train, y_val, lam, max_itr, alpha)


[J,h_train]=costFunction(x_train, y_train, theta,0)

h_train,acc=prediction(h_train,y_train)
print("\naccuracy (train):  ",acc)

[J,h_test]=costFunction(x_test, y_test, theta,0)

h_test,acc=prediction(h_test,y_test)
print("\naccuracy:  ",acc)


for i in range(y_test.shape[0]):
  print( "{}      {} " .format(h_test[i] , y_test[i]))

# THEREFORE AFTER THE ANALYSIS, WE FOUND
# WE NEED TO USE, DEGREE 3 FOR SEPAL LENGTH AND SEPAL WIDTH.
# FOR THE VALUE OF LAMBDA, WE FIND THE VALUE 0 TO BE THE BEST