#logistic regression for iris data
# data analysis for IRIS
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
op_cases= 3 
iris_df = pd.read_csv("Iris.csv",index_col='Id')

def iris_type1(x):
    if x=='Iris-setosa':
        return 1
    else:
        return 0
def iris_type2(x):
    if x=='Iris-versicolor':
        return 1
    else:
        return 0
def iris_type3(x):
    if x=='Iris-virginica':
        return 1
    else:
        return 0

def plot(data):
    type1=data[data['type']==0]
    plt.plot(type1['A'],type1['B'],'rx')
    type1=data[data['type']==1]
    plt.plot(type1['A'],type1['B'],'bo')
    type1=data[data['type']==2]
    plt.plot(type1['A'],type1['B'],'g*')
    plt.show()
    return

def costFunction(X,y,theta):
    m=X.shape[0]
    n=theta.shape[0]
    #print(m)
    #print(n)
    prod=X.dot(theta)
    h=1/(1+np.exp(-prod))
    #print("\n",prod)
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
    
#print("\n\n",iris_df)

#print("\n\n",iris_df.shape)
iris_df['type1']=iris_df['Species'].apply(iris_type1)
iris_df['type2']=iris_df['Species'].apply(iris_type2)
iris_df['type3']=iris_df['Species'].apply(iris_type3)
print(iris_df['type2'])


data=iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','type1','type2','type3']]
data.rename(columns={
    'PetalWidthCm':'A',
    'PetalLengthCm':'B'
    },inplace=True)
print(data)

#plot(data)

Data=data.to_numpy()
#print(Data)
X = Data[:,:4]
#X=(np.reshape(X,(-1,150))).T
#print(X)
one=np.ones((150,1))
X=np.append(one,X,axis=1)
print(X)

y=Data[:,4:7]
#print(y)
#y=(np.reshape(y,(-1,150))).T
print(y)
initial_theta=np.zeros((X.shape[1],1))
print(y[:,1:2])
#print(initial_theta)

#print(1/(1+np.exp(-X.dot(initial_theta))))


[J,h]=costFunction(X,y,initial_theta)
#print(J,h)
alpha=.1
max_itr=20000
theta = np.zeros((X.shape[1],op_cases))
J_hist = np.zeros((max_itr , op_cases))
for i in range(op_cases):
    [theta[:,i:i+1],J_hist[:,i:i+1]]=grad_descent(X,y[:,i:i+1],initial_theta,max_itr,alpha)

print(theta)
print(J_hist)
i=np.arange(1,20001,1)
#plt.plot(i,J_hist)
#plt.show()

sample_data = data.sample(frac=.3)
print(sample_data)
Sample_data=sample_data.to_numpy()
X_s = Sample_data[:,:4]
one=np.ones((X_s.shape[0],1))
X_s=np.append(one,X_s,axis=1)
#print(X_s)
y_s = Sample_data[:,4:7]
#print(y_s)
h_s = 1/(1+np.exp(-X_s.dot(theta)))
h_s,acc=prediction(h_s,y_s)
print(h_s,"\naccuracy:\n",acc)








