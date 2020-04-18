# prediction of accuracy of trained model
import pandas as pd
import numpy as np




theta_df=pd.read_csv("theta.csv")
op_cases=7
wine_df = pd.read_csv("wineQualityImputed.csv")

def prediction(h,y):
    h_out=np.zeros((h.shape[0],h.shape[1]))
    add=0
    for i in range(h.shape[0]):
        result = np.where(h[i:i+1,:] == np.amax(h[i:i+1,:],axis=1))
        list_of_cordinates= list(zip(result[0],result[1]))
        print(list_of_cordinates)
        cord = list_of_cordinates[0][1]
        h_out[i:i+1,cord]=1
        print("\n {}   {}  ".format(h[i:i+1,:] , y[i:i+1,:]))
        if np.all(h_out[i:i+1,:]==y[i:i+1,:]):
            add=add+1

    accuracy = add / h.shape[0] *100
    return [h_out,accuracy]


theta = theta_df.to_numpy()
#print(theta.shape)
data = wine_df.sample(frac=.05)
sample_data=data.to_numpy()
X_s=sample_data[:,1:12]
one=np.ones((X_s.shape[0],1))
X_s=np.append(one,X_s,axis=1)
#print(X_s.shape)
y_s=sample_data[:,12:13]
#print(y_s.shape)
prod =-X_s.dot(theta)
prod=prod.astype(float)
h_s = 1/(1+np.exp(prod))
#print(h_s)
h_s,acc=prediction(h_s,y_s)

print("\n {} \n {}".format(h_s,acc))
