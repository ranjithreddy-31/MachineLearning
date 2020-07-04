import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
dataset=pd.read_csv('Social_Network_Ads.csv')
gender={'Male':1,'Female':2}
dataset.Gender=[gender[item] for item in dataset.Gender]
X=dataset.iloc[:,1:4].values
y=dataset.iloc[:,4:5].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
y_train=y_train.reshape(-1,1)
regressor=LogisticRegression()
regressor.fit(X_train,y_train.ravel())
y_pred=regressor.predict(X_test)
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
print('ACCURACY: ',metrics.accuracy_score(y_test,y_pred))
print('PRECISION: ',metrics.precision_score(y_test,y_pred))
print('RECALL: ',metrics.precision_score(y_test,y_pred))