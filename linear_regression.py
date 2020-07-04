#PREDICTING SALARY BASED ON EXPERIENCE USING SIMPLE LINEAR REGRESSION
import pandas as pd
dataset=pd.read_csv('Salary_Data.txt')
X=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
X_train=X_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
for i in range(len(y_pred)):
    print(y_pred[i],y_test[i])
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.xlabel('EXPERIENCE')
plt.ylabel('SALARY')
plt.title('EXPERIENCE vs SALARY(Training Set)')
plt.show()
import matplotlib.pyplot as plt
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.xlabel('EXPERIENCE')
plt.ylabel('SALARY')
plt.title('EXPERIENCE vs SALARY(Testing Set)')
plt.show()
#Make a new Prediction
exp=int(input('Enter experience in numbers'))
print('Predicted salary is',regressor.predict([[exp]]))

