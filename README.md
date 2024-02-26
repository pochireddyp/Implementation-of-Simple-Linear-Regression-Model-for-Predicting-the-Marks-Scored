# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by : pochireddy.p
RegisterNumber : 212223240115
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
print("HEAD:")
print(df.head())
print("TAIL:")
print(df.tail())
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="green")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
MSE = mean_squared_error(Y_test,Y_pred)
print('MSE = ',MSE)
MAE = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',MAE)
RMSE=np.sqrt(MSE)
print("RMSE = ",RMSE)
```

## Output:
### df.head() & df.tail()
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/c6df6ee7-5304-407f-9da0-d7c6b7c8e941)
### Array values of X
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/9cd3e2a2-6050-4c04-a426-195e7a27a76c)
### Array values of Y
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/3a26c13a-afb3-472a-826e-c93fb31b1384)
### Predicted Values of Y
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/94556c75-1e07-49d7-a76a-22c68c09bedf)
### Test Values of Y
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/0f2832f3-0d76-46f3-8511-fd862acd0e09)
### Training Set Graph
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/9f7fefcd-654c-4f99-a973-e928296f9595)
### Testing Set Graph
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/19bf3806-6f54-49e6-b139-37a03fdac85f)
### Values of MSE, MAE & RMSE
![image](https://github.com/Madhavareddy09/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742470/2009f2aa-f13f-414c-b133-fef60a322144)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
