# Exp-06 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SURIYA M
RegisterNumber:  212223110055
*/
```
~~~
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()
~~~
![428065887-c137c01b-df00-484c-97d5-236e947c0ccf](https://github.com/user-attachments/assets/a529740a-63e8-440e-9e99-2dbbf1274202)
~~~
data = data.drop(['sl_no', 'salary'], axis=1)
data
~~~
![428066058-20fb24a0-1129-43ea-8248-16506b8dc298](https://github.com/user-attachments/assets/47d50c77-9ef9-4a1d-8845-04b6509e293d)
~~~
data["gender"]=data["gender"].astype('category') 
data["ssc_b"]=data["ssc_b"].astype('category') 
data["hsc_b"]=data["hsc_b"].astype('category') 
data["degree_t"]=data["degree_t"].astype('category') 
data["workex"]=data["workex"].astype('category') 
data["specialisation"]=data["specialisation"].astype('category') 
data["status"]=data["status"].astype('category') 
data["hsc_s"]=data["hsc_s"].astype('category') 
data.dtypes
~~~
![428066179-222ec0f5-fd00-4b15-8e5a-07869cadaeec](https://github.com/user-attachments/assets/f50a1202-38da-4861-bf27-69a12bb462bd)
~~~
data["gender"]=data["gender"].cat.codes 
data["ssc_b"]=data["ssc_b"].cat.codes 
data["hsc_b"]=data["hsc_b"].cat. codes
data["degree_t"]=data["degree_t"].cat.codes 
data["workex"]=data["workex"].cat.codes 
data["specialisation"]=data["specialisation"].cat.codes 
data["status"]=data["status"].cat.codes 
data["hsc_s"]=data["hsc_s"].cat.codes 
data
~~~
![428066474-c9f9beb8-0cc9-444e-8be2-e09fc7c3fd01](https://github.com/user-attachments/assets/650bf55c-7dcd-4e43-bdbb-b7df243f0efb)
~~~
x = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values 
y
~~~
![428066638-c8c31bb0-d54c-472f-b6eb-f36bcbde5bef](https://github.com/user-attachments/assets/7d048205-f457-41e4-b12f-b8fb065040c9)
~~~
theta = np.random.randn(x.shape[1]) 
Y = y
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y): 
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent(theta, X, y, alpha, num_iterations): 
    m = len(y)
    for i in range(num_iterations): 
        h = sigmoid(X.dot(theta)) 
        gradient = X.T.dot(h - y) / m 
        theta -= alpha * gradient 
    return theta
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred

y_pred = predict(theta, x) 
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy: ", accuracy) 
print(y_pred)
~~~
![428067211-2ba0f57b-f2a5-4f00-9b69-25cb967c7594](https://github.com/user-attachments/assets/f8752f06-1332-4b30-aa93-ccd4e9868aae)
~~~
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
~~~

![428067396-a4746cc7-6c49-45fc-909b-c7457caa2edd](https://github.com/user-attachments/assets/04d4f991-705b-49db-ac00-61c026ef77ba)
~~~
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
~~~
![428067556-93a005d6-07b5-4273-8a78-c2ad1238db1a](https://github.com/user-attachments/assets/22324dde-4eb0-4e0a-a21f-e32ceb1e0934)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

