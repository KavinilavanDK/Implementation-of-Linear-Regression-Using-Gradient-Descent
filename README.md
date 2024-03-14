## Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KAVI NILAVAN DK
RegisterNumber:  2122232330103
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```
## Output:
![ml 1](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/835057df-c475-4852-9fd5-4c62c1977d6e)

![ml 2](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/805adeb3-99a7-4c02-baa2-cca33c6dc80e)

![ml 3](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/18c47293-016b-423a-bb86-9a7c6b10660b)

![ml 4](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/00507ea1-a8fa-4941-a692-d3815c2ed880)

![ml 5](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/e60e342f-7625-4ae0-9191-06b3b5e151ee)

![ml 6](https://github.com/KavinilavanDK/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144870429/54b09571-9023-47d7-813c-1d3c54442632)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
