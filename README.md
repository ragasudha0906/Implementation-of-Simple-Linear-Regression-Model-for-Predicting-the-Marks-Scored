# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import the required libraries such as NumPy, Matplotlib, and Linear Regression from sklearn.
2. Define the dataset by storing the hours studied as the independent variable (X) and marks scored as the dependent variable (Y). 
3. Create the Simple Linear Regression model and train it using the given dataset.
4. Predict the marks using the trained regression model for the given input values.
5. Plot the graph by displaying the actual data points using a scatter plot and the regression line using a line plot, then display the graph.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RAGASUDHA R
RegisterNumber: 212224230215
*/
```

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("student_scores.csv")

X = data['Hours'].values
Y = data['Scores'].values

X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean) * (Y[i] - Y_mean)
    den += (X[i] - X_mean) ** 2

m = num / den

b = Y_mean - (m * X_mean)

Y_pred = m * X + b

print("Slope (m):", m)
print("Intercept (b):", b)

hours = float(input("Enter number of study hours: "))
predicted_marks = m * hours + b
print("Predicted Marks:", predicted_marks)

plt.scatter(X, Y)
plt.plot(X, Y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression – Marks Prediction")
plt.show()


```

## Output:

<img width="972" height="835" alt="Screenshot 2026-01-28 161356" src="https://github.com/user-attachments/assets/1d959879-2e62-4887-b2d6-7c5e0415d5a4" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
