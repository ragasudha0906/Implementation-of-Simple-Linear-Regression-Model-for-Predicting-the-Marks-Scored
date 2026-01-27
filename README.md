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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
marks = np.array([20, 25, 35, 45, 50, 60, 65, 70, 80, 85])

model = LinearRegression()
model.fit(hours, marks)

predicted_marks = model.predict(hours)

plt.scatter(hours, marks, label="Actual Data")

plt.plot(hours, predicted_marks, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Hours vs Marks (Simple Linear Regression)")
plt.legend()
plt.show()
 
*/
```

## Output:


<img width="763" height="573" alt="Screenshot 2026-01-27 152531" src="https://github.com/user-attachments/assets/9e83ad2f-c7c8-45c6-86c8-2d290acfeb7e" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
