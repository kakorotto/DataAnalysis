# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_predict = reg.predict(x_test) # predict y values for x_test values

# visualization
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, reg.predict(x_train), color = "blue")
plt.title("salary VS exp")
plt.xlabel("exp_years")
plt.ylabel("salary")
plt.show()

# test
plt.scatter(x_test, y_test, color = "green")
plt.plot(x_test, reg.predict(x_test), color = "brown")
plt.title("salary VS exp")
plt.xlabel("exp_years")
plt.ylabel("salary")
plt.show()


"""
missed values: using mean value
    
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = "NaN", strategy = "mean")
imp = imp.fit(x[:, 1:3])
x[:, 1:3] = imp.transform(x[:, 1:3])
    
"""