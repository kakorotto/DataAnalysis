# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:44:00 2019

@author: Philo Ibrahim
"""
# classification

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read datasest
dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2, 3]].values
y = dataset.iloc[:,4].values

# cutting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)



#freature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#logic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train) 

# test the model 
y_predict = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)

x = cm[0][0]
y = cm [0][1]
z = cm [1][0]
w = cm [1][1]

accuracy = (x+w) * 100 / (x+y+z+w)

print("the accuracy is:", accuracy,"%")

