# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:26:45 2019

@author: Pola Attya
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:,[2, 3]].values
y = dataset.iloc[:,4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = LogisticRegression()
classifier.fit(x_train, y_train) 
y_predict = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
x = cm[0][0]
y = cm [0][1]
z = cm [1][0]
w = cm [1][1]
accuracy = (x+w) * 100 / (x+y+z+w)

print("the accuracy is:", accuracy,"%")

"""
accuracy without scaling = 65.83333333333333
accuracy with scaling = 87.5
"""