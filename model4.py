# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:26:45 2019

@author: Pola Attya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
sum_acc=0
X=30
for i in range (0,X):
      
    dataset = pd.read_csv("Social_Network_Ads.csv")
    x = dataset.iloc[:,[2, 3]].values
    y = dataset.iloc[:,4].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = i)
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
    knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    cm2 = confusion_matrix(y_test, y_pred)
    print(cm2)
    x = cm2[0][0]
    y = cm2 [0][1]
    z = cm2 [1][0]
    w = cm2 [1][1]
    accuracy = (x+w) * 100 / (x+y+z+w)
    sum_acc=sum_acc+accuracy
avg=sum_acc/X
print(avg)
"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""
"""
accuracy without scaling = 65.83333333333333
accuracy with scaling = 87.5
"""