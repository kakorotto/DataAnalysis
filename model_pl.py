# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 23:24:47 2019

@author: Pola Attya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = (pd.DataFrame)(dataset.iloc[:,:-1].values)
y = (pd.DataFrame)(dataset.iloc[:,3].values)

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = (pd.DataFrame)(imputer.transform(X[:, 1:3]))
"""
#Encode catagorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(Categorical_features = [0])
X = onehotencoder.fit_trsnsform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_X.fit_transform(y)

#Splitting the data

from sklearn.cross_valdiation import train_test_split
"""