# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:19:15 2019

@author: Philo Ibrahim
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
x=(pd.DataFrame)(x)
y = dataset.iloc[:,4].values

#encoding
from sklearn.preprocessing import LabelIncoder, OneHotEncoder
labIn = LabelIncoder()
x[:, 3] = labIn.fit_transform(x[:, 3])
OHE = OneHotEncoder(categorical_features = [3])
x = OHE.fit_transform(x).toarray()
x = x[:, 1:]

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y)
y_predict = reg.predict(x)