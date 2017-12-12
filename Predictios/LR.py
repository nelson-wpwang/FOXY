#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:33:00 2017

@author: linglimei
"""

#LINEAR REGRESSION
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
lr = LinearRegression()
df=pd.read_csv('LightningExport_modified.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],encoding='gbk',dtype=str)
'''soldprice = df['SOLDPRICE']
df.drop(labels=['SOLDPRICE'], axis=1,inplace = True)
df.insert(0, 'SOLDPRICE', soldprice)
df
print (df)'''
y=df['SOLDPRICE']
X=df.iloc[:,2:32]
y=y[0:10000]
X=X[0:10000]
X2=X.fillna(0)
X2.drop('HEATING',axis=1, inplace=True)
X2.drop('ELEMENTARYSCHOOL',axis=1, inplace=True)
X2.drop('JUNIORHIGHSCHOOL',axis=1, inplace=True)
X2.drop('HIGHSCHOOL',axis=1, inplace=True)
X2.drop('LEVEL',axis=1, inplace=True)
X2.drop('CITY',axis=1, inplace=True)
X2.drop('STYLE',axis=1, inplace=True)

X2.drop('ZIP',axis=1, inplace=True)


#X.drop('LISTPRICE',axis=1, inplace=True)

offset = int(X2.shape[0] * 0.6)
X_train2, y_train2 = X2[:offset], y[:offset]
X_test2, y_test = X2[offset:], y[offset:]
y_t=pd.to_numeric(y_test)
lr.fit(X_train2, y_train2)
lr_preds = lr.predict(X_test2)
LR=(y_t-lr_preds)
LR=LR/y_t
LR_mean=np.mean(LR.abs())
print (LR_mean)