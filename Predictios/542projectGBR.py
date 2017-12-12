#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:28:40 2017

@author: linglimei
"""
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



# #############################################################################
# Load data
df=pd.read_csv('LightningExport_modified.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],encoding='gbk',dtype=str)
'''soldprice = df['SOLDPRICE']
df.drop(labels=['SOLDPRICE'], axis=1,inplace = True)
df.insert(0, 'SOLDPRICE', soldprice)
df
print (df)'''
y=df['SOLDPRICE']
X=df.iloc[:,2:32]
y=y[0:8000]
X=X[0:8000]

Xgbr=X.fillna('0')

#print (X)
#X1=pd.get_dummies(Xgbr['ZIP'], prefix='zip')
X2=pd.get_dummies(Xgbr['LEVEL'], prefix='level')
#X3=pd.get_dummies(Xgbr['STYLE'], prefix='style')
X4=pd.get_dummies(Xgbr['CITY'], prefix='city')
#X5=pd.get_dummies(Xgbr['HEATING'], prefix='heating')
#X6=pd.get_dummies(Xgbr['ELEMENTARYSCHOOL'], prefix='elementaryL')
#X7=pd.get_dummies(Xgbr['HIGHSCHOOL'], prefix='highschool')
#X8=pd.get_dummies(Xgbr['JUNIORHIGHSCHOOL'], prefix='juniorschool')
#print (X1)
#print (X2)








#all_dummy_df.head()

#print (Xgbr)
#print(type(X['CITY'][1]))
#print(type(X['STYLE'][1]))
'''le.fit(X1['CITY'])
op=le.transform(X1['CITY'])
X1['CITY']=op

le.fit(X1['STYLE'])
op=le.transform(X1['STYLE'])
X1['STYLE']=op
#print (X)
le.fit(X1['LEVEL'])
op=le.transform(X1['LEVEL'])
X1['LEVEL']=op


le.fit(X1['HEATING'])
op=le.transform(X1['HEATING'])
X1['HEATING']=op

le.fit(X1['ELEMENTARYSCHOOL'])
op=le.transform(X1['ELEMENTARYSCHOOL'])
X1['ELEMENTARYSCHOOL']=op

le.fit(X1['JUNIORHIGHSCHOOL'])
op=le.transform(X1['JUNIORHIGHSCHOOL'])
X1['JUNIORHIGHSCHOOL']=op
le.fit(X1['HIGHSCHOOL'])
op=le.transform(X1['HIGHSCHOOL'])
X1['HIGHSCHOOL']=op'''


#X=X.iloc[:,0:9]
Xgbr.drop('HEATING',axis=1, inplace=True)
Xgbr.drop('ELEMENTARYSCHOOL',axis=1, inplace=True)
Xgbr.drop('JUNIORHIGHSCHOOL',axis=1, inplace=True)
Xgbr.drop('HIGHSCHOOL',axis=1, inplace=True)
Xgbr.drop('CITY',axis=1, inplace=True)
Xgbr.drop('STYLE',axis=1, inplace=True)
Xgbr.drop('ZIP',axis=1, inplace=True)
#X1.drop('ELEMENTARYSCHOOL',axis=1, inplace=True)
#Xgbr.drop('SQFT',axis=1, inplace=True)
#Xgbr.drop('LOTSIZE',axis=1, inplace=True)
#Xgbr.drop('AGE',axis=1, inplace=True)
Xgbr.drop('LEVEL',axis=1, inplace=True)
#Xgbr.drop('DOM',axis=1, inplace=True)
#Xgbr.drop('DTO',axis=1, inplace=True)
#X1.drop('GARAGE',axis=1, inplace=True)
#X1.drop('LISTPRICE',axis=1, inplace=True)
#print (X.head(10))
#X.ELEMENTARYSCHOOL.fillna(0).JUNIORHIGHSCHOOL.fillna(0).HIGHSCHOOL.fillna(0).LEVEL.fillna(0)
#result1 = pd.merge(X1, X2)
#print (X1)

#print (X2)
#print (X1)
Xnew=pd.concat([X2,X4],axis=1)
Xnew2=pd.concat([Xnew,Xgbr],axis=1)
#Xnew3=pd.concat([Xnew2,Xgbr],axis=1)
#Xnew4=pd.concat([Xnew3,X8],axis=1)
#Xnew5=pd.concat([Xnew4,X],axis=1)
#Xnew6=pd.concat([Xnew5,X8],axis=1)
#Xnew7=pd.concat([Xnew6,X],axis=1)'''





print (Xnew2)
#print (df)
#print (X)

#X=X[0:10000]
offset = int(Xnew2.shape[0] * 0.3)

X_train1, y_train1 = Xnew2[:offset], y[:offset]
X_test1, y_test = Xnew2[offset:], y[offset:]
#min_max_scaler = preprocessing.MinMaxScaler()

#X_train_minmax = min_max_scaler.fit_transform(X['LISTPRICE'])
#X['LISTPRICE']=X_train_minmax





# #############################################################################
# Fit regression model
params = {'n_estimators': 3000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.06, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train1, y_train1)
#featurelist=X1.columns.values.tolist()

#print (featurelist)

#print (X)



test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test1)):
    y_t=pd.to_numeric(y_test)
    QAQ=(y_t-y_pred)
    QAQ=QAQ/y_t
    qaq=np.mean(QAQ.abs())
    test_score[i] = qaq

    

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Iterations')
plt.ylabel('qaq')

plt.subplot(212)
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Iterations')
plt.ylabel('qaq')
plt.show()
print(min(test_score))

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, featurelist[sorted_idx])
#print(sorted_idx)
#featurelist=X1.columns.values.tolist()
list=[]
for i in range (0,357):
    
    list.append(Xnew2.columns[sorted_idx[i]])
print (list)

#print (featurelist)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
