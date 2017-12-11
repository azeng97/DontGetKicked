# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:13:38 2017

@author: azeng
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm

training = pd.read_csv('./training.csv') 
training = training[['IsBadBuy','VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost']]
training = training.dropna()

y = training['IsBadBuy']
X = training[['VehicleAge', 'VehOdo', 'WarrantyCost']]

prices = training[['MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']]
AvePrice = np.mean(prices, axis=1).round()
cost = training['VehBCost']
diff = cost-AvePrice
X = X.assign(costDiff=diff.values)
X_bad = X[y==1]
X_good = X[y==0]

plt.scatter(X_bad['VehicleAge'], X_bad['costDiff'], s=10, c='b')
plt.scatter(X_good['VehicleAge'], X_good['costDiff'], s=1, c='g')
plt.title("Vehicle Age vs Difference between cost and value")
plt.ylabel("Vehicle cost - average price")
plt.xlabel("Vehicle Age")
plt.legend(["Bad", "Good"])



X = X.as_matrix()
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = LogisticRegression(penalty='l1', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print (cm)
score0 = clf.score(X_test[(y_test==0)], y_test[(y_test==0)])
score1 = clf.score(X_test[(y_test==1)], y_test[(y_test==1)])
print("score 0: ", score0," score 1: ", score1)
apc = (cm[0,0]+cm[1,1])/y_test.shape[0]
print("average per-class accuracy: ", apc)





