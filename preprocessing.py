# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:13:38 2017

@author: azeng
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

training = pd.read_csv('./training.csv') 
test = pd.read_csv('./test.csv') 

buyers = pd.crosstab(training.IsBadBuy, training.BYRNO, normalize='columns')
buyers.drop(buyers.index[0], inplace=True)
average = np.mean(training['IsBadBuy'])

prices = training[['MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                  'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']]
cost = training['VehBCost']

prices = prices[1:100][:]
cost = cost[1:100][:]


plt.figure(1)
plt.title('Purchase price vs values')
for column in prices:
    color = np.random.rand(3,)
    plt.scatter(cost, prices[column], s=10, c=color)
    par = np.polyfit(cost, prices[column], 1, full=True)
    x1 = [min(cost), max(cost)]
    y1 = [par[0][0]*xx + par[0][1] for xx in x1]
    plt.plot(x1, y1, c=color)
plt.legend()
plt.xlabel('Purchase Cost')
plt.ylabel('Values')
plt.figure(2)
row = buyers.iloc[0]
row.plot(kind='bar', stacked = True)
plt.axhline(y=average, color = 'r', linestyle = '-')
plt.title('Bad purchases of different buyers')
plt.ylabel('Bad Buy Rate')
plt.show()


X_train = training.drop(['RefId', 'WheelType', 'Model', 'PurchDate', 'VehYear', 'Trim', 'SubModel', 'Nationality', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART',
                         'VNZIP1', 'IsOnlineSale'], axis = 1)
X_test = test.drop(['RefId', 'WheelType', 'Model', 'PurchDate', 'VehYear', 'Trim', 'SubModel', 'Nationality', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART',
                         'VNZIP1', 'IsOnlineSale'], axis = 1)
    
X_train['WheelTypeID'].fillna(0, inplace=True)

X_train['MMRAcquisitionAuctionAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRAcquisitionAuctionCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRAcquisitionRetailAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRAcquisitonRetailCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRCurrentAuctionAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRCurrentAuctionCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRCurrentRetailAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train['MMRCurrentRetailCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_train.dropna(inplace=True)
#X_test.dropna(subset = ['MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
#                 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'], axis = 0, inplace=True)
X_test['MMRAcquisitionAuctionAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRAcquisitionAuctionCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRAcquisitionRetailAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRAcquisitonRetailCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRCurrentAuctionAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRCurrentAuctionCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRCurrentRetailAveragePrice'].fillna(value = X_test['VehBCost'], inplace=True)
X_test['MMRCurrentRetailCleanPrice'].fillna(value = X_test['VehBCost'], inplace=True)

y_train = X_train['IsBadBuy']
X_train = X_train.drop(['IsBadBuy'], axis = 1)

frames = [X_train, X_test]
X_combined = pd.concat(frames)


prices = X_combined[['MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']].values
pca = PCA(n_components = 2)
prices = pca.fit_transform(prices)
prices = pd.DataFrame(prices)
X_combined = X_combined.drop(['MMRAcquisitionAuctionAveragePrice',	'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice',
                 'MMRCurrentAuctionAveragePrice','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice'], axis =1)
X_combined = pd.concat([X_combined, prices], axis = 1, join_axes=[X_combined.index])


X_done = pd.get_dummies(X_combined, columns=['Auction','Make','Color', 'Transmission', 'WheelTypeID', 'Size', 'VNST', 'BYRNO'], prefix = ["auction", "make", "color", "trans", "wheel", "size", "state", "byrno"])
X_train2 = X_done[:X_train.shape[0]][:]
X_test2 = X_done[X_train.shape[0]:][:]


X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train2, y_train, test_size=0.33, random_state=42)

