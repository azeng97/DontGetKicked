# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:19:25 2017

@author: azeng
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import (train_test_split,KFold)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm

apc_scorer = make_scorer(accuracy_score)
clf = GradientBoostingClassifier()
clf.fit(X_train3, y_train3)
y_pred = clf.predict(X_test3) 
cm = confusion_matrix(y_test3, y_pred)
print (cm)
score0 = clf.score(X_test3[(y_test3==0)], y_test3[(y_test3==0)])
score1 = clf.score(X_test3[(y_test3==1)], y_test3[(y_test3==1)])
print("score 0: ", score0," score 1: ", score1)
apc = (cm[0,0]+cm[1,1])/y_test3.shape[0]
print("average per-class accuracy: ", apc)
#cv_score = cross_val_score(clf, X_train2, y_train, scoring=apc_scorer, cv = 3)
#print("cv score: ", cv_score )

#clf1 = DecisionTreeClassifier('gini', max_depth=3)
#clf1.fit(X_train3,y_train3)
#with open("decision_tree_gini.dot", 'w') as f:
#    f = tree.export_graphviz(clf1, out_file=f)
#y_pred = clf1.predict(X_test3) 
#cm = confusion_matrix(y_test3, y_pred)
#print (cm)
#score0 = clf1.score(X_test3[(y_test3==0)], y_test3[(y_test3==0)])
#score1 = clf1.score(X_test3[(y_test3==1)], y_test3[(y_test3==1)])
#print("score 0: ", score0," score 1: ", score1)
#apc = (cm[0,0]+cm[1,1])/y_test3.shape[0]
#print("average per-class accuracy: ", apc)
#cv_score = cross_val_score(clf1, X_train2, y_train, scoring=apc_scorer)
#print("cv score: ", cv_score )


