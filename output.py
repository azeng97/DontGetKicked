# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:49:01 2017

@author: azeng
"""
 

clf = DecisionTreeClassifier("gini", max_depth=4)
clf.fit(X_train2, y_train)
y_pred2 = clf.predict_proba(X_test2)
ref = test['RefId']
df1 = pd.DataFrame(y_pred2[:,1])
df2 = pd.DataFrame(ref)
df = pd.concat([df2, df1], axis = 1)
df.to_csv('submission.csv', header=['RefId','IsBadBuy'], index=None)
