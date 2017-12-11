# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 04:52:23 2017

@author: azeng
"""
from sklearn.metrics import roc_curve, roc_auc_score
logr = KNeighborsClassifier()
problda = logr.fit(X_train3, y_train3).predict_proba(X_test3)
fpr1, tpr1, _ = roc_curve(y_test3, problda[:, 1])
roc_auc1 = roc_auc_score(y_test3, problda[:, 1] )

rdf = AdaBoostClassifier()
probqda = rdf.fit(X_train3, y_train3).predict_proba(X_test3)
fpr2, tpr2, _ = roc_curve(y_test3, probqda[:, 1])
roc_auc2 = roc_auc_score(y_test3, probqda[:, 1] )

dt = GradientBoostingClassifier()
probgnb = dt.fit(X_train3, y_train3).predict_proba(X_test3)
fpr3, tpr3, _ = roc_curve(y_test3, probgnb[:, 1])
roc_auc3 = roc_auc_score(y_test3, probgnb[:, 1] )


plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='darkorange',
         lw=lw, label='KNN(area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='darkgreen',
         lw=lw, label='AdaBoost(area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='darkblue',
         lw=lw, label='GDBT(area = %0.2f)' % roc_auc3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()