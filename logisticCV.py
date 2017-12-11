# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:40:28 2017

@author: azeng
"""
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

apc = make_scorer(accuracy_score)
gs = GridSearchCV(DecisionTreeClassifier('gini'), param_grid={'max_depth': [1,2,3,4,5,6,7,8,9,10]}, scoring = apc, cv = 3)
gs.fit(X_train2, y_train)
results = gs.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV",
          fontsize=16)

plt.xlabel("max depth")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
ax.set_xlim(0, 10)
ax.set_ylim(0.87, 0.91)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_max_depth'].data, dtype=float)
sample_score_mean = results['mean_test_score']
sample_score_std = results['std_test_score']
ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1, color='g')
ax.plot(X_axis, sample_score_mean, '-', color='g',
        alpha=1,
        label="score (test)")

best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
best_score = results['mean_test_score'][best_index]

# Plot a dotted vertical line at the best score for that scorer marked by x
ax.plot([X_axis[best_index], ] * 2, [0, best_score],
        linestyle='-.', color='g', marker='x', markeredgewidth=3, ms=8)

# Annotate the best score for that scorer
ax.annotate("%0.2f" % best_score,
            (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()

clf = DecisionTreeClassifier('gini', max_depth=best_index)
clf.fit(X_train3, y_train3)
y_pred = clf.predict(X_test3)
cm = confusion_matrix(y_test3, y_pred)
print (cm)
score8 = clf.score(X_test3[(y_test3==0)], y_test3[(y_test3==0)])
score9 = clf.score(X_test3[(y_test3==1)], y_test3[(y_test3==1)])
print("Accuracy for y=0: ", score8," Accuracy for y=1: ", score9)
apc = (cm[0,0]+cm[1,1])/y_test3.shape[0]
print("average per-class accuracy: ", apc)