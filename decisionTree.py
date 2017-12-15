import csv
import pandas as pd
from preprocessing import preprocess
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.externals import joblib

def my_scoring(true, predicted):
    c_matrix = confusion_matrix(true, predicted, labels=[0, 1])
    return average_accuracy(c_matrix)

def average_accuracy(matrix):
    correct = 0
    total = 0
    for i in range(len(matrix[0])):
        correct += matrix[i][i]
        total += sum(matrix[i])

    return correct / total

def get_column_name(data, index):
    print("Name of column %d is %s" % (index, data.columns[index]))


# def get(string):
#     a = np.array(data)
#     return list(map(str, a[1:, data[0].index(string)]))
#
# def column(matrix, i):
#     return [row[i] for row in matrix]
#
# def submatrix(matrix, x, y):
#     return [row[x::] for row in matrix[y::]]

# with open('training.csv', 'r') as f:
#     reader = csv.reader(f)
#     initial = list(reader)

# y = column(initial, 1)[1:]
# X = submatrix(initial,2,1)

X, y = preprocess(True, False)

for i in [62, 0, 7, 2, 191, 92]:
    get_column_name(X,i)

# print(y)

my_scorer = make_scorer(my_scoring)
gcv = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid={'max_depth': range(1,5)}, scoring=my_scorer)
gcv.fit(X, y)
print('Best depth parameter = %d' % gcv.best_params_['max_depth'])

dt = DecisionTreeClassifier(criterion='entropy', max_depth= gcv.best_params_['max_depth'])
dt.fit(X, y)
cm = confusion_matrix(y, dt.predict(X))

print(cm)
print(average_accuracy(cm))



with open("_gini.dot", 'w') as f:
    f = tree.export_graphviz(dt, out_file=f)