import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

df = pd.read_csv('diabetes_data_upload.csv', sep=';')

# print(df.head())

X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable
# print(X)
# print(y)

feature_cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                'muscle stiffness', 'Alopecia', 'Obesity']
class_cols = ['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

# Confusion Matrix visualization
# metrics.plot_confusion_matrix(clf, X_test, y_test)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict, labels=clf.classes_).ravel()

accuracy = metrics.accuracy_score(y_test, y_predict)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f_score = (2 * sensitivity * specificity) / (sensitivity + specificity)

# print(' Accuracy Score=', accuracy, '\n', 'Sensitivity=', sensitivity, '\n', 'Specivity=', specificity, '\n',
#       'F Soore=', f_score)

# ROC curve visualization
# metrics.plot_roc_curve(clf, X_test, y_test)
# plt.show()

# Desicion Tree visualization
# plt.figure(figsize=(8, 8))
# tree.plot_tree(clf, fontsize=6)
# plt.show()

# save the model to disk

print("Current working directory: {0}".format(os.getcwd()))
os.chdir(os.getcwd() + '\django\mysite')
print("Current working directory: {0}".format(os.getcwd()))
filename = 'finalized_model.sav'
joblib.dump(clf, open(filename, 'wb'))
