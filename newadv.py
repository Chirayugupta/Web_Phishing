# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:04:50 2019

@author: shashikant, sailesh, chirayu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import metrics


dataset = pd.read_csv("dataset_kaggle.csv")

dataset=dataset.drop(['index'],axis=1)

list=[0,1,6,10,11,12,13,14,15,16,18,21,23,25,27]

x=dataset.iloc[:,0:30]
y=dataset.loc[:,['Result']]



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )


classifier = RandomForestClassifier(n_estimators = 50, criterion = "gini", max_features = 'log2',  random_state = 0)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



joblib.dump(classifier, 'rf_final.pkl')



importances =classifier.feature_importances_
names = dataset.iloc[:,0:30].columns
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height = 0.7)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()
