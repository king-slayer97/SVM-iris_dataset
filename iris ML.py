# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:20:48 2018

@author: Sandeep
"""
from sklearn import datasets,linear_model,svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc

iris=datasets.load_iris()
iris_x=iris.data[:,1:]
iris_y=iris.target
iris_x_train,iris_x_test,iris_y_train,iris_y_test=train_test_split(iris_x,iris_y,test_size=0.24,random_state=0)
regr=svm.SVC()
regr.fit(iris_x_train,iris_y_train)
pred=regr.predict(iris_x_test)
print(acc(pred,iris_y_test))