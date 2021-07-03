# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cancer = pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/Prostate_Cancer.csv")
#EDA
cancer.info()
cancer.head(10)
cancer.tail(5)
cancer.describe()
cancer.columns
#removing a column
cancer.drop(['id'], axis=1,inplace=True)

#replacing diagnosis data to object from integers
cancer.diagnosis_result = [1 if each == 'M' else 0 for each in cancer.diagnosis_result]
cancer.diagnosis_result.value_counts()
cancer.radius.value_counts()

#decision tree
#splitting the data

from sklearn.model_selection import train_test_split
Y = cancer['diagnosis_result']
X = cancer.drop(columns=['diagnosis_result'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10)

# train the model
logreg.fit(X_train, Y_train)

# predict target values
Y_predict1 = logreg.predict(X_test)
score_logreg = logreg.score(X_test, Y_test)
print(score_logreg)
#accuracy=0.85

#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)

# train model
knncla.fit(X_train, Y_train)
predict target values
Y_predict6 = knncla.predict(X_test)

#accurcay
score_knncla= knncla.score(X_test, Y_test)
print(score_knncla)
#accurcay=0.7

#conclusion - Logistic Regression is a better model than KNN for the given data














































X_train,Y_train,X_test,Y_test=train_test_split(x,y,test_size = 0.2)
X_train = train.iloc[:,1:]
Y_train= train.iloc[:,0]
X_test= test.iloc[:,1:]
Y_test= test.iloc[:,0]

from sklearn.preprocessing import StandardScaler

#standar Scaler
sc=StandardScaler()
x=sc.fit_transform(x)

#model fitting
acc={}

from sklearn.tree import DecisionTreeClassifier
tree.plot_tree(dtc)
dtc=DecisionTreeClassifier()
dtc.fit(X_train, Y_train)                        
acc['Decision Tree']=accuracy_score(Y_test,dtc.predict(X_test))                      
accuracy_score(Y_test,dtc.predict(X_test))     
#accuracy=0.7                                          

#random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()                                            
rfc.fit(X_train,Y_train)                                            
acc['Random Forest']=accuracy_score(Y_test, rfc.predict(X_test))                                          
accuracy_score(Y_test, rfc.predict(X_test))  
#accuracy=0.8

#adaboost classifer
from sklearn.ensemble import AdaBoostClassifier 
abc=AdaBoostClassifier()                              
abc.fit(X_train,Y_train)                                            
acc['Ada Boost']= accuracy_score(Y_test, rfc.predict(X_test))                                           
accuracy_score(Y_test, rfc.predict(X_test))  
#accuracy=0.8    