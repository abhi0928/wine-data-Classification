# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:49:35 2019

@author: Abhishek Omi
"""

# Importing Libraries
import numpy as np 
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Loading the wine dataset
wine = pd.read_csv('winequality-red.csv')

wine['quality'] = pd.cut(wine['quality'], bins = [0, 6, 10], labels=  ['bad', 'good'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
wine['quality'] = le.fit_transform(wine['quality'])


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(wine, wine['quality']):
    train_set = wine.loc[train_index]
    test_set = wine.loc[test_index]

train_f = train_set.drop('quality', axis = 1)
test_f = test_set.drop('quality', axis = 1)
train_l = train_set['quality']
test_l = test_set['quality']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_f = sc.fit_transform(train_f)
test_f = sc.transform(test_f)

clf_RF = RandomForestClassifier(max_features = 'log2', n_estimators = 600)
clf_RF.fit(train_f, train_l)

predictions = clf_RF.predict(test_f)

from joblib import load, dump
dump(clf_RF, 'modal_usage.joblib')




                      


