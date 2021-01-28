# -*- coding: utf-8 -*-
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#method 1
def remove_by_size(contours, points = 10):
    cont = copy.deepcopy(contours)  
    for i in range(len(cont[0])): 
        for c in range(len(cont[0][i])):
            if len(cont[0][i][c]) <= points:
                cont[0][i][c] = np.array([])
              
    for i in range(len(cont[1])):
        for c in range(len(cont[1][i])):
            if len(cont[1][i][c]) <= points:
                cont[1][i][c] = np.array([])   
                
    idx = []
    x_, _=cont
    i=0
    for c_list in x_:
        for c in c_list:
            if np.size(c) == 0:
                idx.append(i)
            i+=1  
            
    return idx 

#method 2
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state = 42
    )
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train);
    
    return rf

def remove_by_rf(rf, X):
    predictions = rf.predict(X)
    idx = []
    i=0
    for c in predictions:
        if c == 0:
            idx.append(i)
        i+=1  
    
    return idx