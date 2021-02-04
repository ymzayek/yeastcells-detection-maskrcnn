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
def get_labels(pred_s, gt_s, output):
    pred = copy.deepcopy(pred_s)
    gt = copy.deepcopy(gt_s)
    masks = [m for i in output for m in np.array(
        i['instances'].pred_masks.to('cpu'), dtype=int
    )]
    y = np.zeros((len(pred)))
    c1 = 0
    for pred_frame, mask in zip(pred[:,0], masks):
        for gt_frame, gt_x, gt_y in zip(gt[:,0], gt[:,2], gt[:,3]):
            if (pred_frame==gt_frame) & (mask[gt_y,gt_x]==1):
                y[c1]=1
        c1+=1    
    
    return y    

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