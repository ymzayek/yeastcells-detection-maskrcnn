# -*- coding: utf-8 -*-
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from .features import get_area_and_growth_rate

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
def add_features(polygons, labels, pred_s): #add to pipeline instead
    areas, g_rate = get_area_and_growth_rate(polygons, labels, pred_s)
    pred_s_df = pd.DataFrame(
    pred_s, columns=["Frame_number", "Cell_number",
                     "Position_X", "Position_Y"]
    )
    pred_s_df["Area"] = areas
    pred_s_df["Growth_rate"] = g_rate
    features = pred_s_df[["Growth_rate", "Area"]]
    features = np.array(features) 
    
    return features

def train_RF(tp_label, features):
    X_features, y_features, X_labels, y_labels = train_test_split(
        features, tp_label, test_size = 0.25, random_state = 42
    )
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    rf.fit(X_features, X_labels);
    
    return rf

def remove_by_RF(rf, features):
    predictions = rf.predict(features)
    idx = []
    i=0
    for c in predictions:
        if c == 0:
            idx.append(i)
        i+=1  
    
    return idx