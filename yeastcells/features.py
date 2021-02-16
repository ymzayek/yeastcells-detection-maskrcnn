# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
from shapely.geometry import Polygon, Point
import math
from .clustering import existance_vectors

def extract_contours(output):
    outputs = output
    x, y = [], []
    for o in outputs:
        x_, y_ = [], []
        for mask in np.array(o['instances'].pred_masks.to('cpu')):
            if mask.max() == False:
                x_.append(np.array([]))
                y_.append(np.array([]))
            else:
                contour, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_TREE, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                x_.append(np.concatenate(
                    [contour[0][:, 0, 0], contour[0][:1, 0, 0]])
                )
                y_.append(np.concatenate(
                    [contour[0][:, 0, 1], contour[0][:1, 0, 1]])
                )
        x.append(x_), y.append(y_)
        
    return x, y

def get_centroids(coordinates, labels):
    centroids = np.zeros(((len(labels),3))).astype(int)    
    centroids[:,0] = labels
    centroids[:,1] = coordinates[:,2] #x
    centroids[:,2] = coordinates[:,1] #y
    
    return centroids

def get_instance_numbers(output):
    o = list(map(existance_vectors, output))
    inst_num = np.array([], dtype=int)
    for f in range(len(o)):
        instance = len(o[f])
        tmp = np.arange(instance)
        inst_num = np.hstack((inst_num,tmp))
    
    coordinates = np.array([
        (t, ) + tuple(map(np.mean, np.where(mask)))
        for t, o in enumerate(output)
        for mask in o['instances'].pred_masks.to('cpu')
    ])    
        
    return inst_num, coordinates    

def group(l, outputs):
    boundaries = np.cumsum([0] + [len(o['instances']) for o in outputs])
    return [
        l[a:b]
        for a, b in zip(boundaries, boundaries[1:])
    ]

def get_seg_track(labels, output, frame = None):
    if frame is None:
        segs = print('The number of segmentations is ' + str(len(labels)))
        tracks = print('The number of tracked cells is ' + str(len(np.unique(labels[labels>=0]))))
    else:
        grouped = group(labels, output) # group labels by frame
        segs = print(f'The number of segmentations in frame {frame} is ' + str(len(grouped[frame-1])))
        tracks = print(f'The number of tracked cells in frame {frame} is ' + str(len(np.unique(grouped[frame-1][labels>=0]))))
    
    return segs, tracks

def track_len(cluster_labels, label_num = 0):
    counts = Counter(cluster_labels)
    return counts[label_num]

def polygons_per_instance(contours, output):
    o = list(map(existance_vectors, output))
    inst = np.array([], dtype=int)
    for f in range(len(o)):
        instance = len(o[f])
        tmp = np.arange(instance)
        inst = np.hstack((inst,tmp))
    polygons_inst = {l: {}  for l in set(inst) if l>=-1}
    for z, (inst_, x, y) in enumerate(zip(group(inst, output), *contours)):
        for i, x_, y_ in zip(inst_, x, y):
            p = np.concatenate(
                (x_[:, None], y_[:, None]),
                axis=1
            )
            shape = Polygon(p) if len(p) >= 3 else Point(x_.mean(), y_.mean())
            polygons_inst[i][z] = shape
            
    return polygons_inst

def polygons_per_cluster(labels, contours, output):
    polygons_clust = {l: {}  for l in set(labels) if l>=0}
    for z, (labels_, x, y) in enumerate(zip(group(labels, output), *contours)):
        for label, x_, y_ in zip(labels_, x, y):
            p = np.concatenate(
                (x_[:, None], y_[:, None]),
                axis=1
            )
            shape = Polygon(p) if len(p) >= 3 else Point(x_.mean(), y_.mean())
            polygons_clust[label][z] = shape
            
    return polygons_clust

def get_distance(p1, p2): 
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    
    return distance

def get_features_df(polygons_inst, labels, pred_df): #make sure polygons include noise
    pred_features_df = pred_df.copy()
    poly_area =np.zeros((len(labels)),dtype=float)
    n=0
    for lab, frame in zip(pred_df.Cell_label, pred_df.Frame_number):
        poly_area[n] = polygons_inst[lab][frame-1].area
        n+=1
    pred_features_df["Area"] = poly_area  
    
    area_std = np.zeros((len(labels)),dtype=float)
    n=0
    for l in pred_df.Cell_label:
        if l is -1:
            area_std[n] = 0
        else:    
            area_std[n] = np.std(
                pred_df.loc[pred_df['Cell_label'] == l, 'Area']
            )
        n+=1
    pred_features_df["Area_stdev"] = poly_area
    
    position_std = np.zeros((len(labels)),dtype=float)
    n=0
    for l in pred_df.Cell_label:
        if l is -1:
            position_std[n] = 0
        else:    
            points_xy = np.array(
                pred_df.loc[
                pred_df['Cell_label'] == l, ('Position_X', 'Position_Y')
            ])
            dist_xy = []
            for i in range(len(points_xy)-1):
                dist_xy.append(get_distance(points_xy[i], points_xy[i+1]))
            position_std[n] = np.std(dist_xy)
        n+=1  
    pred_features_df["Position_stdev"] = poly_area
    
    # g_rate = np.zeros((len(polygons_inst)),dtype=float)
    # for l in polygons_inst.keys():
    #     end = max(pred_df['Frame_number'].values[labels==l])
    #     start = min(pred_df['Frame_number'].values[labels==l])
    #     g_rate[l] = (
    #         polygons_inst[l][end-1].area-polygons_inst[l][start-1].area)/(
    #         polygons_inst[l][start-1].area+0.00001
    #     ) # -1 because pred_s starts at 1 and polygons_inst starts at 0
    #     g_rate = abs(g_rate)
    # #assign growth rate per segmented instance    
    # g_rate_ = labels.astype(float).copy() 
    # for g in range(len(g_rate)):
    #     g_rate_[g_rate_==g] = g_rate[g]  
    # pred_features_df["Growth_rate"] = g_rate_ 
        
    return pred_features_df 

def get_masks(output):
    masks = [
        m for i in output for m in np.array(
        i['instances'].pred_masks.to('cpu'), dtype=int
    )]
    
    return masks