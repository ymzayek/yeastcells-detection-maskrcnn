# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
from shapely.geometry import Polygon, Point
import math

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

def polygons_per_cluster(labels, contours, output, include_noise=False):
    if include_noise is True:
        polygons = {l: {}  for l in set(labels) if l>=-1}
    else:
        polygons = {l: {}  for l in set(labels) if l>=0}
    for z, (labels_, x, y) in enumerate(zip(group(labels, output), *contours)):
        for label, x_, y_ in zip(labels_, x, y):
            p = np.concatenate(
                (x_[:, None], y_[:, None]),
                axis=1
            )
            shape = Polygon(p) if len(p) >= 3 else Point(x_.mean(), y_.mean())
            polygons[label][z] = shape
            
    return polygons

def get_distance(p1, p2): 
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    
    return distance

def get_features_df(polygons, labels, pred_s_df):
    pred_s_features_df = pred_s_df.copy()
    poly_area =np.zeros((len(labels)),dtype=float)
    n=0
    for lab, frame in zip(pred_s_df.Cell_number, pred_s_df.Frame_number):
        poly_area[n] = polygons[lab][frame-1].area
        n+=1
    pred_s_features_df["Area"] = poly_area  
    
    area_std = np.zeros((len(labels)),dtype=float)
    n=0
    for l in pred_s_df.Cell_number:
        if l is -1:
            area_std[n] = 0
        else:    
            area_std[n] = np.std(
                pred_s_df.loc[pred_s_df['Cell_number'] == l, 'Area']
            )
        n+=1
    pred_s_features_df["Area_stdev"] = poly_area
    
    position_std = np.zeros((len(labels)),dtype=float)
    n=0
    for l in pred_s_df.Cell_number:
        if l is -1:
            position_std[n] = 0
        else:    
            points_xy = np.array(
                pred_s_df.loc[
                pred_s_df['Cell_number'] == l, ('Position_X', 'Position_Y')
            ])
            dist_xy = []
            for i in range(len(points_xy)-1):
                dist_xy.append(get_distance(points_xy[i], points_xy[i+1]))
            position_std[n] = np.std(dist_xy)
        n+=1  
    pred_s_features_df["Position_stdev"] = poly_area
    
    # g_rate = np.zeros((len(polygons)),dtype=float)
    # for l in polygons.keys():
    #     end = max(pred_s_df['Frame_number'].values[labels==l])
    #     start = min(pred_s_df['Frame_number'].values[labels==l])
    #     g_rate[l] = (
    #         polygons[l][end-1].area-polygons[l][start-1].area)/(
    #         polygons[l][start-1].area+0.00001
    #     ) # -1 because pred_s starts at 1 and polygons starts at 0
    #     g_rate = abs(g_rate)
    # #assign growth rate per segmented instance    
    # g_rate_ = labels.astype(float).copy() 
    # for g in range(len(g_rate)):
    #     g_rate_[g_rate_==g] = g_rate[g]  
    # pred_s_features_df["Growth_rate"] = g_rate_ 
        
    return pred_s_features_df 

