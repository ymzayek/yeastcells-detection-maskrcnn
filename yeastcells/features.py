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
        tracks = print(f'The number of tracked cells in frame {frame} is ' + str(len(np.unique(grouped[frame-1][grouped[frame-1]>=0]))))
    
    return segs, tracks

def track_len(labels, label = 0):
    counts = Counter(labels)
    return counts[label]

def polygons_per_instance(labels, contours, output):
    polygons_inst = {l: {}  for l in set(labels)}
    for z, (labels_, x, y) in enumerate(zip(group(labels, output), *contours)):
        for label, x_, y_ in zip(labels_, x, y):
            p = np.concatenate(
                (x_[:, None], y_[:, None]),
                axis=1
            )
            shape = Polygon(p) if len(p) >= 3 else Point(x_.mean(), y_.mean())
            polygons_inst[label][z] = shape
            
    return polygons_inst

def polygons_per_cluster(labels, contours, output):
    polygons_clust = {l: {}  for l in set(labels) if l>=0}
    for z, (labels_, x, y) in enumerate(zip(group(labels, output), *contours)):
        for label, x_, y_ in zip(labels_, x, y):
            if label < 0: continue
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

def add_area_df(polygons_inst, labels, pred_df): #make sure polygons include noise
    pred_features_df = pred_df.copy()
    poly_area =np.zeros((len(labels)),dtype=float)
    n=0
    for lab, frame in zip(pred_features_df.Cell_label, pred_features_df.Frame_number):
        poly_area[n] = polygons_inst[lab][frame-1].area
        n+=1
    pred_features_df['Area'] = poly_area  
        
    return pred_features_df 

def get_masks(output):
    masks = [
        m for i in output for m in np.array(
        i['instances'].pred_masks.to('cpu'), dtype=int
    )]
    
    return masks

def get_average_growth_rate(polygons_clust, labels, output):
    offset_frames = np.zeros((len(labels)),dtype=int)
    n=0
    for f in range(len(output)):
        instances = len(output[f])
        offset_=n
        for n in range(offset_,instances+offset_):
            offset_frames[n] = f+1
            n+=1
    agr = np.zeros((len(polygons_clust),2),dtype=float)
    for l in range(0,max(labels)+1): 
        end = max(offset_frames[labels==l])
        start = min(offset_frames[labels==l])
        agr[l,0] = l
        agr[l,1] = ((polygons_clust[l][end-1].area/polygons_clust[l][start-1].area)**(1/len(output))) - 1

    return agr
        
def get_area_std(polygons_clust, labels, pred_features_df):    
    area_std = np.zeros((len(polygons_clust),2),dtype=float)
    for l in range(0,max(labels)+1):
        area_std[l,0] = l
        area_std[l,1] = np.std(
            pred_features_df.loc[pred_features_df['Cell_label'] == l, 'Area']
        )
    
    return area_std
    
def get_position_std(polygons_clust, labels, pred_features_df):     
    position_std = np.zeros((len(polygons_clust),2),dtype=float)
    for l in range(0,max(labels)+1):  
        points_xy = np.array(
            pred_features_df.loc[
            pred_features_df['Cell_label'] == l, ('Position_X', 'Position_Y')
        ])
        dist_xy = []
        for i in range(len(points_xy)-1):
            dist_xy.append(get_distance(points_xy[i], points_xy[i+1]))
        position_std[l,0] = l
        position_std[l,1] = np.std(dist_xy) 
    
    return position_std