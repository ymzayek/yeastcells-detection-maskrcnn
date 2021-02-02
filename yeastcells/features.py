# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
from shapely.geometry import Polygon, Point

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

def get_seg_track(labels):
    segs = print('The number of segmentations is ' + str(len(labels)))
    tracks = print('The number of tracked cells is ' + str(len(np.unique(labels[labels>=0]))))
    
    return segs, tracks

def group(l, outputs):
    boundaries = np.cumsum([0] + [len(o['instances']) for o in outputs])
    return [
        l[a:b]
        for a, b in zip(boundaries, boundaries[1:])
    ]

def polygons_per_cluster(labels, contours, output, include_noise=False):
    if include_noise is True:
        polygons = {l: {}  for l in set(labels) if l>=-1}
    else:
        polygons = {l: {}  for l in set(labels) if l>=0}
    for z, (labels_, x, y) in enumerate(zip(group(labels, output), *contours)):
        for label, x_, y_ in zip(labels_, x, y):
            if label < 0: 
                continue
            p = np.concatenate(
                (x_[:, None], y_[:, None]),
                axis=1
            )
            shape = Polygon(p) if len(p) >= 3 else Point(x_.mean(), y_.mean())
            polygons[label][z] = shape
            
    return polygons

def get_area_and_growth_rate(polygons, labels, pred_s):
    areas =np.zeros((len(labels)),dtype=float)
    n=0
    for lab, frame in zip(pred_s[:,1], pred_s[:,0]):
        areas[n] = polygons[lab][frame-1].area
        n+=1
    g_rate = np.zeros((len(polygons)),dtype=float)
    for l in polygons.keys():
        end = max(pred_s[:,0][labels==l])
        start = min(pred_s[:,0][labels==l])
        g_rate[l] = (
            polygons[l][end-1].area-polygons[l][start-1].area)/(
            polygons[l][start-1].area+0.00001
        ) # -1 because pred_s starts at 1 and polygons starts at 0
        g_rate = abs(g_rate)
    #assign growth rate per segmented instance    
    g_rate_ = labels.astype(float).copy() 
    for g in range(len(g_rate)):
        g_rate_[g_rate_==g] = g_rate[g]    
        
    return areas, g_rate_   