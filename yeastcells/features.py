# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
from shapely.geometry import Polygon, Point
import math
from .clustering import existance_vectors

def extract_contours(output):
    '''
    Parameters
    ----------
    output : dict
        Predictor output from the detecron2 model.
    Returns
    -------
    x : TYPE
        The x coordinates of the contour points for each segmented cell.
    y : TYPE
        The y coordinates of the contour points for each segmented cell.
    '''
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
    '''
    Parameters
    ----------
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    labels : TYPE
        Tracking labels of individual segmented cells.
    Returns
    -------
    centroids : TYPE
        The coordinates of the centroids of the segmented cells.
    '''
    centroids = np.zeros(((len(labels),3))).astype(int)    
    centroids[:,0] = labels
    centroids[:,1] = coordinates[:,2] #x
    centroids[:,2] = coordinates[:,1] #y
    
    return centroids

def get_instance_numbers(output):
    '''
    Parameters
    ----------
    output : dict
        Predictor output from the detecron2 model.
    Returns
    -------
    inst_num : TYPE
        Each segmented cell gets a unique number per frame. 
        This is not the same as the tracking labels since 
        the numbers are unique within a frame but are not 
        consistent for one cell across frames. It serves
        more as a count of cells in each frame.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    '''
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

def get_seg_track(labels, output, frame=None):
    if frame is None:
        segs = print('The number of segmentations is ' + str(len(labels)))
        tracks = print(
            'The number of tracked cells is ' 
            + str(len(np.unique(labels[labels>=0])))
        )
    else:
        grouped = group(labels, output) # group labels by frame
        segs = print(
            f'The number of segmentations in frame {frame} is ' 
            + str(len(grouped[frame-1]))
        )
        tracks = print(
            f'The number of tracked cells in frame {frame} is ' 
            + str(len(np.unique(grouped[frame-1][grouped[frame-1]>=0])))
        )
    
    return segs, tracks

def track_len(labels, label = 0):
    counts = Counter(labels)
    return counts[label]

def polygons_per_instance(labels, contours, output):
    polygons_inst = {l: {}  for l in set(labels)}
    for z, (labels_, x, y) in enumerate(
            zip(group(labels, output), *contours)
    ):
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
    for z, (labels_, x, y) in enumerate(
            zip(group(labels, output), *contours)
    ):
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

def get_masks(output):
    masks = [
        m for i in output for m in np.array(
        i['instances'].pred_masks.to('cpu'), dtype=int
    )]
    
    return masks

def get_frame_offsets(labels, output):
    o = list(map(existance_vectors, output))
    frame_offsets = np.zeros((len(labels)),dtype=int)
    n=0
    for f in range(len(o)):
        instances = len(o[f])
        offset_=n
        for n in range(offset_,instances+offset_):
            frame_offsets[n] = f+1
            n+=1
    
    return frame_offsets

def get_area(polygons_inst, masks, labels, output): #make sure polygons include noise
    frame_offsets = get_frame_offsets(labels, output)    
    poly_area =np.zeros((len(labels)),dtype=float)
    mask_area =np.zeros((len(labels)),dtype=float)
    n=0
    for lab, frame in zip(labels, frame_offsets):
        poly_area[n] = polygons_inst[lab][frame-1].area
        mask_area[n] = masks[n].sum()
        n+=1
        
    return poly_area,  mask_area

def get_average_growth_rate(polygons_clust, labels, output):
    frame_offsets = get_frame_offsets(labels, output)
    agr = np.zeros((len(polygons_clust),2),dtype=float)
    for l in range(0,max(labels)+1): 
        end = max(frame_offsets[labels==l])
        start = min(frame_offsets[labels==l])
        agr[l,0] = l
        agr[l,1] = (
            (polygons_clust[l][end-1].area/polygons_clust[l][start-1].area
             )**(1/len(output))
        ) - 1

    return agr
        
def get_area_std(polygons_clust, labels, pred_features_df):    
    area_std = np.zeros((len(polygons_clust),2),dtype=float)
    for l in range(0,max(labels)+1):
        area_std[l,0] = l
        area_std[l,1] = np.std(
            pred_features_df.loc[
            pred_features_df['Cell_label'] == l, 'Poly_Area(pxl)']
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

def get_pixel_intensity(masks, output, im):
    masks_ = group(masks, output)
    pi = [
        np.sum(im[frame][masks_[frame][i]==1]) 
        for frame in range(len(im)) for i in range(len(masks_[frame]))
    ]
    
    return pi