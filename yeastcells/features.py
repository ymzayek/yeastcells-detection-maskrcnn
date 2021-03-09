# -*- coding: utf-8 -*-
import numpy as np
import cv2
from collections import Counter
import math
from .clustering import existance_vectors

def extract_contours(output):
    '''
    Extract the coordinates of the contour points for each segmented cell.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    Returns
    -------
    x : list
        The x coordinates of the contour points for each segmented cell
        in a nested list.
    y : list
        The y coordinates of the contour points for each segmented cell
        in a nested list.
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

def get_centroids(labels, coordinates):
    '''
    Extract the centroid coordinates for each segmented cell.
    Parameters
    ----------
    labels : ndarray
        Tracking labels of individual cells.  
    coordinates : ndarray
        2D array of centroid coordinates individual cells
        (labels, ([time, Y, X])). 
    Returns
    -------
    centroids : ndarray
        An array with the cell labels and the coordinates of their centroids.
    '''
    centroids = np.zeros(((len(labels),3))).astype(int)    
    centroids[:,0] = labels
    centroids[:,1] = coordinates[:,2] #x
    centroids[:,2] = coordinates[:,1] #y
    
    return centroids

def get_instance_numbers(output):
    '''
    Each segmented cell gets a unique number per frame. 
    This is not the same as the tracking labels since 
    the numbers are unique within a frame but are not 
    consistent for one cell across frames. It serves
    more as a count of cells in each frame.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    Returns
    -------
    inst_num : ndarray
        Array of unique cell number assigned to each cell within a frame 
        containing data with int type.
    coordinates : ndarray
        2D array of centroid coordinates individual cells
        (labels, ([time, Y, X])). ym is this needed?
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

def group(labels, output):
    '''
    Parameters
    ----------
    labels : ndarray
        Tracking labels of individual cells.
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    Returns
    -------
    list
        A nested-list of labels grouped by frame.
    '''
    boundaries = np.cumsum([0] + [len(o['instances']) for o in output])
    return [
        labels[a:b]
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

def get_distance(p1, p2): 
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    
    return distance

def get_masks(output):
    masks = [
        m for i in output for m in np.array(
        i['instances'].pred_masks.to('cpu'), dtype=int
    )]
    
    return masks

def get_areas(masks, labels):
    mask_areas =np.zeros((len(labels)),dtype=float)
    n=0
    for lab in labels:
        mask_areas[n] = masks[n].sum()
        n+=1
        
    return mask_areas

def get_average_growth_rate(mask_areas, labels, output):
    agr = np.zeros((len(np.unique(labels)),2),dtype=float)
    for l in range(0,max(labels)+1): 
        idx = np.where(labels == l)[0]
        areas = mask_areas[idx]
        agr[l,0] = l
        agr[l,1] = ((areas[-1]/areas[0])**(1/len(output))) - 1

    return agr
        
def get_area_std(labels, pred_df):    
    area_std = np.zeros((len(np.unique(labels)),2),dtype=float)
    for l in range(0,max(labels)+1):
        area_std[l,0] = l
        area_std[l,1] = np.std(
            pred_df.loc[
            pred_df['Cell_label'] == l, 'Mask_Area(pxl)']
        )
    
    return area_std
    
def get_position_std(labels, pred_df):     
    position_std = np.zeros((len(np.unique(labels)),2),dtype=float)
    for l in range(0,max(labels)+1):  
        points_xy = np.array(
            pred_df.loc[
            pred_df['Cell_label'] == l, ('Position_X', 'Position_Y')
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

def extract_labels(masks):
    '''
    Extract labels from masks with multiple cells per mask.
    Parameters
    ----------
    masks : ndarray
        3D mask array (frames, length, width) containing data of int type. ym
    Returns
    -------
    labels : ndarray
        Tracking labels of individual cells.
    labels_grouped : list
        Grouping of labels in a nested-list by frame.
    coordinates : ndarray
        2D array of centroid coordinates individual cells
        (labels, ([time, Y, X])).  
    instances : ndarray
        2D mask array for each labeled cell. ym
    '''
    labels_grouped=[]            
    for i in range(len(masks)):
        label = np.unique(masks[i])[1:]
        labels_grouped.append(label)
    instance = np.cumsum([0] + [len(l) for l in labels_grouped])
    instances = np.zeros(
        (instance.max(), np.array(np.shape(masks[i][0])).item(),np.array(
            np.shape(masks[i][1])).item())
    )
    i = 0
    for mask,label in zip(masks,labels_grouped):
        for n in label:
            instances[i] = (mask==n).astype(int)
            i+=1
    coordinates = np.array(
        [(t, ) + tuple(map(np.mean, np.where(m == 1.))) for t,m in enumerate(
            instances)]
    )  
    labels = np.hstack(labels_grouped)
    offset_frames = np.zeros((len(labels)),dtype=int)
    n=0
    for f in range(len(labels_grouped)):
        tmp = len(labels_grouped[f])
        offset_=n
        for n in range(offset_,tmp+offset_):
            offset_frames[n] = f
            n+=1
    coordinates[:,0] = offset_frames
    
    return labels, labels_grouped, coordinates, instances 