# -*- coding: utf-8 -*-
import numpy as np
import copy
from collections import Counter

def get_seg_performance(pred_s, gt_s, output, pipeline='maskrcnn'):
    '''
    For evaluating segmentation performance on benchmark data from 
    Yeast Image Toolkit.
    Compares the prediction results with ground truth data. 
    Parameters
    ----------
    pred_s : ndarray
        Segmentation prediction data array with int type.
    gt_s : ndarray
        Segmentation ground truth data array with int type.
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    pipeline : str, optional
        Can be set to 'maskrcnn' or 'YeaZ'. The default is 'maskrcnn'.
    Returns
    -------
    dict
        Performance indicators: true positives (tp), false positives (fp), 
        false negatives (fn), joined segmentations (join), 
        split segmentations (split).
    '''
    pred = copy.deepcopy(pred_s)
    gt = copy.deepcopy(gt_s)
    if pipeline == 'YeaZ':
        masks = [mask for mask in output]
    if pipeline == 'maskrcnn':    
        masks = [
            m for i in output for m in np.array(
                i['instances'].pred_masks.to('cpu'), dtype=int)
        ]
    r = np.zeros((len(pred),len(gt)))
    c1 = 0
    for pred_frame, mask in zip(pred[:,0], masks):
        c2 = 0
        for gt_frame, gt_x, gt_y in zip(gt[:,0], gt[:,2], gt[:,3]):
            if (pred_frame==gt_frame) & (mask[gt_y,gt_x]==1):
                r[c1,c2]=1
            c2+=1
        c1+=1
    
    join = [1 for i in range(len(r)) if r[i].sum()>1]
    split = [1 for i in range(len(r.T)) if r.T[i].sum()>1]      
    tp = [1 for i in range(len(r)) if r[i].sum()==1]
    tp = len(tp) + len(split)
    fp = [1 for i in range(len(r)) if r[i].sum()==0]
    fp = len(fp) + len(split)
    fn = len(gt) - tp
    
    return {
        "tp": tp, "fp": fp, "fn": fn, 
        "join": len(join), "split": len(split)
    }

def get_track_performance(pred_t, gt_t, output, pipeline='maskrcnn'):
    '''
    For evaluating tracking performance on benchmark data from 
    Yeast Image Toolkit.
    Compares the prediction results with ground truth data. 
    Parameters
    ----------
    pred_t : ndarray
        Tracking prediction data array with int type.
    gt_t : ndarray
        Tracking ground truth data array with int type.
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    pipeline : str, optional
        Can be set to 'maskrcnn' or 'YeaZ'. The default is 'maskrcnn'.
    Returns
    -------
    dict
        Performance indicators: true positives (tp), false positives (fp), 
        false negatives (fn), joined tracks (join), 
        split tracks (split).
    '''
    pred = copy.deepcopy(pred_t)
    gt = copy.deepcopy(gt_t)
    if pipeline == 'YeaZ':
        masks = [mask for mask in output]
    if pipeline == 'maskrcnn':    
        masks = [
            m for i in output for m in np.array(
                i['instances'].pred_masks.to('cpu'), dtype=int)
        ]
    r = np.zeros((len(pred),len(gt)))
    c1 = 0
    labels_matched = []
    for pred_frame, pred_lab, mask in zip(pred[:,0], pred[:,1], masks):
        c2 = 0
        for gt_frame, gt_lab, gt_x, gt_y in zip(
                gt[:,0], gt[:,1], gt[:,2], gt[:,3]
        ):
            if (pred_frame==gt_frame) & (mask[gt_y,gt_x]==1) & (pred_lab!=-1):
                r[c1,c2]=1
                labels_matched.append((pred_lab, gt_lab))
            c2+=1
        c1+=1
    
    #n_matched_tracks = len(Counter(labels_matched))            
    tracking_pairs = [i for i in Counter(labels_matched).keys()]
    tracking_pairs = [
        [i for i,j in tracking_pairs], [j for i,j in tracking_pairs]
    ]    
    join, c_0 = np.unique(tracking_pairs[0], return_counts=True)
    join = len(join[c_0>1])
    split, c_1 = np.unique(tracking_pairs[1], return_counts=True)
    split = len(split[c_1>1])
    
    #calculate by # of correct links
    n_matched_links = sum(np.array(list(Counter(labels_matched).values()))-1)
    pred_number_of_links = sum(np.array(list(Counter(pred[:,1]).values()))-1)
    gt_number_of_links = sum(np.array(list(Counter(gt[:,1]).values()))-1)
    fn = gt_number_of_links - n_matched_links
    fp = pred_number_of_links - n_matched_links
    
    return {
        "tp": n_matched_links, "fp": fp, "fn": fn,
        "join": join, "split": split
    }

def calculate_metrics(results, pred, gt):
    '''
    Calculate 4 standard performance metrics using performance indicators.
    Parameters
    ----------
    results : dict
        Contains 5 performance indicators: 
        true positives (tp), false positives (fp), false negatives (fn), 
        joined tracks (join), split tracks (split).
    pred : ndarray
        Array with segmentation or tracking predictions.
    gt : ndarray
        Array with segmentation or tracking ground truth.
    Returns
    -------
    dict
        Performance metrics outcomes for 
        F1-score, accuracy, precision and recall.
    '''
    precision = results["tp"]/(results["tp"]+results["fp"])
    recall = results["tp"]/(results["tp"]+results["fn"])
    accuracy = results["tp"]/(results["tp"]+results["fp"]+results["fn"])
    F = 2 * ((precision*recall) / (precision + recall))
    
    return {
        'F1-score': F, 'Accuracy': accuracy, 
        'Precision': precision, 'Recall': recall
    }