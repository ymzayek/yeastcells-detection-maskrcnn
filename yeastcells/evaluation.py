# -*- coding: utf-8 -*-
import numpy as np
import copy
from collections import Counter

def get_seg_performance(pred_s, gt_s, output, removed_fp = False, index=None):
    pred = copy.deepcopy(pred_s)
    gt = copy.deepcopy(gt_s)
    masks = [
        m for i in output for m in np.array(
            i['instances'].pred_masks.to('cpu'), dtype=int)
    ]
    if removed_fp is True:
        masks = [i for j, i in enumerate(masks) if j not in index]
        pred = np.array([i for j, i in enumerate(pred) if j not in index]) #remove false positives
    r = np.zeros((len(pred),len(gt)))
    c1 = 0
    for pred_frame, mask in zip(pred[:,0], masks):
        c2 = 0
        for gt_frame, gt_x, gt_y in zip(gt[:,0], gt[:,2], gt[:,3]):
            if (pred_frame==gt_frame) & (mask[gt_y,gt_x]==1):
                r[c1,c2]=1
            c2+=1
        c1+=1
    
    #calculate true positives, false positives, joined segmentations, and split segmentations
    tp = [1 for i in range(len(r)) if r[i].sum()==1]
    fp = [1 for i in range(len(r)) if r[i].sum()==0]
    join = [1 for i in range(len(r)) if r[i].sum()>1]
    split = [1 for i in range(len(r.T)) if r.T[i].sum()>1]      
    
    return pred, {
        "tp": len(tp), "fp": len(fp), "join": len(join), "split": len(split)
    }

def get_track_performance(pred_t, gt_t, output, removed_fp = False, index=None):
    pred = copy.deepcopy(pred_t)
    gt = copy.deepcopy(gt_t)
    masks = [
        m for i in output for m in np.array(
            i['instances'].pred_masks.to('cpu'), dtype=int)
    ]
    if removed_fp is True:
        masks = [i for j, i in enumerate(masks) if j not in index]
        pred = np.array([i for j, i in enumerate(pred) if j not in index]) #remove false positives
    r = np.zeros((len(pred),len(gt)))
    c1 = 0
    labels_matched = []
    for pred_frame, pred_lab, mask in zip(pred[:,0], pred[:,1], masks):
        c2 = 0
        for gt_frame, gt_lab, gt_x, gt_y in zip(gt[:,0], gt[:,1], gt[:,2], gt[:,3]):
            if (pred_frame==gt_frame) & (mask[gt_y,gt_x]==1) & (pred_lab != -1):
                r[c1,c2]=1
                labels_matched.append((pred_lab,gt_lab))
            c2+=1
        c1+=1
    
    #calculate true positives, false positives, joined tracks, and split tracks
    n_matched_tracks = len(Counter(labels_matched))            
    tracking_pairs = [i for i in Counter(labels_matched).keys()]
    tracking_pairs = [[i for i,j in tracking_pairs], [j for i,j in tracking_pairs]]    
    join,c_0 = np.unique(tracking_pairs[0], return_counts=True)
    join = len(join[c_0 > 1])
    split,c_1 = np.unique(tracking_pairs[1], return_counts=True)
    split = len(split[c_1 > 1])
    
    #calculate by # of correct links
    
    n_matched_links = sum(np.array(list(Counter(labels_matched).values())) - 1)
    pred_number_of_links = sum(np.array(list(Counter(pred[:,1]).values()))-1)
    gt_number_of_links = sum(np.array(list(Counter(gt[:,1]).values())) -1)
    
    return labels_matched, pred_number_of_links, gt_number_of_links, {
        "tp": n_matched_links, "fp": (pred_number_of_links-n_matched_links),
        "join": join, "split": split
    }

def calculate_metrics(results, pred, gt):
    precision = results["tp"]/len(pred)
    recall = results["tp"]/len(gt)
    F = 2 * ((precision*recall) / (precision + recall))
    
    return {'F1-score': F, 'Precision': precision, 'Recall': recall}