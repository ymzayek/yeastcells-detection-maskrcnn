# -*- coding: utf-8 -*-
import pandas as pd
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


import warnings
def calculate_metrics(results, *args):
    '''
    Calculate 4 standard performance metrics using performance indicators.
    Parameters
    ----------
    results : dict
        Contains at least 3 performance indicators: 
        true positives (tp), false positives (fp), false negatives (fn)
    Returns
    -------
    dict
        Performance metrics outcomes for 
        F1-score, accuracy, precision and recall.
    '''
    if len(args) > 0:
        warnings.warn('Using deprecated arguments in calculate_metrics, they will be removed.')
        
    precision = results["tp"]/(results["tp"]+results["fp"])
    recall = results["tp"]/(results["tp"]+results["fn"])
    accuracy = results["tp"]/(results["tp"]+results["fp"]+results["fn"])
    F = 2 * ((precision*recall) / (precision + recall))
    
    return {
        'F1-score': F, 'Accuracy': accuracy, 
        'Precision': precision, 'Recall': recall
    }


def match_detections_and_ground_truths(ground_truth, detections, masks):
  """Considering ground truth coordinates versus segmentation masks,
  yields tuples (ground_truth_index, detection_index) for every
  ground truth sample and detection in the same frame, such that the
  ground truth (x, y) coordinate matches the mask:

      # dataframe location, NOT index
      `masks[detection_location, y, x] == True`
  
  `detections` and `masks` must have the same length, as each item of mask is a
  height x width segmentation mask for that detection.

  Note that masks indices, should match detection locations.

  `ground_truths`, `detections` must be dataframes with columns
  [`frame`, `x`, `y`] and [`frame`, `mask`] respecitvely.
  
  Their indices must be unique.

  The `mask` column must point to the index of the mask for that detection,
  usually this is incremental from 0.
  """
  matches = []
  # iterate through grount truth and detected cells per time frame
  for frame, frame_ground_truths in ground_truth.groupby('frame'):
    frame_detections = detections[detections['frame'] == frame]
    frame_masks = masks[frame_detections['mask'].values]
    x, y = np.round(frame_ground_truths[['x', 'y']].values).astype(int).T
    mask_values_at_yx = frame_masks[:, y, x]
    found = mask_values_at_yx.sum(0) > 0
    detection_indices, ground_truth_indices = np.where(mask_values_at_yx)
    found = found[ground_truth_indices]
    matches.extend(zip(
        frame_ground_truths.index[ground_truth_indices[found]],
        frame_detections.index[detection_indices[found]]))
  return pd.DataFrame(matches, columns=['ground truth index', 'detection index'])


def get_segmention_metrics(ground_truth, detections, masks):
  """For the segmentation task, returns how many true positives and true/false
  positives as a dictionary including how many ground truths were detected by
  the same mask (merged).
  
  Arguments the same as `match_detections_and_ground_truths`"""
  matches = match_detections_and_ground_truths(ground_truth, detections, masks)

  # We defined true positives to be those ground truths that are picked up as
  # a detection, but only if this detection has only one ground truths
  # assinged to it.

  # figuring out which detections have more than one ground truth, they 'join'
  # ground truths together, and filtering out matches on this detection.
  detection_joining_gt = matches.groupby('detection index').count() > 1
  detection_joining_gt = detection_joining_gt.index[detection_joining_gt['ground truth index']]
  unjoined_matches = matches[~matches['detection index'].isin(detection_joining_gt)]
  # then the amount of true positives, equals the amount of ground truths that
  # still have a detection assigned.
  tp = len(unjoined_matches['ground truth index'].unique())

  # Anything that
  split = int((unjoined_matches.groupby('ground truth index').count() - 1).sum())
  metrics = {
    'tp': tp,
    'fp': len(set(detections.index) - set(unjoined_matches['detection index'])) + split,
    'fn': len(set(ground_truth.index) - set(unjoined_matches['ground truth index'])),
    'join': len(detection_joining_gt), 'split': split,
  }
  return metrics


def compare_links(a, b, mapping):
  """
  `a` and `b` dataframes with columns `frame` and `cell`.

  For every pair of rows in `a`, where the second is one
  frame ahead of the first, and where both have the same `cell` value, count as:

  over matching: if the first occurs in several such pairs (except for the first found),
  unmapped: if both can't be mapped as per indices to rows in b,
  unmatched: if the matched rows in `b` have different `cell` values,
  true: if the matched rows in `b` have the `cell` values,
  false counts all pairs that didn't fall into true, e.g. this sums overmatching,
  unmapped and unmatched."""
  true, over_matching, unmapped, unmatched = 0, 0, 0, 0
  for cell, rows in a.groupby('cell'):
    for index0, frame0 in rows['frame'].iteritems():
      matched = rows.index[(rows['frame'] - frame0) == 1]
      # should be 1 match, others are assumed falses.
      over_matching += len(matched[1:])
      for index1 in matched[:1]: # loops once or not at all
        if index0 not in mapping or index1 not in mapping:
          unmapped += 1
        elif b['cell'][mapping[index0]] == b['cell'][mapping[index1]]:
          assert mapping[index0] != mapping[index1], (
              "Uncanny, different ground truths were mapped to the same detection")
          true += 1
        else:
          unmatched += 1
  return {'true': true, 'false': over_matching+unmapped+unmatched,
          'over matching': over_matching, 'unmapped': unmapped,
          'unmatched': unmatched}


def get_tracking_metrics(ground_truth, detections, masks):
  """For the tracking task, returns how many true positives and true/false
  positives as a dictionary including how many ground truths were detected by
  the same mask (merged).
  
  Arguments the same as `match_detections_and_ground_truths`"""
  matches = pd.DataFrame(
      match_detections_and_ground_truths(ground_truth, detections, masks),
      columns=['ground truth index', 'detection index'])
  gt_to_det = {gt: rows['detection index'].values[0] for gt, rows in matches.groupby('ground truth index') if len(rows) == 1}
  det_to_gt = {det: rows['ground truth index'].values[0] for det, rows in matches.groupby('detection index') if len(rows) == 1}

  comparison_gt = compare_links(ground_truth, detections, gt_to_det)
  comparison_det = compare_links(detections, ground_truth, det_to_gt)
  assert comparison_det['true'] == comparison_det['true'], (
      "Uncanny, different links matches going from ground truth to "
      "detections as vice versa. This shouldn't happen"
  )
  assert comparison_gt['over matching'] == 0, (
      "Ground truth has marked multiple cells in one frame as the same")
  return {'tp': comparison_gt['true'], 'fp': comparison_det['false'],
          'fn': comparison_gt['false'],
          # when a cell was tracked multiple times in a frame.
          'over matching': comparison_det['over matching'],
          # also specify propagated segmentation errors
          'segmentation fn': comparison_gt['unmapped'],
          'segmentation fp': comparison_det['unmapped']}
