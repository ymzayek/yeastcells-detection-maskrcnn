# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
import random

def existance_vectors(output):
    '''
    Converts image matrix into 1D vector for each segmented cell.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    Returns
    -------
    ndarray
        Array of vectors containing data of float type.
    '''
    o = np.array(output['instances'].pred_masks.to('cpu'))
    return o.reshape(o.shape[0], -1).astype(np.float32)

def calc_iou(outputs0, outputs1):
    '''
    Calculate the intersection-over-union (IOU) between cells 
    in a pair of frames.
    Parameters
    ----------
    outputs0 : ndarray
        Array of vectors with float type.
    outputs1 : ndarray
        Array of vectors with float type.
    Returns
    -------
    iou : ndarray
        Array of IOU values with float type.
    '''
    overlaps = np.dot(outputs0, outputs1.T)
    szi, szj = outputs0.sum(1), outputs1.sum(1)
    summed = (szi[:, None] + szj[None])
    iou = overlaps / (summed - overlaps)
    
    return iou

def get_distances(outputs, dmax=5, progress=False):
    '''
    Build a sparse matrix of distances based on calculating 
    the IOUs between all cell instances from one frame to another. 
    This distance matrix is passed to DBSCAN for clustering 
    the cells into tracks.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    dmax : int, optional
        The maximum frame distance to look ahead and behind to calculate 
        the interframe IOUs between the cells. The default is 5.
    progress : bool, optional
        The default is False.
    Returns
    -------
    coo_matrix
        2D sparse matrix of distances in COOrdinate format.
    '''
    tqdm_ = tqdm if progress else (lambda x: x)
    offsets = np.cumsum([0] + [len(o['instances']) for o in outputs])
    outputs = list(map(existance_vectors, tqdm_(outputs)))
    rows = []
    cols = []
    values = []

    for i in tqdm_(range(len(outputs))):
        for j in range(max(0, i-dmax), min(len(outputs), i+dmax+1)):
            if i == j:
                continue
            iou = calc_iou(outputs[i], outputs[j]) #similarity
            iou = (1 - iou) #distance
        
            rows_, cols_ = np.where(iou > 0)
            rows.extend(rows_ + offsets[i])
            cols.extend(cols_ + offsets[j])
            values.extend(iou[rows_, cols_])

    return (coo_matrix((values, (rows, cols)), shape=(offsets[-1], offsets[-1])))

def cluster_cells(segmentation, dmax=5, min_samples=3, eps=0.6, progress=False): 
    '''
    Configure and run DBSCAN clustering algorithm to find cell tracks.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    dmax : int, optional
        Set the maximum frame distance to look ahead and behind to calculate 
        the interframe IOUs between the cells. The default is 5.
    min_samples : int, optional
        Set minimum samples hyperparameter of DBSCAN.. The default is 3. 
    eps : float, optional
        Set epsilon hyperparameter of DBSCAN. The default is 0.6. 
    progress : bool, optional
        The default is False.
    Returns
    -------
    ndarray
        Tracking labels of individual segmented cells.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    '''
    distances = get_distances(segmentation, dmax=dmax, progress=progress)
    tqdm_ = tqdm if progress else (lambda x: x)
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters.fit(distances)     

    frame, y, x = np.array([
        (t, ) + tuple(map(np.mean, np.where(mask)))
        for t, o in enumerate(tqdm_(segmentation))
        for mask in o['instances'].pred_masks.to('cpu')
    ]).T
    
    return pd.DataFrame({'frame': frame, 'cell': clusters.labels_, 'x': x, 'y': y})

def cluster_cells_grid(output, param_grid, dmax=5, progress=False):
    '''
    Adds an option to run a grid search to the cluster_cells function 
    for hyperparameter tuning of epsilon and min_samples.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    param_grid : dict
        Contains list of a range of values for hyperparameters 
        'eps' and 'min_samples'.
    dmax : int, optional
        The maximum frame distance to look ahead and behind to calculate 
        the interframe IOUs between the cells. The default is 5.
    progress : bool, optional
        The default is False.
    Returns
    -------
    clusters_labels : list
        Outcomes of DBSCAN clustering for each evaluation.
    dbscan_params : list
        Parameters corresponding to each evaluation.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    '''
    distances = get_distances(output, dmax=dmax, progress=progress)
    tqdm_ = tqdm if progress else (lambda x: x)
    clusters_labels = []
    dbscan_params = []  
    for eps_val in param_grid['eps']:
        for sample in param_grid['min_samples']:
            clusters = DBSCAN(**param_grid, metric='precomputed')
            clusters = clusters.fit(distances)
            clusters_labels.append(clusters.labels_)
            tmp = np.array(np.unique(clusters.labels_, return_counts=True))
            n_clusters = len(tmp[0,:])
            dbscan_params.append([eps_val,sample,n_clusters])
            
    coordinates = np.array([
        (t, ) + tuple(map(np.mean, np.where(mask)))
        for t, o in enumerate(tqdm_(output))
        for mask in o['instances'].pred_masks.to('cpu')
    ])

    return clusters_labels, dbscan_params, coordinates

def cluster_cells_random(output, param_grid, dmax=5, evals=20, progress=False):
    '''
    Adds an option to run a randomized search to the cluster_cells function.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    param_grid : dict
        Contains list of a range of values for hyperparameters 
        'eps' and 'min_samples'.
    dmax : int, optional
        The maximum frame distance to look ahead and behind to calculate 
        the interframe IOUs between the cells. The default is 5.
    evals : int, optional
        Number of evaluations to run by randomly choosing hyperparameter 
        settings from the parameter grid. The default is 20.
    progress : bool, optional
        The default is False.
    Returns
    -------
    clusters_labels : list
        Outcomes of DBSCAN clustering for each evaluation.
    dbscan_params : list
        Parameters corresponding to each evaluation.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    '''
    distances = get_distances(output, dmax=dmax, progress=progress)
    tqdm_ = tqdm if progress else (lambda x: x)
    clusters_labels = []
    dbscan_params = []  
    for i in range(evals):  
        hyperparameters = {
            k: random.sample(v, 1)[0] for k, v in param_grid.items()
        }    
        clusters = DBSCAN(**hyperparameters, metric='precomputed')
        clusters = clusters.fit(distances)
        clusters_labels.append(clusters.labels_)
        tmp = np.array(np.unique(clusters.labels_, return_counts=True))
        n_clusters = len(tmp[0,:])
        dbscan_params.append(
            [hyperparameters['eps'],hyperparameters['min_samples'],n_clusters]
        )
            
    coordinates = np.array([
        (t, ) + tuple(map(np.mean, np.where(mask)))
        for t, o in enumerate(tqdm_(output))
        for mask in o['instances'].pred_masks.to('cpu')
    ])

    return clusters_labels, dbscan_params, coordinates
