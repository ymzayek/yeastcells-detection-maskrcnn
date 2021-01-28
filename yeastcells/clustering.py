# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
import random
from collections import Counter
import lapjv

def existance_vectors(output):
    o = np.array(output['instances'].pred_masks.to('cpu'))
    return o.reshape(o.shape[0], -1).astype(np.float32)

def calc_iou(outputs0, outputs1):
    overlaps = np.dot(outputs0, outputs1.T)
    szi, szj = outputs0.sum(1), outputs1.sum(1)
    summed = (szi[:, None] + szj[None])
    iou = overlaps / (summed - overlaps)
    
    return iou

def get_distances(outputs, dmax=5, progress=False):
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

def cluster_cells(outputs, dmax=5, min_samples=3, eps=0.6, progress=False): 
    distances = get_distances(outputs, dmax=dmax, progress=progress)
    tqdm_ = tqdm if progress else (lambda x: x)
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters.fit(distances)     

    coordinates = np.array([
        (t, ) + tuple(map(np.mean, np.where(mask)))
        for t, o in enumerate(tqdm_(outputs))
        for mask in o['instances'].pred_masks.to('cpu')
    ])
    
    return clusters.labels_, coordinates

def cluster_cells_grid(outputs, param_grid, dmax=5, progress=False):
    distances = get_distances(outputs, dmax=dmax, progress=progress)
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
        for t, o in enumerate(tqdm_(outputs))
        for mask in o['instances'].pred_masks.to('cpu')
    ])

    return clusters_labels, dbscan_params, coordinates

def cluster_cells_random(outputs, param_grid, dmax=5, evals=20, progress=False):
    distances = get_distances(outputs, dmax=dmax, progress=progress)
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
        for t, o in enumerate(tqdm_(outputs))
        for mask in o['instances'].pred_masks.to('cpu')
    ])

    return clusters_labels, dbscan_params, coordinates

def compare_clusters(gt, pred, coordinates): #loop with clusters_labels output #YM
    if -1 in gt:
        gt=gt+1
    if -1 in pred:
        pred=pred+1
    labels = set(gt)|set(pred) 
    counts=Counter(zip(gt, pred))
    cost_matrix=np.zeros([max(labels)+1,max(labels)+1])
    for (i, j), cost in counts.items(): cost_matrix[i, j] = -cost
    row,col,_=lapjv(cost_matrix)
    overlap = gt[(col[pred]==gt)]
    idx_overlap = np.where(col[pred]==gt)[-1] 
    coord_overlap = coordinates[idx_overlap,:]
    no_overlap = gt[(col[pred]!=gt)] 
    idx_no_overlap = np.where(col[pred]!=gt)[-1]
    coord_no_overlap = coordinates[idx_no_overlap,:]
    overlap_mean=(col[pred]==gt).mean()*100

    return overlap_mean, {
        'overlap': overlap, 'coordinates_overlap': coord_overlap, 
        'no_overlap': no_overlap, 'coordinates_no_overlap': coord_no_overlap
    }
