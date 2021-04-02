# -*- coding: utf-8 -*-
from sklearn.exceptions import EfficiencyWarning
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cluster import DBSCAN
import warnings

import torch


def get_iou_torch(left, right, device='cpu'):
    """Returns pairwise iou similarity matrix between left and right using
    torch on device"""
    if len(left) * len(right) == 0:
        return np.zeros((len(left), len(right)), float)

    left_ = torch.tensor(left.reshape(len(left), -1), device=device).float()
    right_ = torch.tensor(right.reshape(len(right), -1), device=device).float()

    overlaps = torch.matmul(left_, right_.T)
    summed = left_.sum(1)[:, None] + right_.sum(1)[None]
    iou = overlaps / (summed - overlaps)

    return iou.to('cpu').numpy()


def get_iou_sparse(left, right, device=None):
    """Returns pairwise iou similarity matrix between left and right using
    sparse matrices, device is ignored"""
    left_ = sparse.csc_matrix(left.reshape(len(left), -1)).astype(int)
    right_ = sparse.csc_matrix(right.reshape(len(right), -1)).astype(int)
    overlaps = left_ @ right_.T
    summed = left_.sum(1) + right_.sum(1).T
    return overlaps / (summed - overlaps)


get_iou = get_iou_torch


def get_distances(detections, masks, dmax=5, device='cpu'):
    rows, cols, values = [], [], []
    for frame, detections0 in detections.groupby('frame'):
        detections1 = detections[detections['frame'].isin(np.arange(frame + 1, frame + dmax + 1))]
        left = masks[detections0['mask'].values]
        right = masks[detections1['mask'].values]
        iou = get_iou(left, right, device=device) # similarity
        iou = 1 - iou # distance
        rows_, cols_ = np.where(iou > 0)
        values_ = iou[rows_, cols_]

        rows.extend(detections0['mask'].iloc[rows_])
        cols.extend(detections1['mask'].iloc[cols_])
        cols.extend(detections0['mask'].iloc[rows_])
        rows.extend(detections1['mask'].iloc[cols_])
        values.extend(values_)
        values.extend(values_)
    distances = coo_matrix((values, (rows, cols)), shape=(len(masks), len(masks)))

    # correct for masks that did not appear in the detections, for example due to post filtering.
    return distances.tocsc()[detections['mask'], :][:, detections['mask']].tocoo()


def track_cells(detections, masks, dmax=5, min_samples=3, eps=0.6, device='cpu', distances=None):
    '''
    Configure and run DBSCAN clustering algorithm to find cell tracks.
    Parameters
    ----------
    detections: dataframe
        dataframe containing frame and mask columns as provided by segmentation
    masks : dict
        segmentation masks aligned with mask column
    dmax : int, optional
        Set the maximum frame distance to look ahead and behind to calculate 
        the interframe IOUs between the cells. The default is 5.
    min_samples : int, optional
        Set minimum samples hyperparameter of DBSCAN.. The default is 3. 
    eps : float, optional
        Set epsilon hyperparameter of DBSCAN. The default is 0.6.
    device:
        cuda device to do IoU distance calculation on
    Returns
    -------
    dataframe
        segmentation dataframe with added cell labels in column cell, -1 for unteacked outliers
    '''
    if distances is None:
        distances = get_distances(detections, masks, dmax=dmax, device=device)
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=EfficiencyWarning)
        clusters.fit(distances)

    detections.loc[:, 'cell'] = clusters.labels_
    # rearrange to convenient column order.
    return detections[['frame', 'cell', 'mask', 'x', 'y'] +
                      list(set(detections.columns) -
                           {'frame', 'cell', 'mask', 'x', 'y'})].copy()


# def cluster_cells_grid(output, param_grid, dmax=5, progress=False):
#     '''
#     Adds an option to run a grid search to the cluster_cells function
#     for hyperparameter tuning of epsilon and min_samples.
#     Parameters
#     ----------
#     output : dict
#         Detecron2 predictor output from the detecron2 Mask R-CNN model.
#     param_grid : dict
#         Contains list of a range of values for hyperparameters
#         'eps' and 'min_samples'.
#     dmax : int, optional
#         The maximum frame distance to look ahead and behind to calculate
#         the interframe IOUs between the cells. The default is 5.
#     progress : bool, optional
#         The default is False.
#     Returns
#     -------
#     clusters_labels : list
#         Outcomes of DBSCAN clustering for each evaluation.
#     dbscan_params : list
#         Parameters corresponding to each evaluation.
#     coordinates : ndarray
#         Coordinates of centroid of individual instances with 2 dimensions
#         (labels, ([time, Y, X])).
#     '''
#     distances = get_distances(output, dmax=dmax, progress=progress)
#     tqdm_ = tqdm if progress else (lambda x: x)
#     clusters_labels = []
#     dbscan_params = []
#     for eps_val in param_grid['eps']:
#         for sample in param_grid['min_samples']:
#             clusters = DBSCAN(**param_grid, metric='precomputed')
#             clusters = clusters.fit(distances)
#             clusters_labels.append(clusters.labels_)
#             tmp = np.array(np.unique(clusters.labels_, return_counts=True))
#             n_clusters = len(tmp[0,:])
#             dbscan_params.append([eps_val,sample,n_clusters])
#
#     coordinates = np.array([
#         (t, ) + tuple(map(np.mean, np.where(mask)))
#         for t, o in enumerate(tqdm_(output))
#         for mask in o['instances'].pred_masks.to('cpu')
#     ])
#
#     return clusters_labels, dbscan_params, coordinates
#
# def cluster_cells_random(output, param_grid, dmax=5, evals=20, progress=False):
#     '''
#     Adds an option to run a randomized search to the cluster_cells function.
#     Parameters
#     ----------
#     output : dict
#         Detecron2 predictor output from the detecron2 Mask R-CNN model.
#     param_grid : dict
#         Contains list of a range of values for hyperparameters
#         'eps' and 'min_samples'.
#     dmax : int, optional
#         The maximum frame distance to look ahead and behind to calculate
#         the interframe IOUs between the cells. The default is 5.
#     evals : int, optional
#         Number of evaluations to run by randomly choosing hyperparameter
#         settings from the parameter grid. The default is 20.
#     progress : bool, optional
#         The default is False.
#     Returns
#     -------
#     clusters_labels : list
#         Outcomes of DBSCAN clustering for each evaluation.
#     dbscan_params : list
#         Parameters corresponding to each evaluation.
#     coordinates : ndarray
#         Coordinates of centroid of individual instances with 2 dimensions
#         (labels, ([time, Y, X])).
#     '''
#     distances = get_distances(output, dmax=dmax, progress=progress)
#     tqdm_ = tqdm if progress else (lambda x: x)
#     clusters_labels = []
#     dbscan_params = []
#     for i in range(evals):
#         hyperparameters = {
#             k: random.sample(v, 1)[0] for k, v in param_grid.items()
#         }
#         clusters = DBSCAN(**hyperparameters, metric='precomputed')
#         clusters = clusters.fit(distances)
#         clusters_labels.append(clusters.labels_)
#         tmp = np.array(np.unique(clusters.labels_, return_counts=True))
#         n_clusters = len(tmp[0,:])
#         dbscan_params.append(
#             [hyperparameters['eps'],hyperparameters['min_samples'],n_clusters]
#         )
#
#     coordinates = np.array([
#         (t, ) + tuple(map(np.mean, np.where(mask)))
#         for t, o in enumerate(tqdm_(output))
#         for mask in o['instances'].pred_masks.to('cpu')
#     ])
#
#     return clusters_labels, dbscan_params, coordinates
