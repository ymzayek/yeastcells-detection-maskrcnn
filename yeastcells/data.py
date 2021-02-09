# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
import os
from .clustering import existance_vectors

def load_data(path):
    '''
    Parameters
    ----------
    path : str
        Path to tiff image files.
    Returns
    -------
    fns : ndarray
        All filenames ending with .tif in the path.
    '''
    fns = [
        f'{path}/{fn}' 
        for fn in os.listdir(f'{path}/') 
        if fn.endswith('.tif')
    ]
    fns= np.sort(fns)
    
    return fns

def read_image(fn):
    '''
    Parameters
    ----------
    fn : str
        Path to one multi-image tiff file.
    Returns
    -------
    image : ndarray
        An array with 4 dimensions (frames, length, width, channels).
    '''
    image = imread(fn)  
    if image.ndim==4:  
        return (
            (image / 65535 * 255)[:, ..., :1] * [[[1, 1, 1]]]
        ).astype(np.uint8)
    if image.ndim==3:  
        return (
            (image / image.max() * 255)[:, ..., None] * [[[1, 1, 1]]]
        ).astype(np.uint8)
    
def read_images_cat(fns): #for reading multiple single-image tiffs and concatenating them
    '''
    Parameters
    ----------
    fns : str
        Path to multiple tiff files.
    Returns
    -------
    image : ndarray
        An array with 4 dimensions (frames, length, width, channels).
    '''
    image=[]
    image=[imread(i) for i in list(fns)]
    image= np.array(image)
    
    return (
        (image / image.max() * 255)[:, ..., None] * [[[1, 1, 1]]]
    ).astype(np.uint8) 

def read_tiff_mask(path):
    '''
    Parameters
    ----------
    path : str
        Path to one multi-image tiff mask.
    Returns
    -------
    masks : ndarray
        A mask array with 4 dimensions (frames, length, width, channels).
    '''
    img = Image.open(path)
    masks = []
    for i in range(img.n_frames):
        img.seek(i)
        masks.append(np.array(img))
    masks = np.array(masks)
    return masks

def extract_labels(masks, progress=False):
    '''
    Parameters
    ----------
    masks : ndarray
        A mask array with 4 dimensions (frames, length, width, channels).
    Returns
    -------
    labels_grouped : list
        Grouping of labels in a list by frame.
    labels : ndarray
        Tracking labels of individual instances.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions (labels, (label#, Y, X)).
    instances : ndarray
        A mask array for each labeled instances.
    '''
    labels_grouped=[]            
    for i in range(len(masks)):
        label = np.unique(masks[i])[1:]
        labels_grouped.append(label)
    instance = np.cumsum([0] + [len(l) for l in labels_grouped])
    instances = np.zeros((instance.max(), np.array(np.shape(masks[i][0])).item(),np.array(np.shape(masks[i][1])).item()))
    i = 0
    for mask,label in zip(masks,labels_grouped):
        for n in label:
            instances[i] = (mask==n).astype(int)
            i+=1
    coordinates = np.array([(t, ) + tuple(map(np.mean, np.where(m == 1.))) for t,m in enumerate(instances)])  
    labels = np.hstack(labels_grouped)
    return labels_grouped, labels, coordinates, instances 

def get_gt(seg_path, track_path):
    gt_s_df=pd.read_csv(
        f'{seg_path}'
    )
    gt_t_df=pd.read_csv(
        f'{track_path}'
    )
    gt_s_df.columns = [n.replace(' ', '') for n in gt_s_df.columns] 
    gt_t_df.columns = [n.replace(' ', '') for n in gt_t_df.columns]
    gt_s_df = gt_s_df.drop(columns=['Cell_colour'])
    gt_s = np.round(gt_s_df.to_numpy(copy=True)).astype(int)
    
    gt_t_df = gt_t_df.drop(columns=['Cell_number'])
    gt_t = gt_t_df.to_numpy(copy=True)
    gt_t = np.column_stack((gt_t,gt_s[:,2]))
    gt_t = np.column_stack((gt_t,gt_s[:,3]))
    
    return gt_s, gt_t

def get_pred(output, labels, coordinates):
    o = list(map(existance_vectors, output))
    pred_s= np.zeros(((len(labels),4))).astype(int)
    i=0
    for f in range(len(o)):
        instance = len(o[f])
        offset=i
        for i in range(offset,instance+offset):
            pred_s[i,0] = f+1 # Frame_number
            i+=1
    pred_s[:,1] = labels # Cell_number
    pred_s[:,2] = coordinates[:,2] # Position_X
    pred_s[:,3] = coordinates[:,1] # Position_Y
    pred_t = pred_s.copy()
    pred_s_df = pd.DataFrame(pred_s, columns=["Frame_number", "Cell_number", "Position_X", "Position_Y"])
    pred_t_df = pd.DataFrame(pred_t, columns=["Frame_number", "Cell_number", "Position_X", "Position_Y"])

    return pred_s, pred_s_df, pred_t, pred_t_df