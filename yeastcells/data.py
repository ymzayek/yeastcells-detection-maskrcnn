# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
import os
from .clustering import existance_vectors

def load_data(path, ff = '.tif'):
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
        if fn.endswith(ff)
    ]
    fns= np.sort(fns)
    
    return fns

def read_image(fn, single_im=False, shape=1, start_frame=1, channel=1, grayscale=True):
    '''
    Reads images into an array with correct shape for input into detectron2 
    predictor.
    Parameters
    ----------
    fn : str
        Path to one multi-image tiff file.
    Returns
    -------
    ndarray
        A 4-dimensional array (frames, length, width, channels).
    ''' 
    if shape==1: 
        image = imread(fn)
        image = image[start_frame-1:]
    elif shape==2: 
        image = imread(fn) 
        image = np.rollaxis(image,1,4)
        image = image[start_frame-1:]
    if image.ndim==4 and single_im==False:  
        if grayscale==True:
            return (
                (image / image.max() * 255)[:, ..., channel-1:channel] * [[[1, 1, 1]]]
            ).astype(np.uint8)
        else:
            return (
                image[:, ..., channel-1:channel] * [[[1, 1, 1]]]
            ).astype(np.uint8)
    elif image.ndim==3 and single_im==False:  
        if grayscale==True:
            return (
                (image / image.max() * 255)[:, ..., None] * [[[1, 1, 1]]]
            ).astype(np.uint8)
        else:
            return (
                image[:, ..., None] * [[[1, 1, 1]]]
            ).astype(np.uint8)            
    elif image.ndim==3 and single_im==True:  
        if grayscale==True:
            return (
              (image / image.max() * 255)[None, ..., channel-1:channel] * [[[1, 1, 1]]]
            ).astype(np.uint8) 
        else:
            return (
                image[None, ..., channel-1:channel] * [[[1, 1, 1]]]
            ).astype(np.uint8) 
        
def read_images_cat(fns): 
    '''
    For reading multiple single-image tiffs and concatenating them.
    Parameters
    ----------
    fns : str
        Path to multiple tiff files.
    Returns
    -------
    ndarray
        A 4-dimensional array (frames, length, width, channels).
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
        A 4-dimensional array (frames, length, width, channels). ym
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
    Extract labels for each segmented instance in masks.
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
        Coordinates of centroid of individual instances with 2 dimensions 
        (labels, ([time, Y, X])).
    instances : ndarray
        A mask array for each labeled instances.
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
    return labels_grouped, labels, coordinates, instances 

def get_gt_yit(seg_path, track_path):
    '''
    Retrieves and reformats ground truth data into arrays.
    Parameters
    ----------
    seg_path : str
        Path to segmentation ground truth csv file.
    track_path : str
        Path to tracking ground truth csv file.      
    Returns
    -------
    gt_s : ndarray
        Segmentation ground truth. ym
    gt_t : ndarray
        Tracking ground truth.    ym
    '''
    gt_s_df=pd.read_csv(f'{seg_path}')
    gt_t_df=pd.read_csv(f'{track_path}')
    gt_s_df.columns = [n.replace(' ', '') for n in gt_s_df.columns] 
    gt_t_df.columns = [n.replace(' ', '') for n in gt_t_df.columns]
    gt_s_df = gt_s_df.drop(columns=['Cell_colour'])
    gt_s = np.round(gt_s_df.to_numpy(copy=True)).astype(int)
    
    gt_t_df = gt_t_df.drop(columns=['Cell_number'])
    gt_t = gt_t_df.to_numpy(copy=True)
    gt_t = np.column_stack((gt_t,gt_s[:,2]))
    gt_t = np.column_stack((gt_t,gt_s[:,3]))
    
    return gt_s, gt_t

def get_pred(output, labels, coordinates, ti=3, start=1):
    '''
    Reformats prediction outputs into 1 array.
    Parameters
    ----------
    output : dict
        Predictor output from the detecron2 model.
    labels : ndarray
        Tracking labels of individual instances.    
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).        
    Returns
    -------
    pred_s : ndarray
        Segmentation predictions.     
    pred_t : ndarray
        Tracking predictions.
    pred_df : ndarray
        Combined segmentation and tracking results
        -1 labels considered as noise for tracking.         
    '''
    o = list(map(existance_vectors, output))
    pred_s= np.zeros(((len(labels),4))).astype(int)
    cell_num = np.array([], dtype=int)
    time_min = []
    i=0
    for f in range(len(o)):
        instance = len(o[f])
        tmp = np.arange(instance)
        cell_num = np.hstack((cell_num,tmp))
        offset=i
        for i in range(offset,instance+offset):
            pred_s[i,0] = f+start # Frame_number
            i+=1
            time_min.append((f+start-1)*ti)
    pred_s[:,1] = cell_num # Cell_number
    pred_s[:,2] = coordinates[:,2] # Position_X
    pred_s[:,3] = coordinates[:,1] # Position_Y
    pred_t = pred_s.copy()
    pred_t[:,1] = labels
    pred_df = pd.DataFrame(pred_s, columns=[
        'Frame_number', 'Cell_number', 
        'Position_X', 'Position_Y'
    ])
    pred_df['Cell_label'] = labels 
    pred_df['Time(min)'] = time_min
    pred_df = pred_df[[
        'Frame_number', 'Time(min)', 'Cell_number', 
        'Cell_label', 'Position_X', 'Position_Y'
    ]]
    
    return pred_s, pred_t, pred_df

def get_pred_yeaz(labels_grouped, labels, coordinates):
    '''
    An alternative to get_pred and is used specifically
    for reformatting YeaZ predictions into an array. 
    Parameters
    ----------
    labels_grouped : list ym
        Grouping of labels in a list by frame.
    labels : ndarray
        Tracking labels of individual instances.    
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions 
        (labels, ([time, Y, X])).        
    Returns
    -------
    pred_s : ndarray
        Segmentation predictions. ym     
    pred_t : ndarray
        Tracking predictions.    ym          
    '''
    pred_s= np.zeros(((len(labels),4))).astype(int)
    i=0
    for f in range(len(labels_grouped)):
        instance = len(labels_grouped[f])
        offset=i
        for i in range(offset,instance+offset):
            pred_s[i,0] = f+1 # Frame_number
            i+=1
    pred_s[:,1] = labels # Cell_number
    pred_s[:,2] = coordinates[:,2] # Position_X
    pred_s[:,3] = coordinates[:,1] # Position_Y 
    pred_t = pred_s.copy()
    
    return pred_s, pred_t