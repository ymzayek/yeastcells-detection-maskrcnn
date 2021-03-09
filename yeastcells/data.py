# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
import os
from .clustering import existance_vectors

def load_data(path, ff = ''):
    '''
    Reads filenames from path.
    Parameters
    ----------
    path : str
        Path to image file(s).
    ff : str
        Input file(s) based on ending (e.g. '.tif') or write full filename.    
    Returns
    -------
    fns : list of str
        All filenames in the path.
    '''
    fns = [
        f'{path}/{fn}' 
        for fn in os.listdir(f'{path}/') 
        if fn.endswith(ff)
    ]
    fns= np.sort(fns)
    
    return fns

def read_image(fn, time_series=True, channel=1, flourescent=False):
    '''
    Reads 3D or 4D image files with shape 
    (frames, length, width, channels) or (frames, length, width) 
    into an array for input into detectron2 predictor. 
    If channel is bright-field, it will be converted to grayscale.
    Parameters
    ----------
    fn : str
        Path to one multi-image tiff file.
    time_series : bool, optional
        If reading a single image instead of time-series image stack use False. 
        The default is True.
    channel : int, optional
        Choose which channel to read. The default is 1.
    flourescent : bool, optional
        Set to true if reading from a flourescent channel. 
        The default is False (assuming bright-field).
    Returns
    -------
    ndarray
        4D array containing data with int type.
    '''
    image = imread(fn)
    if image.ndim==4 and time_series==True:  
        if flourescent==False:
            return (
                (image / image.max() * 255)[:, ..., channel-1:channel] 
                * [[[1, 1, 1]]]
            ).astype(np.uint8)
        else:
            return (image[:, ..., channel-1]).astype(np.uint8)
    elif image.ndim==3 and time_series==True:  
        if flourescent==False:
            return (
                (image / image.max() * 255)[:, ..., None] * [[[1, 1, 1]]]
            ).astype(np.uint8)         
        else:
            return (image[:, ..., None]).astype(np.uint8) 
    elif image.ndim==3 and time_series==False: 
        if flourescent==False:
            return (
                (image / image.max() * 255)[None, ..., channel-1:channel] 
                * [[[1, 1, 1]]]
            ).astype(np.uint8) 
        else:
            return (image[None, ..., channel-1]).astype(np.uint8)    
        
def read_images_cat(fns): 
    '''
    For reading multiple single-image files and concatenating them.
    Parameters
    ----------
    fns : str
        Path to multiple image files.
    Returns
    -------
    ndarray
        4D array containing data with int type.
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
        Path to mask file.
    Returns
    -------
    masks : ndarray 
        3D mask array containing data with int type.
    '''
    img = Image.open(path)
    masks = []
    for i in range(img.n_frames):
        img.seek(i)
        masks.append(np.array(img))
    masks = np.array(masks)
    
    return masks

def get_gt_yit(seg_path, track_path):
    '''
    Retrieves and reformats ground truth data from YIT into arrays.
    Parameters
    ----------
    seg_path : str
        Path and filename of segmentation ground truth csv file.
    track_path : str
        Path and filename of tracking ground truth csv file.      
    Returns
    -------
    gt_s : ndarray
        Segmentation ground truth data array containing data with int type.
    gt_t : ndarray
        Tracking ground truth data array containing data with int type.
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
    Reformats prediction outputs into arrays and a pandas DataFrame. 
    -1 labels are considered as noise for tracking.
    Parameters
    ----------
    output : dict
        Detecron2 predictor output from the detecron2 Mask R-CNN model.
    labels : list
        Tracking labels of individual cells.  
    coordinates : ndarray
        2D array of centroid coordinates individual cells
        (labels, ([time, Y, X])). 
    ti : int
        Time (min) interval representing frame rate of time-series data.
    start : int
        The frame number where the model started segmenting.
    Returns
    -------
    pred_s : ndarray
        Segmentation prediction data array containing data with int type. ym    
    pred_t : ndarray
        Tracking prediction data array containing data with int type. ym
    pred_df : pd.DataFrame
        Dataframe containing segmentation and tracking results with columns:
            =============  ==================================================
            Frame_number   The frame number in time-series image data.
            Time(min)      The time in minutes corresponding to each frame.
            Cell_number    Number given to each segmented cell within a frame.
            Cell_label     Number output by tracking clustering algorithm.
                           Each tracked cell has a unique label number.
            Position_X     The x coordinate of a given cell centroid.
            Position_Y     The y coordinate of a given cell centroid.
            =============  ==================================================
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

def get_pred_yeaz(labels, labels_grouped, coordinates):
    '''
    An alternative to get_pred function used specifically
    for reformatting YeaZ predictions into arrays. 
    Parameters
    ----------
    labels : list
        Tracking labels of individual cells.
    labels_grouped : list
        Grouping of labels in a nested-list by frame.
    coordinates : ndarray
        2D array of centroid coordinates individual cells
        (labels, ([time, Y, X])).      
    Returns
    -------
    pred_s : ndarray
        Segmentation prediction data array containing data with int type. ym    
    pred_t : ndarray
        Tracking prediction data array containing data with int type. ym        
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