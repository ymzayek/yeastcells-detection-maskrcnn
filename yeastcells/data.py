# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from skimage.io import imread
import os
from .clustering import existance_vectors

def load_data(path):
    filenames = [
        f'{path}/{fn}' 
        for fn in os.listdir(f'{path}/') 
        if fn.endswith('.tif')
    ]
    filenames= np.sort(filenames)
    
    return filenames

def read_image(fn): #for reading one multi-image tiff
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
    image=[]
    image=[imread(i) for i in list(fns)]
    image= np.array(image)
    
    return (
        (image / image.max() * 255)[:, ..., None] * [[[1, 1, 1]]]
    ).astype(np.uint8) 

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
    pred_s_df = pd.DataFrame(pred_s, columns=["Frame_number", "Cell_number", "Cell_colour", "Position_X", "Position_Y"])
    pred_t_df = pd.DataFrame(pred_t, columns=["Frame_number", "Cell_number", "Position_X", "Position_Y"])

    return pred_s, pred_s_df, pred_t, pred_t_df