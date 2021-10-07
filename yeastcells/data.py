# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from skimage.io import imread
from PIL import Image
import os

import warnings

def load_data(path, ff = ''):
    '''
    Reads filenames from a path.
    Parameters
    ----------
    path : str
        Path to image file(s).
    ff : str
        Input file(s) based on file format (e.g. '.tif') 
        or write full filename.    
    Returns
    -------
    fns : list of str
        All filenames in the path.
    '''
    fns = [
        f'{path}/{fn}' 
        for fn in sorted(os.listdir(path))
        if fn.endswith(ff)
    ]
    return fns