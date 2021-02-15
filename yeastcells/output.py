# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def out_cvs(pred_df, fn = 'output.csv)'):
    
    return pred_df.to_csv(fn, index=False)