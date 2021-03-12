# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from shapely.geometry import Polygon
import cv2
from .features import group 

def plot_paths(
        labels, coordinates, xlim=(0, 512), ylim=(0, 512), 
        ax=None, style={}, subset=None, title=None
    ):
    '''
    Plots the paths of labelled ('tracked') cells in a 3D figure.
    Parameters
    ----------
    labels : ndarray
        Tracking labels of individual segmented cells.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    xlim : tuple, optional
        X-axis scale. The default is (0, 512).
    ylim : tuple, optional
        Y-axis scale. The default is (0, 512).
    ax : Matplotlib axis object, optional
        Sets axes instance. The default is None.
    style : dict, optional
        Keyword arguments to pass to matplotlib.axes.Axes.plot. 
        The default is {}.
    subset : list, optional
        List of a subset of labels to plot. The default is None.
    title : str, optional
        Title for plot. The default is None.
    Returns
    -------
    ax : Matplotlib axis object
        Plots axis into figure.
    '''
    if ax is None:
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    sub = (
        (coordinates[:, 0] >= xlim[0]) & (coordinates[:, 0] <= xlim[1]) &
        (coordinates[:, 1] >= ylim[0]) & (coordinates[:, 1] <= ylim[1]) &
        (labels >= 0 if -1 in labels else labels > 0) 
    )
    coordinates__ = coordinates[sub]
    clusters__ = labels[sub]
  
    for label in range(clusters__.max()+1):
        if subset is None or label in subset:
            coords = coordinates__[clusters__ == label]
            ax.plot(*coords.T, **style)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
  
    return ax

def create_scene(
        frames, output, labels, contours, subset=None, thickness=1, 
        color=None, framenum=False, labelnum=False
    ):
    '''
    Sets up 4D data array of formatted time-series images to be passed to the 
    function visualize.show_animation to make an animation.
    Parameters
    ----------
    frames : ndarray
        4D array with int type representing time-series images.
    output : dict
        Predictor output from the detecron2 model.
    labels : ndarray
        Tracking labels of individual segmented cells.
    contours : list
        Zipped list of cell boundary points. 
        Contours[0] gives the x coordinates and 
        contours[1] gives the y coordinates.
    subset : list, optional
        List of a subset of labels to plot. The default is None.
    thickness : int, optional
        Thickness of cell boundary line that is drawn on the image. 
        The default is 1.
    color : tuple, optional
        Set BGR for color of cell boundary lines. The default is None. 
        If default is chosen, different and random colors are 
        assigned to each cell.
    framenum : bool, optional
        If True, the frame number will be displayed in the 
        upper left hand corner. The default is False.
    labelnum : bool, optional
        If True, the label number of each segmented cell will 
        be displayed in addition to the colored boundary line. 
        The default is False.
    Returns
    -------
    canvas : ndarray
        4D array with int type representing time-series images.
    '''
    canvas = frames.copy()
  
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, labels.max()+1))
    np.random.shuffle(colors)
  
    labels = group(labels, output)
    x, y = contours
  
    for frame_num, (frame, x_, y_, label) in enumerate(
            zip(canvas, x, y, labels)
    ):
        if framenum:
            style = {
                'color': [255, 0, 0], 'fontScale': 2,
                'fontFace': cv2.FONT_HERSHEY_TRIPLEX,
                'thickness':2
            }
            if type(framenum) == dict:
                style.update(framenum)
            cv2.putText(
                frame, f'{frame_num}',
                org=(5*style['fontScale'], 30*style['fontScale']),**style
            )
    
        for x__, y__, label_ in zip(x_, y_, label):
            poly = np.concatenate([[
                x__[:, None],
                y__[:, None]
            ]], axis=1).astype(np.int32).T
      
            if len(poly[0])==0:
                continue
            if subset is None or label_ in subset:
                frame[:] = cv2.polylines(
                    frame.astype(np.uint8),
                    poly,
                    isClosed=True,
                    color = 255*colors[label_, :3] if color is None else color,
                    thickness=thickness
                )
                style = {
                    'color': [255, 0, 0], 'fontScale': 1,
                    'fontFace': cv2.FONT_HERSHEY_PLAIN,
                    'thickness':1
                }
                if labelnum is True:
                    poly_ = Polygon(poly[0]) if len(poly[0]) >= 3 else (
                        x__.mean(), y__.mean()
                    )
                    poly_x,poly_y = zip(*(poly_.centroid.coords)) if len(
                        poly[0]) >= 3 else poly_
                    if type(poly_x) is tuple:
                        poly_x,poly_y = poly_x[0],poly_y[0]
                    frame[:] = cv2.putText(
                        frame, f'{label_}', org=(int(poly_x), 
                        int(poly_y)), **style
                    )

    return canvas

def select_cell(scene, coordinates, labels, w=40, l=0):
    '''
    Sets up 4D data array of formatted time-series images to be passed to the 
    function visualize.show_animation to make an animation. 
    It takes the output from the visualize.create_scene function as input 
    and allows for selecting and zooming in on one cell to follow throughout 
    the time-series.   
    Parameters
    ----------
    scene : ndarray
        4D array with int type representing time-series images.
    coordinates : ndarray
        Coordinates of centroid of individual instances with 2 dimensions
        (labels, ([time, Y, X])).
    labels : ndarray
        Tracking labels of individual segmented cells.
    w : int, optional
        Determines the zoom magnitude based on the cell in focus. 
        The default is 40.
    l : int, optional
        The label of the cell that you want to center and zoom on. 
        The default is 0.
    Returns
    -------
    ndarray
        4D array with int type representing time-series images.
    '''
    label = l
    z, y, x = coordinates[labels == label].T 
    xmin, xmax = int(max(0, x.mean() - w)), int(x.mean() + w)
    ymin, ymax = int(max(0, y.mean() - w)), int(y.mean() + w)
    sub = (slice(ymin, ymax), slice(xmin, xmax)) 

    return scene[:,sub[0],sub[1]]

def show_animation(scene, title=None, delay = 500):
    '''
    Creates and displays a movie of the time-series images.
    Parameters
    ----------
    scene : ndarray
        4D array with int type representing time-series images.
    title : str, optional
        Set figure title. The default is None.
    delay : int, optional
        Set delay between frames in milliseconds. The default is 500.
    Returns
    -------
    movie : FuncAnimation
        Object of class matplotlib.animation.FuncAnimation 
        that makes an animation.
    '''
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle(title)
    im = plt.imshow(scene[0])

    def update(frame):
        nonlocal im, scene, fig
        im.set_array(scene[frame])
    
    movie = FuncAnimation(
        fig, update, frames=range(len(scene)), 
        blit=False, repeat=True, interval=delay
    )
    
    return movie

def plot_area_profiles(mask_areas, time_min, labels, label_list=[0], ax=None, title=None):
    '''
    Useful to visualize area profile of individual or multiple cells over time. 
    If multiple cells, e.g. choose a mother/daughter pair to plot.
    Parameters
    ----------
    mask_area : ndarray
        Array containing mask area data with float type.
    time_min : ndarray
        Array with time offset data with int type.
    labels : ndarray
        Tracking labels of individual segmented cells.        
    label_list : list, optional
        List of labels for which you want to plot the area over time. 
        The default is [0].
    ax : Matplotlib axis object, optional
        Sets axes instance. The default is None.
    title : str, optional
        Set figure title. The default is None.
    Returns
    -------
    ax : Matplotlib axis object
        Plots axis into figure.
    '''
    if ax is None:
        fig = plt.figure()  
        fig.suptitle(title)
    for label in label_list:
        idx = np.where(labels == label)[0]
        areas = mask_areas[idx]
        time_min= time_min[idx]
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Area')
        ax.scatter(time_min, areas, s=2)
        ax.legend(label_list, loc="upper left")
        
    return ax 

def plot_mask_overlay(
        masks, labels, output, frames, 
        label_list=[0], frame=0, ax=None, title=None
    ):
    '''
    Display the masks of the segmented cells over the image of the cells. 
    Useful to visually assess segmentation accuracy.
    Parameters
    ----------
    masks : ndarray 
        3D binary mask array of segmented cells containing data with int type.
    labels : ndarray
        Tracking labels of individual segmented cells. 
    output : dict
        Predictor output from the detecron2 model.
    frames : ndarray
        4D array of the time-series images
        (frames, length, width, channels).
    label_list : list, optional
        List of labels for which you want to plot the area over time. 
        The default is [0].
    frame : int, optional
        Select the frame that you would like to plot. The default is 0.
    ax : Matplotlib axis object, optional
        Sets axes instance. The default is None.
    title : str, optional
        Set figure title. The default is None.
    Returns
    -------
    ax : Matplotlib axis object
        Plots axis into figure.
    '''
    if ax is None:
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        ax = plt.imshow(frames[frame])
    labels_ = group(labels, output)
    for label in label_list:
        mask = [masks[i] for i in np.where(labels==label)[0]]
        if label in labels_[frame]: #for verifying correct label is chosen
            mask = mask[frame]    
        ax = plt.imshow(mask, alpha=0.1)
        
    return ax      