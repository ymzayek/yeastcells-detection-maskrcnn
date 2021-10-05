# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from shapely.geometry import Polygon
import cv2
from .features import get_contours


def plot_paths(detections, xlim=(0, 512), ylim=(0, 512),
               ax=None, style={}, title=None, fig_kws={}):
    '''
    Plots the paths of labelled ('tracked') cells in a 3D figure.
    Parameters
    ----------
    detections : ndarray
        Tracking results of individual segmented cells, including columns frame, cell, x and y
    xlim : tuple, optional
        X-axis scale. The default is (0, 512).
    ylim : tuple, optional
        Y-axis scale. The default is (0, 512).
    ax : Matplotlib axis object, optional
        Sets axes instance. The default is None.
    style : dict, optional
        Keyword arguments to pass to matplotlib.axes.Axes.plot.
        The default is {}.
    title : str, optional
        Title for plot. The default is None.
    Returns
    -------
    ax : Matplotlib axis object
        Plots axis into figure.
    '''
    if ax is None:
        fig = plt.figure(**fig_kws)
        fig.suptitle(title)
        ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    tracked_cells = set(detections[
        (detections['x'] >= xlim[0]) & (detections['x'] <= xlim[1]) &
        (detections['y'] >= ylim[0]) & (detections['y'] <= ylim[1]) &
        (detections['cell'] >= 0)
    ]['cell'].unique())

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('frame')
    for cell, track in detections.groupby('cell'):
        if cell in tracked_cells:
            ax.plot(*track[['x', 'y', 'frame']].values.T, **style)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    return ax


def create_scene(frames, detections, masks, cell_style={},
                 frame_style=None, label_style=None):
    '''
    Sets up 4D data array of formatted time-series images to be passed to the
    function visualize.show_animation to make an animation.
    Parameters
    ----------
    frames : ndarray
        4D array with int type representing time-series images.
    masks : dict
        array with boolean segemntation masks
    detections : ndarray
        Tracking results of individual segmented cells, including columns frame, cell, x and y
    cell_style : dict, optional
        Style options passed to cv2.polylines to draw a cell.
    frame_style, label_style : bool or dict
        Adds a frame number in each frame, respectively a label to each cell if
        True or a dictionary with style options for cv2.putText.
    Returns
    -------
    canvas : ndarray
        4D array with int type representing time-series images.
    '''
    canvas = frames.copy()
    contours = get_contours(masks)
    colors = (
            255 * plt.get_cmap('hsv')(
        np.linspace(0, 1, len(detections['cell'].unique())))[:, :3]).astype(int)
    np.random.shuffle(colors)

    if 'mask' not in detections.columns:
        assert len(detections) == len(contours), (
            "When filtering out tracks, ensure to add a mask column with the "
            "cumulative mask index in segmentation, such that visualisation  "
            "can figure out which mask belongs to which detection, e.g.:\n"
            "    tracking['mask'] = np.arange(len(tracking))\n"
            "before filtering.")
        detections = detections.copy()
        detections['mask'] = np.arange(len(contours))

    for frame_num, track in detections.groupby('frame'):
        frame = canvas[int(frame_num)]
        if frame_style is not None and frame_style != False:
            style = {
                'color': [255, 0, 0], 'fontScale': 2,
                'fontFace': cv2.FONT_HERSHEY_TRIPLEX, 'thickness': 2}
            if isinstance(frame_style, dict):
                style.update(frame_style)
            cv2.putText(frame, f'{int(frame_num)}',
                        org=(5 * style['fontScale'], 30 * style['fontScale']), **style)

        for contour, label in track[['mask', 'cell']].values:
            # draw contour:
            contour = contours[contour]
            if len(contour) == 0:
                continue
            style = {'thickness': 1, 'color': tuple(map(int, colors[label]))}
            style.update(cell_style)
            frame[:] = cv2.polylines(
                frame.astype(np.uint8), contour[None], isClosed=True, **style)

            if label_style is not None and label_style != False:
                style = {'color': [255, 0, 0], 'fontScale': 1, 'thickness': 1,
                         'fontFace': cv2.FONT_HERSHEY_PLAIN}
                if len(contour) >= 3:
                    x, y = Polygon(contour).centroid.coords.xy
                    x, y = int(np.round(x[0])), int(np.round(y[0]))
                else:
                    x, y = contour.mean(0).round().astype(int)
                frame[:] = cv2.putText(frame, f'{label:X}' if label > 0 else 'x', org=(x, y), **style)
    return canvas


def select_cell(scene, detections, label, w=40):
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
    detections: dataframe
        dataframe containing columns frame, cell, x and y
    label : int
        The label of the cell that you want to center and zoom on.
        The default is 0.
    w : int, optional
        Determines the zoom magnitude based on the cell in focus. 
        The default is 40.
    Returns
    -------
    ndarray
        4D array with int type representing time-series images.
    '''
    detections = detections[detections['cell'] == label]
    xmin, ymin, fmin = (detections[['x', 'y', 'frame']].values.min(0) - [w, w, 0]).clip(0).round().astype(int)
    xmax, ymax, fmax = (detections[['x', 'y', 'frame']].values.max(0) + [w, w, 0]).round().astype(int)
    return scene[fmin:fmax, ymin:ymax, xmin:xmax]


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
