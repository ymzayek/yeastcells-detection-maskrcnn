# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon
import numpy as np
import cv2
from .features import group 

def plot_paths(
        labels, coordinates, xlim=(0, 512), ylim=(0, 512), 
        ax=None, style={}, subset=None, title=None
    ):
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
    canvas = frames.copy()
  
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, labels.max()+1))
    np.random.shuffle(colors)
  
    labels = group(labels, output)
    x, y = contours
  
    for frame_num, (frame, x_, y_, label) in enumerate(zip(canvas, x, y, labels)):
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
                org=(5 * style['fontScale'], 30 * style['fontScale']), **style
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
                    color = 255 * colors[label_, :3] if color is None else color,
                    thickness=thickness
                )
                style = {
                    'color': [255, 0, 0], 'fontScale': 1,
                    'fontFace': cv2.FONT_HERSHEY_PLAIN,
                    'thickness':1
                }
                if labelnum is True:
                    poly_ = Polygon(poly[0]) if len(poly[0]) >= 3 else (x__.mean(), y__.mean())
                    poly_x,poly_y = zip(*(poly_.centroid.coords)) if len(poly[0]) >= 3 else poly_
                    if type(poly_x) is tuple:
                        poly_x,poly_y = poly_x[0],poly_y[0]
                    frame[:] = cv2.putText(
                        frame, f'{label_}', org=(int(poly_x), 
                        int(poly_y)), **style
                    )

    return canvas

def select_cell(scene, coordinates, labels, w=40, l=0):
    label = l
    z, y, x = coordinates[labels == label].T 
    xmin, xmax = int(max(0, x.mean() - w)), int(x.mean() + w)
    ymin, ymax = int(max(0, y.mean() - w)), int(y.mean() + w)
    sub = (slice(ymin, ymax), slice(xmin, xmax)) 

    return scene[:,sub[0],sub[1]], label

def show_animation(scene, title='', delay = 500):
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

def plot_area_profiles(polygons, ti=3, label_list=[0], ax=None, title=None):
    if ax is None:
        fig = plt.figure()  
        fig.suptitle(title)
    for label in label_list:
        time_min = []
        for t in polygons[label].keys():
            time_min.append(t*ti)
        area = np.zeros((len(polygons[label])))
        for i,p in enumerate(polygons[label]):
            area[i] = polygons[label][p].area
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Area')
        ax.scatter(time_min, area, s=1)
        ax.legend(label_list, loc="upper left")
        
    return ax  

def plot_polygon_mask(
        masks, labels, output, frames, polygons, 
        label_list=[0], frame=0, ax=None, title=None
    ):
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
        x, y = polygons[label][frame].exterior.coords.xy
        ax = plt.plot(x, y)
        
    return ax      