# Yeast Cell Segmentation and Tracking Pipeline

Automatic segmentation and tracking of budding yeast cells in time-series brightfield microscopy images using a Mask R-CNN

# Participants

* Dr. Andreas Milias Argeitis, principal investigator, University of Groningen, Faculty of Science and Engineering
* MSc. Paolo Guerra, second principal investigator, University of Groningen, Faculty of Science and Engineering
* MSc Herbert Teun Kruitbosch, data scientist, University of Groningen, Data science team
* Msc Yasmin Mzayek, IT trainee, University of Groningen, Data Science trainee
* MA Sara Omlor, IT trainee, University of Groningen, Data Science trainee

# Project description

* See https://github.com/prhbrt/yeast-cell-detection

# Implementation

See example [notebook](https://git.webhosting.rug.nl/P301081/yeastcells-detection-maskrcnn/src/branch/master/notebooks/example_pipeline.ipynb).
The example pipeline gives segmentation and tracking results of brightfield time-lapse yeast microscopy.

**Segmentation** 
* **Input** Brightfield 512x512 time-lapse images. The source file is a multi-image tiff. (For reading multiple single-image tiffs and concatenating them use data.read_image_cat function instead). The frame rate is 300 seconds. #fig and movie
* **Output** The `output` variable provides a prediction box, prediction score, and a prediction mask for each instance segmentation in each frame. You can access the prediction masks by `np.array(output[<frame>]['instances'].pred_masks.to('cpu'))`. #fig and movie


**Tracking**  
* **Input** The tracking results are obtained by applying the `cluster_cells` function on the `output` which uses DBSCAN to cluster the detections into labels representing the same cell over time. You can set a maximum time-distance of `<dmax>` frames for the algorithm to consider. A higher number could control for intermittent false negatives but also increases the porbability of misclassification due to cell growth and movement. The `min_samples` and `eps` variables are required arguments for the DBSCAN algorithm. For further explanation see [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
* **Output** A list of `labels` and a list of their `coordinates` [time,y,x]. The labels give the same number (starting from 0) to the cells that are the same over time. -1 indicates noise. #fig and movie

