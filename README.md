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

See example [pipeline notebook](https://git.webhosting.rug.nl/P301081/yeastcells-detection-maskrcnn/src/branch/master/notebooks/example_pipeline.ipynb).
The example pipeline gives segmentation and tracking results of brightfield time-lapse yeast microscopy.

**Segmentation** 
* **Input** Brightfield 512x512 time-lapse images. The source file is a multi-image tiff. (For reading multiple single-image tiffs and concatenating them use data.read_image_cat function instead). The frame rate is 300 seconds. 

<table>
  <tr>	
    <td>
        <img src="images/figures/movie1_image_example.png"/>
    </td>
  </tr>
    <tr>
    <td>Figure 1. Example of our input brightfield image.</td>
  </tr>
</table>

---

* **Output** The `output` variable provides a prediction box, prediction score, and a prediction mask for each instance segmentation in each frame. You can access the prediction masks by `np.array(output[<frame>]['instances'].pred_masks.to('cpu'))`.


**Tracking**  
* **Input** The tracking results are obtained by applying the `clustering.cluster_cells` function on the `output` which uses DBSCAN to cluster the detections into labels representing the same cell over time. You can set a maximum time-distance of `<dmax>` frames for the algorithm to consider. A higher number could control for intermittent false negatives but also increases the porbability of misclassification due to cell growth and movement. The `min_samples` and `eps` variables are required arguments for the DBSCAN algorithm. For further explanation see [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
* **Output** A list of `labels` and a list of their `coordinates` [time,y,x]. The labels give the same number (starting from 0) to the cells that are the same over time. -1 indicates noise. 
<table>
  <tr>	
    <td>
        <img src="images/figures/output_xy01_animation.gif"/>
    </td>
  </tr>
    <tr>
    <td>Figure 2. Example of segmented and tracked yeast cells from pipeline output.</td>
  </tr>
</table>

---

**Feature extraction**

This pipeline allows you to extract several features about the detected yeast cells in the time-series. The `features.group` function groups the segmentations by frame. You can use the `features.get_seg_track` function to find the number of segmentations and number of tracked cells (in total or by frame). The `features.extract_contours` function gives the contour points [x,y] for each segmentation. You can visualize the segmentations and tracks in a movie using `visualize.create_scene` and `visualize.show_animation`. Further, you can use `visualize.select_cell` to select a particular cell by label and zoom in on it to observe it better in the movie. The movie displayed with default options gives each cell a unique color that stays the same throughout the movie if the cell is tracked correctly. You also have the options to display the label number by setting the parameter `labelnum` to `True`. The `features.polygons_per_cluster` function uses the [Shapely](https://shapely.readthedocs.io/en/stable/manual.html) polygon class to output polygons for each label. The polygon class can be used to extract the area of the polygon and plot it over time.

# Evaluation

See [evaluation notebook](https://git.webhosting.rug.nl/P301081/yeastcells-detection-maskrcnn/src/branch/master/notebooks/example_evaluation.ipynb). 

We evaluated our pipeline using benchmark data from the [Yeast Image Toolkit](http://yeast-image-toolkit.biosim.eu/) (YIT) (Versari et al., 2017). On this platform, several exisiting pipelines have been evaluated for their segmentation and tracking performance. We tested our pipeline and that of YeaZ (Dietler et al., 2020) on several test sets from this platform. We chose to compare our pipeline with YeaZ because they too use a deep learning CNN, unlike the other pipelines evalued on YIT. 

The YeaZ segmentation and tracking output was obtained by using the [YeaZ-GUI](https://github.com/lpbsscientist/YeaZ-GUI) with the recommended default parameters.

We matched the centroids provided in the benchmark ground truth data to the mask outputs of our pipeline and YeaZ. We then calculated precision, recall, and the F1-score. #insert table #YM

# False positive removal 

