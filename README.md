# Deep learning pipeline for yeast cell segmentation and tracking 

 In this pipeline we created synthetic brightfield images of yeast cells and trained a Mask R-CNN model on them. Then we used the trained network on real time-series brightfield microscopy data to automaticly segment and track budding yeast cells.

# Participants

* MSc Herbert Teun Kruitbosch, data scientist, University of Groningen, Data science team
* MSc Yasmin Mzayek, IT trainee, University of Groningen, Data Science trainee
* MA Sara Omlor, IT trainee, University of Groningen, Data Science trainee
* MSc Paolo Guerra, PhD student, University of Groningen, Faculty of Science and Engineering, [Molecular Systems Biology](https://www.rug.nl/research/molecular-systems-biology)
* Dr Andreas Milias Argeitis, principal investigator, University of Groningen, Faculty of Science and Engineering, [Molecular Systems Biology](https://www.rug.nl/research/molecular-systems-biology/research-dr.andreas-milias-argeitis?lang=en)

# Project description

**Goals** 
* To create synthetic image data to train a deep convolutional neural network
* To implement an automatic segmentation pipeline using this network
* To track cells across time frames

# Get started on Google Colab

We've tried to make our experiments outsider accessible, particularly by setting up the installation for `detectron2` in Google Colab and by downloading all external resources when needed. Please note that in these notebooks the first cells install all dependencies, this should work without restarting. However, try restarting via the Colab Runtime menu on errors, since inappropriate versions might have been imported into the runtime before the appropriate ones were installed. Particularly the `Train model on synthetic data` might require this. For the other notebooks not restarting will unlikely cause issues.

 * [![Example cell detection](https://colab.research.google.com/assets/colab-badge.svg) Example cell detection (several minutes)](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/example_pipeline.ipynb)
* [![Evaluation](https://colab.research.google.com/assets/colab-badge.svg) Evaluation of our Mask R-CNN model against YeaZ and YeastNet2](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/compare_models.ipynb)
 * [![Mask R-CNN calibration](https://colab.research.google.com/assets/colab-badge.svg) Hyperparameter tuning for Mask R-CNN segmentation and tracking (~ 30-200 minutes)](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/Calibration.ipynb)

The two notebooks below allow you to create synthetic data and train a model. For a proof of concept, respectively set the `sets` and `max_iter` parameters to the lower values suggested. If you want to run them for a realistic use-case, please know these scripts take several hours to complete, and Google Colab is not intended for this. The results are large (~0.5 - 2GB) and on Colab you might easily fail to safe guard them when Google Colab shuts down the machine due to inactivity.

 * [![Create synthetic data set](https://colab.research.google.com/assets/colab-badge.svg) Create synthetic data set  (8 hours)](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/create_synthetic_dataset_for_training.ipynb)
 * [![Train model on synthetic data](https://colab.research.google.com/assets/colab-badge.svg) Train model on synthetic data (8 hours)](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/train_mask_rcnn_network.ipynb)

# Implementation

For creating the synthetic data set and training the network see the notebooks [create_synthetic_dataset_for_training](https://github.com/ymzayek/yeastcells-detection-maskrcnn/blob/main/notebooks/create_synthetic_dataset_for_training.ipynb) and [train_mask_rcnn_network](https://github.com/ymzayek/yeastcells-detection-maskrcnn/blob/main/notebooks/train_mask_rcnn_network.ipynb).

For segmentation and tracking on real data see [example pipeline](https://github.com/ymzayek/yeastcells-detection-maskrcnn/tree/main/notebooks/example_pipeline.ipynb) notebook.

All the notebooks can be run on Google Colab and automatically install and download all needed dependencies and data (see links above).   

<sub>(To run the Mask-RCNN locally, you will need to install the [Detecron2 library](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). For a guide to a Window's installation see these [instructions](https://ivanpp.cc/detectron2-walkthrough-windows/). You also need to download the trained model file from https://datascience.web.rug.nl/models/yeast-cells/mask-rcnn/v1/model_final.pth)</sub>

## Segmentation: `get_segmentation`, `get_model`
* **Input** Brightfield time-lapse images. The source file is either a tiff stack or multiple tiff files forming the time-series.

* **Output** A dataframe with one row for each detection and `# detections` X `height` X `width` `numpy.ndarray` with the boolean segmentation masks, the masks and the dataframe have the same length and the `mask` column refers to the first dimension of the masks array. The dataframe also has columns `frame`, `x` and `y` to mark the frame of the source image and the centroid of the detection.

<table>
  <tr>	
    <td>
        <img src="figures/segmentations/seg_example.png"/>
    </td>
  </tr>
  <tr>
    <td>Example of 512x512 brightfield images and their detections. Detected yeast cells are highlighted by a magenta border. A) and B) show the segmentations in one frame of time-series agarpad experiments, C) shows segmentations in a microfluidic experiment and D) shows a close up of the boundries of detected cells. </td>
  </tr>
</table>

<br>

## Tracking: `track_cells`
* **Input** Besides the dataframe and masks from segmentation, tracking needs hyperparameters for the DBSCAN clustering and the maximum frame distance when determining the distances between detections. You can set a maximum frame distance of `<dmax>` for the algorithm to use to calculate the distances between detections in the current `frame` and both `frame-dmax`, `frame+dmax`. In other words, this will calculate distances between all instances in a current frame and all the instances in the following and previous frames up to `dmax`. A higher `dmax` could control for intermittent false negatives because if a cell is missed in an andjacent frame but picked up again 2 frames ahead, the cell will be tracked. However, this also increases the probability of misclassification due to cell growth and movement with time if you look ahead too far. The `min_samples` and `eps` variables are required arguments for the DBSCAN algorithm. For further explanation see [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

* **Output** The cell column is added to the dataframe of detections, which is -1 if the tracking algorithm marked it as an outlier and hence didn't track it.

<table>
  <tr>	
    <td>
        <img src="figures/gifs/animation.gif"/>
    </td>
  </tr>
  <tr>
    <td>Segmented and tracked yeast cells from Mask R-CNN. The frame rate of these time-series images is 180 seconds. </td>
  </tr>
</table>

<br>

You can visualize the segmentations and tracks in a movie using `visualize.create_scene` and `visualize.show_animation`. Further, you can use `visualize.select_cell` to select a particular cell by label and zoom in on it to observe it better in the movie. The movie displayed with default options gives each cell a unique color that stays the same throughout the movie if the cell is tracked correctly. You also have the options to display the label number by setting the parameter `labelnum` to `True`.

**Information and feature extraction**

This pipeline allows you to extract information about the detected yeast cells in the time-series. The `features.extract_contours` function gives the contour points [x,y] for each segmentation. The masks for all detections can be extracted and their areas can be caulculated as shown in the example pipeline notebook.

<br>

<table>
  <tr>	
    <td>
        <img src="figures/features/mask_overlay.png"/>
    </td>
  </tr>
    <tr>
    <td>A mother/daughter pair of masks are overlayed on the original brightfield image.</td>
  </tr>
</table>

<br>

Further, if a flourescent channel is available, the pixel intensity of within each cell can also be calculated using the masks segmented on the brightfield images.

<table>
  <tr>	
    <td>
        <img src="figures/features/prediction_df.png"/>
    </td>
  </tr>
    <tr>
    <td>Example of Mask R-CNN pipeline output.</td>
  </tr>
</table>

<br>

# Evaluation

* [![Evaluation](https://colab.research.google.com/assets/colab-badge.svg) Evaluation of our model against YeaZ and YeastNet2](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/compare_models.ipynb)

We evaluated our pipeline using benchmark data from the [Yeast Image Toolkit](http://yeast-image-toolkit.biosim.eu/) (YIT) (Versari et al., 2017). On this platform, several exisiting pipelines have been evaluated for their segmentation and tracking performance. We tested our pipeline and that of YeaZ (Dietler et al., 2020) and YeastNet2 (Salem et al., 2021) on several test sets from this platform. 

<br/>

We chose to compare our pipeline with YeaZ and YeastNet2 because they also use a deep learning CNN, unlike the other pipelines evaluated on YIT. 

The YeaZ segmentation and tracking implementation is based on [YeaZ-GUI](https://github.com/lpbsscientist/YeaZ-GUI) with optimized parameters obtained in [this notebook](#UPDATE). Additionally, our implementation allows for the use of GPU for the YeaZ pipeline.

The YeastNet2 segmentation and tracking were implemented using the [YeaZ-GUI](https://github.com/kaernlab/YeastNet).

We matched the centroids provided in the benchmark ground truth data to the mask outputs for each model. This is slightly different than the way it was done on the evaluation platform of YIT but comparable since they matched centroids of the prediction to the centroids of the ground truth using a maximum distance threshold to count a comparison as a true positive (see their [EP](https://github.com/Fafa87/EP) for more detail). We then calculated precision, recall, accuracy, and the F1-score.


In the table below, we report the performance metrics for each test set for both YeaZ and our pipeline for comparison.

<br/>

<table>
  <tr>	
    <td>
        <img src="figures/eval/evaluation_table_seg.png"/>
    </td>
  </tr>
    <tr>
    <td>Segmentation evaluation results from 7 test sets from the YIT. Precision, recall, accuracy, and the F1-score of the performance of our pipeline, YeaZ, and YeastNet2 are reported.</td>
  </tr>
</table> 

<table>
  <tr>	
    <td>
        <img src="figures/eval/evaluation_table_track.png"/>
    </td>
  </tr>
    <tr>
    <td>Tracking evaluation results from 7 test sets from the YIT. Precision, recall, accuracy, and the F1-score of the performance of our pipeline, YeaZ, and YeastNet2 are reported.</td>
  </tr>
</table> 

<br>

We further quantitatively evaluated our segmentation accuracy based on IOU and compared it to YeaZ using publicly available annotated ground truth data from the YeaZ group.

<table>
  <tr>	
    <td>
        <img src="figures/eval/iou_table.png"/>
    </td>
  </tr>
    <tr>
    <td>Average IOU is calculated for true positives using annotated brightfield images of wild-type cells from the YeaZ dataset</td>
  </tr>
</table> 

<br>

# Hyperparameters

 * [![Mask R-CNN calibration](https://colab.research.google.com/assets/colab-badge.svg) Hyperparameter tuning for Mask R-CNN segmentation and tracking (~ 30-200 minutes)](https://colab.research.google.com/github/ymzayek/yeastcells-detection-maskrcnn/blob/master/notebooks/Calibration.ipynb)

For our pipeline, we used calibration curves to set the segmentation threshold score needed by the Mask R-CNN to define the probablity that an instance is a yeast cell. For tracking, we used them to tune the `epsilon` of DBSCAN and `dmax`, the maximum amount of frames between two detections allowed to adjacently track them as the same cell.

<br/>

<table>
  <tr>	
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet1.png"/>
        <p>YIT Test set 1</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet2.png"/>
        <p>YIT Test set 2</p>
    </td>
  </tr>
  <tr>	
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet3.png"/>
        <p>YIT Test set 3</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet4.png"/>
        <p>YIT Test set 4</p>
    </td>
  </tr>
  <tr>	
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet5.png"/>
        <p>YIT Test set 5</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet6.png"/>
        <p>YIT Test set 6</p>
    </td>
  </tr>
  <tr>	
    <td>
        <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-TestSet7.png"/>
        <p>YIT Test set 7</p>
    </td>
    <td>
     <img src="figures/eval/calibration_curves/Threshold-calibration-curve-segmentation-legend.png"/>
    </td>
  </tr>
    <tr>
    <td colspan="2">Calibration curves for each test set showing the 4 different metrics against the segmentation threshold score.</td>
  </tr>
</table>

<br/>

### Metrics

<table>
  <tr>	
    <td>
        <img src="figures/eval/metrics.png"/>
    </td>
  </tr>
    <tr>
    <td>TP: true positive detections <br>
    FP: false positive detections <br>
    FN: false negatives</td>
  </tr>
</table> 

<br/>

<table>
  <tr>	
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet1.png"/>
        <p>YIT Test set 1</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet2.png"/>
        <p>YIT Test set 2</p>
    </td>
  </tr>
  <tr>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet3.png"/>
        <p>YIT Test set 3</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet4.png"/>
        <p>YIT Test set 4</p>
    </td>
  </tr>
  <tr>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet5.png"/>
        <p>YIT Test set 5</p>
    </td>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet6.png"/>
        <p>YIT Test set 6</p>
    </td>
  </tr>
  <tr>
    <td>
        <img src="figures/eval/calibration_curves/Tracking-calibration-curve-segmentation-TestSet7.png"/>
        <p>YIT Test set 7</p>
    </td>
  </tr>
  <tr>
    <td colspan="7">Calibration curves for tracking performance and hyperparameter tuning.</td>
  </tr>
</table>

<br/>
