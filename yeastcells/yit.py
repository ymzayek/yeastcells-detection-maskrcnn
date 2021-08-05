# -*- coding: utf-8 -*-
from . import data
import numpy as np
import pandas as pd
from skimage.io import imread


def get_ground_truth(path, testset_name = 'TestSet1'):
  # we need the load the tracking labels from a differen file as
  # the segmentation coordinates.
  ground_truth_segmentation = pd.read_csv(
    f'{path}/YIT-Benchmark2/{testset_name}/GroundTruth/GroundTruth_Segmentation.csv',
    sep=', ', # there's an awkward space next to the commas
    engine='python',
  ).drop('Cell_colour', axis=1).rename(
    {'Frame_number': 'frame', 'Position_X' :'x', 'Position_Y': 'y'}, axis=1).reindex()

  ground_truth_tracking = pd.read_csv(
    f'{path}/YIT-Benchmark2/{testset_name}/GroundTruth/GroundTruth_Tracking.csv',
    sep=', ', # there's an awkward space next to the commas
    engine='python',
  ).rename(
    {'Frame_number': 'frame',	'Unique_cell_number': 'cell'}, axis=1).reindex()
  ground_truth = ground_truth_tracking.merge(
    ground_truth_segmentation, how='outer',
    left_on=['frame', 'Cell_number'], right_on=['frame', 'Cell_number']
  )
  ground_truth = ground_truth.drop('Cell_number', axis=1)
  assert len(ground_truth) == len(ground_truth_tracking)
  assert len(ground_truth) == len(ground_truth_segmentation)

  # detections start counting at 0, ground truths at 1. rectify:
  ground_truth['frame'] = ground_truth['frame'] - 1
  return ground_truth


def get_test_movie(path, testset_name = 'TestSet1'):
  filenames = data.load_data(f'{path}/YIT-Benchmark2/{testset_name}/RawData', ff = '.tif')
  assert len(filenames) > 0, "No images were found"

  image = [imread(filename) for filename in filenames]

  assert len({im.shape for im in image}) == 1, (
      f"Images have inconsistent shapes: "
      f"{', '.join({'x'.join(map(str, im.shape)) for im in image})}")

  image = np.concatenate([frame[None, ..., None] * [[[1.,1.,1.]]] for frame in image])
  image = (255 * image / image.max()).astype(np.uint8)
  # image.shape # == (frames, length, width, channels)
  return image
