# -*- coding: utf-8 -*-
import os
from . import data
import numpy as np
import pandas as pd
from skimage.io import imread
from collections import Counter


def get_ground_truth(path, testset_name='TestSet1'):
    # we need the load the tracking labels from a differen file as
    # the segmentation coordinates.
    ground_truth_segmentation = pd.read_csv(
        f'{path}/YIT-Benchmark2/{testset_name}/GroundTruth/GroundTruth_Segmentation.csv',
        sep=', ',  # there's an awkward space next to the commas
        engine='python',
    ).drop('Cell_colour', axis=1).rename(
        {'Frame_number': 'frame', 'Position_X': 'x', 'Position_Y': 'y'}, axis=1).reindex()

    ground_truth_tracking = pd.read_csv(
        f'{path}/YIT-Benchmark2/{testset_name}/GroundTruth/GroundTruth_Tracking.csv',
        sep=', ',  # there's an awkward space next to the commas
        engine='python',
    ).rename(
        {'Frame_number': 'frame', 'Unique_cell_number': 'cell'}, axis=1).reindex()
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


def get_test_movie(path, testset_name='TestSet1'):
    filenames = data.load_data(f'{path}/YIT-Benchmark2/{testset_name}/RawData', ff='.tif')
    assert len(filenames) > 0, "No images were found"

    image = [imread(filename) for filename in filenames]

    assert len({im.shape for im in image}) == 1, (
        f"Images have inconsistent shapes: "
        f"{', '.join({'x'.join(map(str, im.shape)) for im in image})}")

    image = np.concatenate([frame[None, ..., None] * [[[1., 1., 1.]]] for frame in image])
    image = (255 * image / image.max()).astype(np.uint8)
    # image.shape # == (frames, length, width, channels)
    return image


def load_yit_segmentation_masks(path, ground_truth=None):
    masks = np.concatenate([np.load(f'{path}/{fn}')[None] for fn in sorted(os.listdir(path))])
    if ground_truth is None:
        frames = sum(([frame] * mask.max() for frame, mask in enumerate(masks)), [])
        masks = np.concatenate([
            np.arange(1, mask.max() + 1)[:, None, None] == mask[None]
            for mask in tqdm(masks)
            if mask.max() > 0
        ])
        return pd.DataFrame({'frame': frames, 'mask': np.arange(len(masks))}), masks
    else:
        frame, y, x = ground_truth[['frame', 'y', 'x']].round().values.astype(int).T
        mask_number = masks[frame, y, x]
        mask_number[mask_number == 0] = -1  #

        # Checks if the masks aren't re-used or ignored.
        mask_use_count = Counter(zip(frame, mask_number))
        reused_masks = [(frame, number) for (frame, number), count in mask_use_count.items() if
                        count > 1 and number >= 0]
        assert len(
            reused_masks), f"Some annotation masks overlap with more than one ground_truth sample coordinates at mask frames and with numbers: {reused_masks}"
        ignored_masks = [(frame, mask) for frame, mask in enumerate(masks) for number in range(1, mask.max() + 1) if
                         mask_use_count[frame, number] == 0]
        assert len(
            ignored_masks) == 0, f"Some annotation masks are not covered by any of the ground_truth sample coordinates at mask frames and with numbers: {ignored_masks}"
        # ground_truth samples without a mask are allowed and actually common (10%) in the annotations of YeastNet's authors, these are typically small cells.

        masks = masks[frame] == mask_number[:, None, None]

        ground_truth['mask'] = np.arange(len(masks))
        return ground_truth, masks