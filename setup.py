# -*- coding: utf-8 -*-
from setuptools import setup

setup(
      name='yeastcells',
      version=__import__('yeastcells').__version__,

      description='Computer vision based yeast cell synthetic data creation and network training, and detection and tracking pipeline.',
      long_description='Yeast cell detection using a deep convolutional network trained on synthetic data to detect, segment, and track cells. Current state allows segmentation of cells, clustering over time to determine which are the same cell, finding cell boudnaries, plotting segmented and tracked cells in figures and animations, and outputting masks and their area, position, and pixel intensity.',
      
      url='https://github.com/ymzayek/yeastcells-detection-maskrcnn/',

      author='Herbert Kruitbosch, Yasmin Mzayek',
      author_email='H.T.Kruitbosch@rug.nl, y.mzayek@rug.nl',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
      ],
      keywords='yeast cell detection, microscopy images, tif, tiff, mask R-CNN, deep learning, image segmentation, tracking, computer vision, DBSCAN',
      
      packages=['yeastcells'],
      install_requires=[
        'scikit-image>=0.17.2',
        'scikit-learn>=0.23.2',
        'opencv-python>=4.4.0.46',
        'opencv-contrib-python>=4.4.0.46',
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'Shapely>=1.7.0'
        'tqdm>=4.51.0',
        'imgaug>=0.4.0',
        'pandas>=1.1.4',
      ],
      zip_safe=True,
)

      