# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
      name='yeastcells',
      version=__import__('yeastcells').__version__,
      
      description='Computer vision based yeast cell detection.',
      long_description='Yeast cell detection using a deep convolutional network to detect cells, and classic computer vision, DBSCAN clustering and machine learning is used for tracking. Current state allows detection of cells, clustering over time to determine which are the same cell, finding cell boudnary using seam carving and pruning false positives.',

      url='https://git.web.rug.nl/P301081/yeastcells-detection-maskrcnn/',
      
      author='Herbert Kruitbosch, Yasmin Mzayek',
      author_email='H.T.Kruitbosch@rug.nl, y.mzayek@rug.nl',
      license='Only for use within the University of Groningen and only with permission from the authors.',
      
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: Only with authors permission',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
      ],
      
      # What does your project relate to?
      keywords='yeast cell detection, microscopy images, tif, tiff, mask R-CNN, deep learning, image segmentation, tracking, computer vision, DBSCAN',
      packages=find_packages(exclude=[]),
      include_package_data=True,
      
      install_requires=[
        'detectron2>=0.2.1+cu102',
        'scikit-image>=0.17.2',
        'scikit-learn>=0.23.2',
        'opencv-contrib-python>=4.3.0.36',
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'Shapely>=1.7.0'
        'tqdm>=4.48.2'
      ],
      extras_require={
        'dev': [],
        'test': [],
      },
      zip_safe=False,
)

      