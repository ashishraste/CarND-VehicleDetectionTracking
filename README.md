# Vehicle Detection and Tracking

Vehicle detection and tracking in a video-feed.


## Overview

This project is part of [Udacity's Self-Driving Car Nanodegree program](https://www.udacity.com/drive)
and much of the source comes from the program's lecture notes and quizzes.

Following steps were performed to setup the vehicle detection and tracking
pipeline for a video-feed obtained from a front-facing camera.

1. Data collection
2. Feature selection: Histogram of Oriented Gradients
3. Training the classifier
4. Sliding window search


## Dependencies
1. Python-3.5
2. OpenCV-Python
3. Moviepy
4. Numpy
5. Matplotlib
6. Pickle


## Running the car detection and tracking pipeline
Switch to the source directory and run the vehicle-detection script.
This will take the video (project_video.mp4)[project_video.mp4]
as input, runs the detection/tracking pipeline on it, and finally saves the output
to [detected_vehicles.mp4](detected_vehicles.mp4) in the parent directory.
```bash
cd src
python vehicle_detection.py
```


## Directory Layout
* src : Contains the following source files.
  * vehicle_detection.py : Contains the detection and tracking pipeline, runs
  the pipeline against an input video.
  * vehicle_classifier.py : Trains the SVM classifier with features extracted
  from the dataset available in training_images/ directory.
  * tracking_util.py : Contains utility methods such as feature-extraction,
  finding sliding-windows/vehicles, etc.
  * vis_util.py : Sample methods for visualizing different stages of the pipeline.
  * pickle_images.py : Serializes train-dataset images by storing their numpy-array
   representation in a pickle-file.
* training_images : Train dataset containing vehicle and non-vehicle images.
* test_images : Images to test different stages of the detection pipeline.
* output_images : Output from pipeline stages run on test-images.
* detected_video.mp4 : Output from the detection pipeline run on the input video.
* model.p : Pickled SVM classifier model.


## Vehicle detection and tracking pipeline description
Let's go through the steps in the pipeline listed above. Please note that rest
of this document refers to the source in `src/` directory.


### Data collection
Labeled images containing [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
provided by Udacity were collected. These images were selected from
[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
and [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).

Numpy arrays of the collected images was pickled for reuse. Script
`pickle_images.py` performs this routine, saving the pickled file in `image_data.p`
in the parent directory.
```bash
cd src
python pickle_images.py
```

- Number of vehicle train images: 6941
- Number of non-vehicle train images: 8968

Sample images present in the dataset:

![sample-images](output_images/sample_images.png)


### Feature selection:  Histogram of Oriented Gradients (HOG)


### Training the classifier
Accuracy of the classifier: 0.9915
