# -*- coding: utf-8 -*-
"""analysis1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ecell/bioimage_workflows/blob/master/analysis1.ipynb
"""

import argparse
parser = argparse.ArgumentParser(description='analysis1 step')
parser.add_argument('--threshold')
args = parser.parse_args()

# import mlflow
# mlflow.start_run(run_name="analysis1", nested=True)

num_samples = 3
interval = 33.0e-3
num_frames = 100
min_sigma = 1
max_sigma = 4
threshold = float(args.threshold)
overlap = 0.5
generated_data = "./artifacts"

from mlflow import log_metric, log_param, log_artifacts
log_param("num_samples", num_samples)
log_param("interval", interval)
log_param("num_frames", num_frames)
log_param("min_sigma", min_sigma)
log_param("max_sigma", max_sigma)
log_param("threshold", threshold)
log_param("overlap", overlap)

nproc = 8

import numpy
timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)

import pathlib
inputpath = pathlib.Path(generated_data.replace("file://", ""))
artifacts = pathlib.Path(generated_data.replace("file://", ""))
artifacts.mkdir(parents=True, exist_ok=True)

import scopyon

import warnings
warnings.simplefilter('ignore', RuntimeWarning)

for i in range(num_samples):
    imgs = [scopyon.Image(data) for data in numpy.load(inputpath / f"images{i:03d}.npy")]
    spots = [
        scopyon.analysis.spot_detection(
            img.as_array(), processes=nproc,
            min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
        for img in imgs]

    spots_ = []
    for t, data in zip(timepoints, spots):
        spots_.extend(([t] + list(row) for row in data))
    spots_ = numpy.array(spots_)
    numpy.save(artifacts / f"spots{i:03d}.npy", spots_)
    
    print("{} spots are detected in {} frames.".format(len(spots_), len(imgs)))

warnings.resetwarnings()

#!ls ./artifacts

log_artifacts("./artifacts")
# mlflow.end_run()
