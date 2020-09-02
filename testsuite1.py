# -*- coding: utf-8 -*-
"""testsuite1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eijSUQYBqgRSsGD_m08VTgXU2MEVfNnB

# Test spot detection

This is a test suite.
"""

import scopyon
import mlflow

"""Set physical parameters."""

config = scopyon.DefaultConfiguration()
config.update("""
default:
    magnification: 360
    detector:
        exposure_time: 0.033
""")

pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5

"""Set the number of processes to enable `multiprocessing`:"""

#config.environ.processes = 2

"""Prepare for generating inputs."""

import numpy

with mlflow.start_run():
    foo = 123
    rng = numpy.random.RandomState(foo)
    N = 1000

    """Collect data."""

    res = []
    for _ in range(10):
        inputs = rng.uniform(-L_2, +L_2, size=(N, 2))
        img, infodict = scopyon.form_image(inputs, config=config, rng=rng, full_output=True)
        spots = scopyon.analysis.spot_detection(
            img.as_array(), min_sigma=1, max_sigma=4, threshold=40.0, overlap=0.5)
        res.append((inputs, img, infodict, spots))

    """Find closest true positions from spots detected."""

    closest = []
    for inputs, img, infodict, spots in res:
        data = numpy.array([(data[2], data[3]) for data in infodict['true_data'].values()])
        for spot in spots:
            distance = data - spot[0: 2]
            idx = (distance ** 2).sum(axis=1).argmin()
            closest.append(distance[idx])
    closest = numpy.array(closest).T

    """Calculate average and std for the accuracy of the spot detection method."""

    print(f"Average along x-axis = {numpy.average(closest[0]):+.5f} pixels, std = {numpy.std(closest[0])}")
    print(f"Average along y-axis = {numpy.average(closest[1]):+.5f} pixels, std = {numpy.std(closest[1])}")

    """Show the heat map."""

    import plotly.express as px
    H, xedges, yedges = numpy.histogram2d(x=closest[0], y=closest[1], bins=41, range=[[-2, +2], [-2, +2]])
    fig = px.imshow(H, x=(xedges[: -1]+xedges[1: ])*0.5, y=(yedges[: -1]+yedges[1: ])*0.5)
    fig.show()
    
    mlflow.log_param("foo", foo)
