# -*- coding: utf-8 -*-
"""evaluation1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ecell/bioimage_workflows/blob/master/evaluation1.ipynb
"""

import argparse

parser = argparse.ArgumentParser(description='evaluation1 step')
parser.add_argument('--generated_data', type=str, default="/tmp/foobar")
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=5)
parser.add_argument('--threshold', type=float, default=50.0)

args = parser.parse_args()

import mlflow
mlflow.start_run(run_name="evaluation1")

generated_data = args.generated_data
num_samples = args.num_samples
num_frames = args.num_frames
threshold = args.threshold

from mlflow import log_metric, log_param, log_artifacts
log_param("generated_data", generated_data)
log_param("num_samples", num_samples)
log_param("num_frames", num_frames)
log_param("threshold", threshold)

import numpy

import pathlib
inputpath = pathlib.Path(generated_data)
# artifacts = pathlib.Path("./artifacts")
# artifacts.mkdir(parents=True, exist_ok=True)

import scopyon
config = scopyon.Configuration(filename=inputpath / "config.yaml")
pixel_length = config.default.detector.pixel_length / config.default.magnification

rates = numpy.zeros(4, dtype=int)

closest = []
for i in range(num_samples):
    true_data_ = numpy.load(inputpath / f"true_data{i:03d}.npy")
    t = true_data_[0, 0]
    true_data = [[true_data_[0, 1: ]]]
    for row in true_data_[1: ]:
        if row[0] == t:
            true_data[-1].append(row[1: ])
        else:
            t = row[0]
            true_data[-1] = numpy.asarray(true_data[-1])
            true_data.append([row[1: ]])
    else:
        true_data[-1] = numpy.asarray(true_data[-1])

    spots_ = numpy.load(inputpath / f"spots{i:03d}.npy")
    t = spots_[0, 0]
    spots = [[spots_[0, 1: ]]]
    for row in spots_[1: ]:
        if row[0] == t:
            spots[-1].append(row[1: ])
        else:
            t = row[0]
            spots[-1] = numpy.asarray(spots[-1])
            spots.append([row[1: ]])
    else:
        spots[-1] = numpy.asarray(spots[-1])

    for true_data_, spots_ in zip(true_data, spots):
        data = true_data_[:, 3: 5]
        for spot in spots_:
            distance = data - spot[0: 2]
            idx = (distance ** 2).sum(axis=1).argmin()
            closest.append(distance[idx])
            
            distance = numpy.sqrt(distance[idx] ** 2).sum()
            if distance < threshold:
                rates[0] += 1
            else:
                rates[1] += 1

        for spot in data:
            distance = spots_[:, : 2] - spot
            distance = (distance ** 2).sum(axis=1)
            idx = distance.argmin()
            distance = numpy.sqrt(distance[idx])
            if distance < threshold:
                rates[2] += 1
            else:
                rates[3] += 1

closest = numpy.asarray(closest).T

x_mean, y_mean = numpy.average(closest[0]), numpy.average(closest[1])
x_std, y_std = numpy.std(closest[0]), numpy.std(closest[1])
print(f"Average along x-axis = {x_mean:+.5f} pixels, std = {x_std}")
print(f"Average along y-axis = {y_mean:+.5f} pixels, std = {y_std}")

log_metric("x_mean", x_mean)
log_metric("y_mean", y_mean)
log_metric("x_std", x_std)
log_metric("y_std", y_std)

import plotly.express as px
w = h = 1
H, xedges, yedges = numpy.histogram2d(x=closest[0], y=closest[1], bins=41, range=[[-w, +w], [-h, +h]])
fig = px.imshow(H, x=(xedges[: -1]+xedges[1: ])*0.5, y=(yedges[: -1]+yedges[1: ])*0.5)
#fig.show()
fig.write_image(generated_data + "/evaluation1_1.png")

r = 6
idx = 0
shapes = [dict(x=row[0], y=row[1], sigma=r, color='green')
        for row in true_data[idx][:, [3, 4]]]
shapes += [dict(x=spot[0], y=spot[1], sigma=r, color='red')
        for spot in spots[idx]]
scopyon.Image(numpy.load(inputpath / "images{:03d}.npy".format(num_samples - 1))[idx]).show(shapes=shapes)

r = rates[: 2].sum() / rates[2: ].sum()
miss_count = rates[1] / rates[: 2].sum()
missing = rates[3] / rates[2: ].sum()
print(f"The ratio between detected and expected = {r:.3f}")
print(f"The fraction of miss counted spots = {miss_count:.3f}")
print(f"The fraction of spots not detected = {missing:.3f}")

log_metric("r", r)
log_metric("miss_count", miss_count)
log_metric("missing", missing)

#log_artifacts("./artifacts")
log_artifacts(generated_data)
mlflow.end_run()
