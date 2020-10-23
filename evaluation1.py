import argparse
import pathlib

import mlflow
from mlflow import log_metric, log_param, log_artifacts

entrypoint = "evaluation1"
parser = argparse.ArgumentParser(description='evaluation1 step')
parser.add_argument('--generation', type=str, default="")
parser.add_argument('--analysis1', type=str, default="")
# parser.add_argument('--analysis2', type=str, default="")
parser.add_argument('--max_distance', type=float, default=50.0)
args = parser.parse_args()

active_run = mlflow.start_run()
mlflow.set_tag("mlflow.runName", entrypoint)

generation = args.generation
analysis1 = args.analysis1
# analysis2 = args.analysis2
max_distance = args.max_distance

for key, value in vars(args).items():
    log_param(key, value)

client = mlflow.tracking.MlflowClient()
generation_run = client.get_run(generation)
num_samples = int(generation_run.data.params["num_samples"])
analysis1_run = client.get_run(analysis1)
# analysis2_run = client.get_run(analysis2)
generation_artifacts = pathlib.Path(client.download_artifacts(generation, "."))
analysis1_artifacts = pathlib.Path(client.download_artifacts(analysis1, "."))
# analysis2_artifacts = pathlib.Path(client.download_artifacts(analysis2, "."))

import tempfile
artifacts = pathlib.Path(tempfile.mkdtemp()) / "artifacts"
artifacts.mkdir(parents=True, exist_ok=True)

#XXX: HERE

import numpy

import scopyon
config = scopyon.Configuration(filename=generation_artifacts / "config.yaml")
pixel_length = config.default.detector.pixel_length / config.default.magnification

rates = numpy.zeros(4, dtype=int)

closest = []
for i in range(num_samples):
    true_data_ = numpy.load(generation_artifacts / f"true_data{i:03d}.npy")
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

    spots_ = numpy.load(analysis1_artifacts / f"spots{i:03d}.npy")
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
            if distance < max_distance:
                rates[0] += 1
            else:
                rates[1] += 1

        for spot in data:
            distance = spots_[:, : 2] - spot
            distance = (distance ** 2).sum(axis=1)
            idx = distance.argmin()
            distance = numpy.sqrt(distance[idx])
            if distance < max_distance:
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

# import plotly.express as px
# w = h = 1
# H, xedges, yedges = numpy.histogram2d(x=closest[0], y=closest[1], bins=41, range=[[-w, +w], [-h, +h]])
# fig = px.imshow(H, x=(xedges[: -1]+xedges[1: ])*0.5, y=(yedges[: -1]+yedges[1: ])*0.5)
# fig.show()
# fig.write_image(str(artifacts / "heatmap1.png"))

# r = 6
# idx = 0
# shapes = [dict(x=row[0], y=row[1], sigma=r, color='green')
#         for row in true_data[idx][:, [3, 4]]]
# shapes += [dict(x=spot[0], y=spot[1], sigma=r, color='red')
#         for spot in spots[idx]]
# scopyon.Image(numpy.load(generation_artifacts / "images{:03d}.npy".format(num_samples - 1))[idx]).show(shapes=shapes)

r = rates[: 2].sum() / rates[2: ].sum()
miss_count = rates[1] / rates[: 2].sum()
missing = rates[3] / rates[2: ].sum()
print(f"The ratio between detected and expected = {r:.3f}")
print(f"The fraction of miss counted spots = {miss_count:.3f}")
print(f"The fraction of spots not detected = {missing:.3f}")

log_metric("r", r)
log_metric("miss_count", miss_count)
log_metric("missing", missing)

#XXX: THERE

log_artifacts(str(artifacts))
mlflow.end_run()

import shutil
shutil.rmtree(str(artifacts))
