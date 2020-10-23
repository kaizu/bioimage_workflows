import argparse
import pathlib

import mlflow
from mlflow import log_metric, log_param, log_artifacts

entrypoint = "analysis2"
parser = argparse.ArgumentParser(description='analysis2 step')
parser.add_argument('--generation', type=str, default="")
parser.add_argument('--analysis1', type=str, default="")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--max_distance', type=float, default=50.0)
args = parser.parse_args()

active_run = mlflow.start_run()
mlflow.set_tag("mlflow.runName", entrypoint)

generation = args.generation
analysis1 = args.analysis1
seed = args.seed
max_distance = args.max_distance

for key, value in vars(args).items():
    log_param(key, value)

client = mlflow.tracking.MlflowClient()
generation_run = client.get_run(generation)
analysis1_run = client.get_run(analysis1)
num_samples = int(generation_run.data.params["num_samples"])
num_frames = int(generation_run.data.params["num_frames"])
interval = float(generation_run.data.params["interval"])
generation_artifacts = pathlib.Path(client.download_artifacts(generation, "."))
analysis1_artifacts = pathlib.Path(client.download_artifacts(analysis1, "."))

import tempfile
artifacts = pathlib.Path(tempfile.mkdtemp()) / "artifacts"
artifacts.mkdir(parents=True, exist_ok=True)

#XXX: HERE

import scopyon
config = scopyon.Configuration(filename=generation_artifacts / "config.yaml")
pixel_length = config.default.detector.pixel_length / config.default.magnification

import numpy

def trace_spots(spots, max_distance=numpy.inf, ndim=2):
    observation_vec = []
    lengths = []
    for i in range(len(spots[0])):
        iprev = i
        for j in range(1, len(spots)):
            displacements = numpy.power(spots[j][:, : ndim] - spots[j - 1][iprev, : ndim], 2).sum(axis=1)
            inext = displacements.argmin()
            displacement = numpy.sqrt(displacements[inext])
            if displacement > max_distance:
                if j > 1:
                    lengths.append(j - 1)
                break
            intensity = spots[j - 1][iprev, ndim]
            observation_vec.append([displacement, intensity])
            iprev = inext
        else:
            lengths.append(len(spots) - 1)
    return observation_vec, lengths

observation_vec = []
lengths = []
ndim = 2
for i in range(num_samples):
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
    # print(spots)

    observation_vec_, lengths_ = trace_spots(spots, max_distance=max_distance, ndim=ndim)
    observation_vec.extend(observation_vec_)
    lengths.extend(lengths_)
observation_vec = numpy.array(observation_vec)

print(len(lengths), sum(lengths))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])
fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=30, histnorm='probability'), row=1, col=1)
fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=30, histnorm='probability'), row=1, col=2)
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75, showlegend=False)
# fig.show()
fig.write_image(str(artifacts / "histogram1.png"))

from scopyon.analysis import PTHMM

rng = numpy.random.RandomState(seed)
model = PTHMM(n_diffusivities=3, n_oligomers=1, n_iter=100, random_state=rng)
model.fit(observation_vec, lengths)

print("diffusivities=\n", model.diffusivities_)
print("D=\n", pixel_length ** 2 * model.diffusivities_ / interval / 1e-12)

print("intensity_means=", model.intensity_means_)
print("intensity_vars=", model.intensity_vars_)

print("startprob=\n", model.startprob_)

P = model.transmat_
k = -numpy.log(1 - P) / interval
k.ravel()[:: k.shape[0] + 1] = 0.0
print("transmat=\n", model.transmat_)
print("state_transition_matrix=\n", k)

expected_vec = numpy.zeros((sum(lengths), 2), dtype=observation_vec.dtype)
for i in range(len(lengths)):
    X_, Z_ = model.sample(lengths[i])
    expected_vec[sum(lengths[: i]): sum(lengths[: i + 1])] = X_

fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])
fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=30, histnorm='probability density'), row=1, col=1)
fig.add_trace(go.Histogram(x=expected_vec[:, 0], nbinsx=30, histnorm='probability density'), row=1, col=1)
fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=30, histnorm='probability density'), row=1, col=2)
fig.add_trace(go.Histogram(x=expected_vec[:, 1], nbinsx=30, histnorm='probability density'), row=1, col=2)
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75, showlegend=False)
# fig.show()
fig.write_image(str(artifacts / "histogram2.png"))

#XXX: THERE

log_artifacts(str(artifacts))
mlflow.end_run()

import shutil
shutil.rmtree(str(artifacts))
