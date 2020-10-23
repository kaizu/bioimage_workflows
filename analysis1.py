import argparse

import mlflow
from mlflow import log_metric, log_param, log_artifacts

entrypoint = "analysis1"
parser = argparse.ArgumentParser(description='analysis1 step')
parser.add_argument('--generation', type=str, default="")
parser.add_argument('--min_sigma', type=int, default=1)
parser.add_argument('--max_sigma', type=int, default=4)
parser.add_argument('--threshold', type=float, default=50.0)
parser.add_argument('--overlap', type=float, default=0.5)

args = parser.parse_args()

active_run = mlflow.start_run()
mlflow.set_tag("mlflow.runName", entrypoint)

generation = args.generation
min_sigma = args.min_sigma
max_sigma = args.max_sigma
threshold = args.threshold
overlap = args.overlap

for key, value in vars(args).items():
    log_param(key, value)

generation_run = mlflow.tracking.MlflowClient().get_run(generation)
num_samples = generation_run.data.params["num_samples"]
num_frames = generation_run.data.params["num_frames"]
interval = generation_run.data.params["interval"]

# nproc = 1
# 
# import numpy
# timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)
# 
# import pathlib
# inputpath = pathlib.Path(generated_data.replace("file://", ""))
# artifacts = pathlib.Path(generated_data.replace("file://", ""))
# artifacts.mkdir(parents=True, exist_ok=True)
# 
# import scopyon
# 
# import warnings
# warnings.simplefilter('ignore', RuntimeWarning)
# 
# for i in range(num_samples):
#     imgs = [scopyon.Image(data) for data in numpy.load(inputpath / f"images{i:03d}.npy")]
#     spots = [
#         scopyon.analysis.spot_detection(
#             img.as_array(), processes=nproc,
#             min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
#         for img in imgs]
# 
#     spots_ = []
#     for t, data in zip(timepoints, spots):
#         spots_.extend(([t] + list(row) for row in data))
#     spots_ = numpy.array(spots_)
#     numpy.save(artifacts / f"spots{i:03d}.npy", spots_)
#     
#     print("{} spots are detected in {} frames.".format(len(spots_), len(imgs)))
# 
# warnings.resetwarnings()
# 
# #!ls ./artifacts
# 
# #log_artifacts("./artifacts")
# log_artifacts(generated_data)

mlflow.end_run()
