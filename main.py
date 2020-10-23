import pathlib
import argparse

import mlflow
from mlflow.utils import mlflow_tags
from mlflow import log_metric, log_param, log_artifacts
from mlflow_utils import _get_or_run

entrypoint = "main"
parser = argparse.ArgumentParser(description='analysis1 step')
parser.add_argument('--threshold', type=float, default=50.0)
parser.add_argument('--min_sigma', type=int, default=1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=5)
args = parser.parse_args()

threshold = args.threshold
min_sigma = args.min_sigma
num_samples = args.num_samples
num_frames = args.num_frames

with mlflow.start_run(nested=True) as active_run:
    mlflow.set_tag("mlflow.runName", entrypoint)
    for key, value in vars(args).items():
        log_param(key, value)

    # artifacts
    #artifacts = pathlib.Path("./artifacts")
    #artifacts.mkdir(parents=True, exist_ok=True)
    # check git version
    git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    # generation
    generation_run = _get_or_run("generation", {"num_samples": num_samples, "num_frames": num_frames}, git_commit)
    # artifactsPath = generation_run.data.params["artifactsPath"]
    # #generation_run = mlflow.run(".", "generation", parameters={"num_samples":num_samples, "num_frames":num_frames})
    # # analysis1
    analysis1_run = _get_or_run("analysis1", {"generation": generation_run.info.run_id, "threshold": threshold, "min_sigma": min_sigma}, git_commit)
    # #analysis1_run = mlflow.run(".", "analysis1", parameters={"threshold":threshold, "num_samples":num_samples})
    # # analysis2
    # analysis2_run = _get_or_run("analysis2", {"threshold":threshold, "num_samples":num_samples, "num_frames":num_frames}, git_commit)
    # #analysis2_run = mlflow.run(".", "analysis2", parameters={"threshold":threshold, "num_samples":num_samples})

#   #   #log_artifacts("./artifacts")
    # # evaluation1
    # evaluation1_run = _get_or_run("evaluation1", {"threshold":threshold, "num_samples":num_samples, "num_frames":num_frames}, git_commit)
    # #evaluation1_run = mlflow.run(".", "evaluation1", parameters={"threshold":threshold, "num_samples":num_samples})
    # 

    for key, value in generation_run.data.metrics.items():
        log_metric(key, value)
    # log_metric("x_mean", float(evaluation1_run.data.metrics["x_mean"]))
    # log_metric("y_mean", float(evaluation1_run.data.metrics["y_mean"]))
    # log_metric("x_std", float(evaluation1_run.data.metrics["x_std"]))
    # log_metric("y_std", float(evaluation1_run.data.metrics["y_std"]))
    # log_metric("r", float(evaluation1_run.data.metrics["r"]))
    # log_metric("miss_count", float(evaluation1_run.data.metrics["miss_count"]))
    # log_metric("missing", float(evaluation1_run.data.metrics["missing"]))
