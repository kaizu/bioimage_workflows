import pathlib
import argparse
import itertools

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
    git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    mlflow.set_tag("mlflow.runName", entrypoint)
    for key, value in vars(args).items():
        log_param(key, value)

    generation_run = _get_or_run("generation", {"num_samples": num_samples, "num_frames": num_frames}, git_commit)
    analysis1_run = _get_or_run("analysis1", {"generation": generation_run.info.run_id, "threshold": threshold, "min_sigma": min_sigma}, git_commit)
    analysis2_run = _get_or_run("analysis2", {"generation": generation_run.info.run_id, "analysis1": analysis1_run.info.run_id, "threshold":threshold}, git_commit)
    evaluation1_run = _get_or_run("evaluation1", {"generation": generation_run.info.run_id, "analysis1": analysis1_run.info.run_id}, git_commit)

    for run_obj in (generation_run, analysis1_run, analysis2_run, evaluation1_run):
        for key, value in run_obj.data.metrics.items():
            log_metric(key, value)
