import subprocess
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id
from mlflow import log_metric, log_param, log_artifacts
import pathlib
import argparse

# _already_ran and _get_or_run code from bellow.
# mlflow/main.py at master · mlflow/mlflow
# https://github.com/mlflow/mlflow/blob/master/examples/multistep_workflow/main.py
# 
# modify little bit at compare param with force string
def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if str(run_value) != str(param_value):
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, git_commit, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

"""Prepare for generating inputs."""
parser = argparse.ArgumentParser(description='analysis1 step')
parser.add_argument('--threshold', type=float, default=50.0)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=5)
args = parser.parse_args()

num_samples = int(args.num_samples)
num_frames = int(args.num_frames)
threshold = float(args.threshold)

with mlflow.start_run(run_name="main", nested=True) as active_run:
    # log param
    log_param("threshold", threshold)
    log_param("num_samples", num_samples)
    log_param("num_frames", num_frames)
    # artifacts
    artifacts = pathlib.Path("./artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    # check git version
    git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
    # generation
    generation_run = _get_or_run("generation", {"num_samples":num_samples, "num_frames":num_frames}, git_commit)
    #generation_run = mlflow.run(".", "generation", parameters={"num_samples":num_samples, "num_frames":num_frames})
    # analysis1
    analysis1_run = _get_or_run("analysis1", {"threshold":threshold, "num_samples":num_samples, "num_frames":num_frames}, git_commit)
    #analysis1_run = mlflow.run(".", "analysis1", parameters={"threshold":threshold, "num_samples":num_samples})
    # analysis2
    analysis2_run = _get_or_run("analysis2", {"threshold":threshold, "num_samples":num_samples, "num_frames":num_frames}, git_commit)
    #analysis2_run = mlflow.run(".", "analysis2", parameters={"threshold":threshold, "num_samples":num_samples})

#     #log_artifacts("./artifacts")
    # evaluation1
    evaluation1_run = _get_or_run("evaluation1", {"threshold":threshold, "num_samples":num_samples, "num_frames":num_frames}, git_commit)
    #evaluation1_run = mlflow.run(".", "evaluation1", parameters={"threshold":threshold, "num_samples":num_samples})
