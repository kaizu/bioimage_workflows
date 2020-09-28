import click
import os
import six

import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id

@click.command()
@click.option("--seed", default=123, type=int)
@click.option("--n", default=1000, type=int)
@click.option("--magnification", default=360, type=int)
@click.option("--exposure-time", default=0.033, type=float)
def workflow(seed, magnification, exposure_time):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        submitted_generation_run = mlflow.run(".", "generation", parameters={"seed": seed, "magnification": magnification, "exposure_time": exposure_time})
        generation_run = mlflow.tracking.MlflowClient().get_run(submitted_generation_run.run_id)
        image_path = generated_run.info.artifact_uri
        
        submitted_analysis1_run = mlflow.run(".", "analysis1", parameters={"generated_images": image_path})
        analysis1_run = mlflow.tracking.MlflowClient().get_run(submitted_analysis1_run.run_id)
        analysis_path = analysis1_run.info.artifact_uri
        
        analysis2_run = mlflow.run(".", "analysis2", parameters={"analysis1_result": analysis_path})

if __name__ == "__main__":
    workflow()
    
