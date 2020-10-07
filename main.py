import mlflow
mlflow.start_run(run_name="main", nested=True)

num_samples = 1
num_frames = 5

from mlflow import log_metric, log_param, log_artifacts
log_param("num_samples", num_samples)
log_param("num_frames", num_frames)

import pathlib
artifacts = pathlib.Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)

import papermill as pm
_ = pm.execute_notebook(
   'generation.ipynb',
   str(artifacts / 'generation.ipynb'),
   parameters=dict(num_samples=num_samples, num_frames=num_frames)
)

_ = pm.execute_notebook(
   'analysis1.ipynb',
   str(artifacts / 'analysis1.ipynb'),
   parameters=dict(num_samples=num_samples, num_frames=num_frames)
)

_ = pm.execute_notebook(
   'analysis2.ipynb',
   str(artifacts / 'analysis2.ipynb'),
   parameters=dict(num_samples=num_samples, num_frames=num_frames)
)

log_artifacts("./artifacts")

_ = pm.execute_notebook(
   'evaluation1.ipynb',
   str(artifacts / 'evaluation1.ipynb'),
   parameters=dict(num_samples=num_samples)
)

mlflow.end_run()
