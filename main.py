import subprocess
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import pathlib

for thre in [50.0, 60.0, 70.0, 80.0, 90.0]:

    with mlflow.start_run(nested=True):
        num_samples = 1
        num_frames = 5
        
        artifacts = pathlib.Path("./artifacts")
        artifacts.mkdir(parents=True, exist_ok=True)

        log_param("num_samples", num_samples)
        log_param("num_frames", num_frames)

        #    import papermill as pm
        #    _ = pm.execute_notebook(
        #       'generation.ipynb',
        #       str(artifacts / 'generation.ipynb'),
        #       parameters=dict(num_samples=num_samples, num_frames=num_frames)
        #    )
        generation_run = mlflow.run(".", "generation")

        #    _ = pm.execute_notebook(
        #       'analysis1.ipynb',
        #       str(artifacts / 'analysis1.ipynb'),
        #       parameters=dict(num_samples=num_samples, num_frames=num_frames)
        #    )
        analysis1_run = mlflow.run(".", "analysis1", parameters={"threshold":thre})
        
#         _ = pm.execute_notebook(
#          'analysis2.ipynb',
#          str(artifacts / 'analysis2.ipynb'),
#          parameters=dict(num_samples=num_samples, num_frames=num_frames)
#         )
        #exec(open("analysis2.py").read())
        analysis2_run = mlflow.run(".", "analysis2", parameters={"threshold":thre})

        log_artifacts("./artifacts")

#         _ = pm.execute_notebook(
#          'evaluation1.ipynb',
#          str(artifacts / 'evaluation1.ipynb'),
#          parameters=dict(num_samples=num_samples)
#         )
        #exec(open("evaluation1.py").read())
        evaluation1_run = mlflow.run(".", "evaluation1", parameters={"threshold":thre})
