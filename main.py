import mlflow
from mlflow import log_metric, log_param, log_artifacts
import pathlib
artifacts = pathlib.Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)

with mlflow.start_run(nested=True):
    num_samples = 1
    num_frames = 5

    for thre in [50.0, 60.0, 70.0, 80.0, 90.0]:

        log_param("num_samples", num_samples)
        log_param("num_frames", num_frames)

        #    import papermill as pm
        #    _ = pm.execute_notebook(
        #       'generation.ipynb',
        #       str(artifacts / 'generation.ipynb'),
        #       parameters=dict(num_samples=num_samples, num_frames=num_frames)
        #    )
        exec(open("generation.py").read())

        #    _ = pm.execute_notebook(
        #       'analysis1.ipynb',
        #       str(artifacts / 'analysis1.ipynb'),
        #       parameters=dict(num_samples=num_samples, num_frames=num_frames)
        #    )
        #exec(open("analysis1.py").read())
        subprocess.run(["python", "analysis1.py", "--threshold", thre])

#         _ = pm.execute_notebook(
#          'analysis2.ipynb',
#          str(artifacts / 'analysis2.ipynb'),
#          parameters=dict(num_samples=num_samples, num_frames=num_frames)
#         )
        #exec(open("analysis2.py").read())
        subprocess.run(["python", "analysis2.py", "--threshold", thre])

        log_artifacts("./artifacts")

#         _ = pm.execute_notebook(
#          'evaluation1.ipynb',
#          str(artifacts / 'evaluation1.ipynb'),
#          parameters=dict(num_samples=num_samples)
#         )
        #exec(open("evaluation1.py").read())
        subprocess.run(["python", "evaluation1.py", "--threshold", thre])
