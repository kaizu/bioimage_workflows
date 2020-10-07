# bioimage_workflows

## How to run this workflow

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
1. run `pip install mlflow` with Miniconda's pip
1. run `tmux` to create multiple shells and save the processes
1. run `mlflow ui -h 0.0.0.0` and create a new tmux pane
1. move to the another tmux pane and run `mlflow run https://github.com/ecell/bioimage_workflows.git`

## How to run specific workflow which is written as entrypoint

If you want to execute `analysis1` in entrypoint, command is following

`mlflow run -e analysis1 https://github.com/ecell/bioimage_workflows.git -P num_samples=1 -P num_frames=5`
