thresholds=(50.0 60.0 70.0 80.0 90.0)
for t in ${thresholds[@]}; do
  mlflow run -e analysis1 . -P threshold=$t -P generated_data="file:///home/azureuser/mlruns/0/16ad0e8f04bc4d5dbbc667838d26e919/artifacts"
done
