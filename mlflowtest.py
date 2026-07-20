# save as src/mlflow_test.py, or paste into a python shell
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/mwaseem@uchicago.edu/athletes-mlops-a2")

with mlflow.start_run(run_name="connection-test"):
    mlflow.log_param("test_param", "hello_databricks")
    mlflow.log_metric("test_metric", 1.0)

print("MLflow tracking test completed.")