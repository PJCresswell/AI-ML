# Logging an experiment
with mlflow.start_run():
    mlflow.log_param("input1", input1)
    mlflow.log_param("input2", input2)
    # Perform operations here like model training.
    mlflow.log_metric("rmse", rmse)

# Log a model - to then reister in the UI when reviewed the results
with mlflow.start_run():
    # code to train model goes here

    # log the model itself (and the environment it needs to be used)
    unique_model_name = "my_model-" + str(time.time())
    mlflow.spark.log_model(spark_model = model,
                           artifact_path=unique_model_name,
                           conda_env=mlflow.spark.get_default_conda_env())

# Rehister a model without reviewing the results
with mlflow.start_run():
    # code to train model goes here

    # log the model itself (and the environment it needs to be used)
    unique_model_name = "my_model-" + str(time.time())
    mlflow.spark.log_model(spark_model=model,
                           artifact_path=unique_model_name
                           conda_env=mlflow.spark.get_default_conda_env(),
                           registered_model_name="my_model")