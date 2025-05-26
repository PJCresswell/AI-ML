# Example Objective function for logistic regression

def objective(params):
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from hyperopt import STATUS_OK

    data_df = get_training_data() # This is just an example!
    splits = data_df.randomSplit([0.7, 0.3])
    training_df = splits[0]
    validation_df = splits[1]

    # Train a model using the provided hyperparameter values
    lr = LogisticRegression(labelCol="label", featuresCol="features",
                            maxIter=params['Iterations'],
                            regParam=params['Regularization'])
    model = lr.fit(training_df)

    # Evaluate the model
    predictions = model.transform(validation_df)
    eval = MulticlassClassificationEvaluator(labelCol="label",
                                             predictionCol="prediction",
                                             metricName="accuracy")
    accuracy = eval.evaluate(predictions)
    
    # Hyperopt *minimizes* the function, so return *negative* accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# Defining the search space

from hyperopt import hp

search_space = {
    'Iterations': hp.randint('Iterations', 10),
    'Regularization': hp.uniform('Regularization', 0.0, 1.0)
}

# Specify the seatch algorithm

from hyperopt import tpe

algo = tpe.suggest

# Run the fmin function

from hyperopt import fmin

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=100)

print("Best param values: ", argmin)

# Using the Trials class

from hyperopt import Trials

# Create a Trials object to track each run
trial_runs = Trials()

argmin = fmin(
  fn=objective,
  space=search_space,
  algo=algo,
  max_evals=100,
  trials=trial_runs)

print("Best param values: ", argmin)

# Get details from each trial run
print ("trials:")
for trial in trial_runs.trials:
    print ("\n", trial)

# Using the Spark Trials class

from hyperopt import SparkTrials

  spark_trials = SparkTrials()
  with mlflow.start_run():
    argmin = fmin(
      fn=objective,
      space=search_space,
      algo=algo,
      max_evals=100,
      trials=spark_trials)
  
  print("Best param values: ", argmin)