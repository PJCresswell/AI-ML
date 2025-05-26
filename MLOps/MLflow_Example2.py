import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error (actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

data = pd.read_csv('datasets/wine-quality.csv', encoding='ISO-8859-1')

# Split the data into training and test sets (0.75, 0.25) split
train, test = train_test_split(data)

# The predicted column is " quality " which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis =1)
test_x = test.drop(["quality"], axis =1)
train_y = train [["quality"]]
test_y = test[["quality"]]

logged_model = 'runs:/6b5363fb354b4cb89c63cd7ba9c01b00/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model.predict(pd.DataFrame(test_x))

'''
alpha = 0.6
l1_ratio = 0.2

mlflow.set_experiment("My 2nd Experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

dataset = mlflow.data.from_pandas(train, targets='quality')

with mlflow.start_run():
    lr = ElasticNet(alpha =alpha, l1_ratio = l1_ratio, random_state =42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print (f"Elasticnet model ( alpha ={ alpha :f}, l1_ratio ={ l1_ratio :f}):")
    print (f"RMSE : {rmse}")
    print (f"MAE: {mae}")
    print (f"R2: {r2}")

    mlflow.log_input(dataset, context="training")

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lr, "model")
'''