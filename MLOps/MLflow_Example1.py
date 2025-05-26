import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")
# Load data
X, y = fetch_california_housing(return_X_y = True)
# X, y = datasets.load_boston(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

mlflow.set_experiment("My First Experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run():
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(score)
    mlflow.log_metric("r2_score", score)

    # Log model
    mlflow.sklearn.log_model(model, "linear_regression_model")