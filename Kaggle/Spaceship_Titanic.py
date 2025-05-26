import pandas as pd
import numpy as np

# Read in the data
dataset = pd.read_csv('Data/Titanic/train.csv')

# Data pre-processing
dataset.drop(columns=['PassengerId', 'Name'], inplace=True)
dataset['Transported'] = dataset['Transported'].astype(int)
mean_age = dataset['Age'].mean()
dataset.fillna({'Age': mean_age}, inplace=True)
dataset.fillna({'RoomService': 0, 'FoodCourt': 0, 'Spa': 0, 'VRDeck': 0, 'ShoppingMall': 0}, inplace=True)
dataset['TotalSpend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
dataset['BinarySpend'] = dataset['TotalSpend'].apply(lambda x: 1.0 if x > 0 else 0.0)
dataset.fillna({'HomePlanet': 'Missing', 'Destination': 'Missing'}, inplace=True)

new_df = dataset['Cabin'].str.split('/', expand=True)
new_df.columns = ['Deck', 'Room', 'Side']
new_df['Room'] = new_df['Room'].astype(np.float64)
combined = pd.concat([dataset, new_df], axis=1)
combined['CryoSleep'] = combined['CryoSleep'].astype(float)
combined['VIP'] = combined['VIP'].astype(float)
encoded = pd.get_dummies(combined, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], dtype=float)
encoded.drop(columns=['Cabin', 'Destination_Missing', 'HomePlanet_Missing'], inplace=True)

# Normalise the numeric columns
col_to_normalise = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']
for col in col_to_normalise:
    encoded[col] = (encoded[col] - encoded[col].min()) / (encoded[col].max() - encoded[col].min())

encoded.dropna(inplace=True)

from sklearn.model_selection import train_test_split

# Splitting the dataset into training/validation and holdout sets
train_val, test = train_test_split(
    encoded,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# Creating X, y for training/validation set
X_train_val = train_val.drop(columns='Transported')
y_train_val = train_val['Transported']

# Creating X, y for test set
X_test = test.drop(columns='Transported')
y_test = test['Transported']

# Splitting training/testing set to create training set and validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    stratify=y_train_val,
    shuffle=True,
    random_state=42
)

from itertools import combinations
import xgboost as xgb
from sklearn import metrics
from mlflow.models.signature import infer_signature

from hyperopt import (
    fmin,
    hp,
    tpe,
    rand,
    SparkTrials,
    Trials,
    STATUS_OK
)
from hyperopt.pyll.base import scope

import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
#mlflow.set_experiment("SpaceshipTitanicParams")

# Setting search space for xgboost model
search_space = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
    'subsample': hp.uniform('subsample', .5, 1.0),
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
    'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
    'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
    'gamma': hp.loguniform('gamma', -10, 10),
    'use_label_encoder': False,
    'verbosity': 0,
    'random_state': 42
}

try:
    EXPERIMENT_ID = mlflow.create_experiment('SpaceshipTitanicParams')
except:
    EXPERIMENT_ID = dict(mlflow.get_experiment_by_name('SpaceshipTitanicParams'))['experiment_id']

# Querying mlflow api instead of using web UI. Sorting by validation aucroc and then getting top run for best run.
runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.validation_aucroc DESC'])
best_run = runs_df.iloc[0]

# print(best_run)

best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']
# Loading model from best run
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')

print('Best Run Name ' + str(best_run['tags.mlflow.runName']))

# Predicting and evaluating best model on holdout set
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = metrics.accuracy_score(y_test, y_test_pred)#.round(3)
test_precision = metrics.precision_score(y_test, y_test_pred)#.round(3)
test_recall = metrics.recall_score(y_test, y_test_pred)#.round(3)
test_f1 = metrics.f1_score(y_test, y_test_pred)#.round(3)
test_aucroc = metrics.roc_auc_score(y_test, y_test_pred_proba)#.round(3)

print(f'Testing Accuracy: {test_accuracy}')
print(f'Testing Precision: {test_precision}')
print(f'Testing Recall: {test_recall}')
print(f'Testing F1: {test_f1}')
print(f'Testing AUCROC: {test_aucroc}')

exit()

def train_model(params):
    """
    Creates a hyperopt training model function that sweeps through params in a nested run
    Args:
        params: hyperparameters selected from the search space
    Returns:
        hyperopt status and the loss metric value
    """
    # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
    # This sometimes doesn't log everything you may want so I usually log my own metrics and params just in case
    # mlflow.xgboost.autolog()

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        # Training xgboost classifier
        model = xgb.XGBClassifier(**params)
        model = model.fit(X_train, y_train)

        # Predicting values for training and validation data, and getting prediction probabilities
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        # Evaluating model metrics for training set predictions and validation set predictions
        # Creating training and validation metrics dictionaries to make logging in mlflow easier
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc']
        # Training evaluation metrics
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)#.round(3)
        train_precision = metrics.precision_score(y_train, y_train_pred)#.round(3)
        train_recall = metrics.recall_score(y_train, y_train_pred)#.round(3)
        train_f1 = metrics.f1_score(y_train, y_train_pred)#.round(3)
        train_aucroc = metrics.roc_auc_score(y_train, y_train_pred_proba)#round(3)
        training_metrics = {
            'Accuracy': train_accuracy,
            'Precision': train_precision,
            'Recall': train_recall,
            'F1': train_f1,
            'AUCROC': train_aucroc
        }
        training_metrics_values = list(training_metrics.values())

        # Validation evaluation metrics
        val_accuracy = metrics.accuracy_score(y_val, y_val_pred)#.round(3)
        val_precision = metrics.precision_score(y_val, y_val_pred)#.round(3)
        val_recall = metrics.recall_score(y_val, y_val_pred)#.round(3)
        val_f1 = metrics.f1_score(y_val, y_val_pred)#.round(3)
        val_aucroc = metrics.roc_auc_score(y_val, y_val_pred_proba)#.round(3)
        validation_metrics = {
            'Accuracy': val_accuracy,
            'Precision': val_precision,
            'Recall': val_recall,
            'F1': val_f1,
            'AUCROC': val_aucroc
        }
        validation_metrics_values = list(validation_metrics.values())

        mlflow.log_param('params', params)

        # Logging model signature, class, and name
        signature = infer_signature(X_train, y_val_pred)
        mlflow.xgboost.log_model(model, 'model', signature=signature)
        mlflow.set_tag('estimator_name', model.__class__.__name__)
        mlflow.set_tag('estimator_class', model.__class__)

        # Logging each metric
        for name, metric in list(zip(metric_names, training_metrics_values)):
            mlflow.log_metric(f'training_{name}', metric)
        for name, metric in list(zip(metric_names, validation_metrics_values)):
            mlflow.log_metric(f'validation_{name}', metric)

        # Set the loss to -1*validation auc roc so fmin maximizes the it
        return {'status': STATUS_OK, 'loss': -1 * validation_metrics['AUCROC']}

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep.
# A reasonable value for parallelism is the square root of max_evals.
# spark_trials = SparkTrials()
# Will need spark configured and installed to run. Add this to fmin function below like so:
# trials = spark_trials
trials = Trials()

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_models'):
    xgboost_best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=10
    )

'''
features = ['Destination_TRAPPIST-1e', 'Deck', 'TotalSpend', 'FoodCourt', 'Side', 'ShoppingMall']
X = final[features]
y = final['Transported']
X_norm = normalize(X, norm='l2')
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, train_size=0.8, random_state=42)
xgb = XGBClassifier().fit(X_train, y_train)
y_hat = xgb.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_hat)
print('Accuracy ' + str(accuracy))

# Generate our list of features
feature_vals = final.columns.values
features = np.delete(feature_vals, np.where(feature_vals == 'Transported'))
#print(features)
feature_num = len(features)
#print (feature_num)

import random
random.shuffle(features)
print(features)

best_score = 0
best_features = []
comb_list = combinations(features, 6)
for example in comb_list:
    X = final[list(example)]
    y = final['Transported']
    X_norm = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, train_size=0.8, random_state=42)
    xgb = XGBClassifier().fit(X_train, y_train)
    y_hat = xgb.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_hat)
    if accuracy > best_score:
        print('New best : ' + str(accuracy) + ' : ' + str(list(example)))
        best_score = accuracy
        best_features = list(example)

import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("SpaceshipTitanic")
comb_list = combinations(features, 3)
for example in comb_list:
    print('Running for features : ' + str(list(example)))
    with mlflow.start_run():
        X = final[list(example)]
        y = final['Transported']
        X_norm = normalize(X, norm='l2')
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, train_size=0.8, random_state=42)
        xgb = XGBClassifier().fit(X_train, y_train)
        y_hat = xgb.predict(X_test)

        mlflow.log_param('Features', list(example))
        mlflow.log_metric('Accuracy', metrics.accuracy_score(y_test, y_hat))
        mlflow.log_metric('Precision', metrics.precision_score(y_test, y_hat, average='binary'))
        mlflow.log_metric('Recall', metrics.recall_score(y_test, y_hat, average='binary'))
        mlflow.end_run()

        #print('F1 : ' + str(metrics.f1_score(y_test, y_hat, average=None)))
        #print('Confusion matrix : \n' + str(metrics.confusion_matrix(y_test, y_hat)))

from xgboost import plot_importance
plot_importance(bst)

from sklearn.neural_network import MLPClassifier
ann = MLPClassifier(hidden_layer_sizes=[16, 8], max_iter=800)
ann.fit(X_train, y_train)
y_hat = ann.predict(X_test)

from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_hat = gnb_model.predict(X_test)

from sklearn.ensemble import AdaBoostClassifier
abc_model = AdaBoostClassifier(algorithm="SAMME", random_state=42)
abc_model.fit(X_train, y_train)
y_hat = abc_model.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(max_depth=20, n_estimators=30, random_state=42)
rfc_model.fit(X_train, y_train)
y_hat = rfc_model.predict(X_test)

from sklearn.svm import SVC
svc_model = SVC(gamma=3, kernel='rbf', C=1, random_state=42, class_weight='balanced').fit(X_train, y_train)
y_hat = svc_model.predict(X_test)
score_model(y_test, y_hat)
'''