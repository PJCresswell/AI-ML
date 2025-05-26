import pandas as pd
import numpy as np

'''
# Read in the training / validation data
dataset = pd.read_csv('Data/Titanic/train.csv')

# Data preparation stage
# Drop the passenger ID and name as not needed
dataset.drop(columns=['PassengerId', 'Name'], inplace=True)
# Convert the boolean columns to 1's and 0's
dataset['Transported'] = dataset['Transported'].astype(int)
dataset['CryoSleep'] = dataset['CryoSleep'].astype(float)
dataset['VIP'] = dataset['VIP'].astype(float)
# Fill in missing numeric data where makes sense : Age to an average, spend values to 0
mean_age = dataset['Age'].mean()
dataset.fillna({'Age': mean_age}, inplace=True)
dataset.fillna({'RoomService': 0, 'FoodCourt': 0, 'Spa': 0, 'VRDeck': 0, 'ShoppingMall': 0}, inplace=True)
# Fill in the missing planet category data
dataset.fillna({'HomePlanet': 'Missing', 'Destination': 'Missing'}, inplace=True)
# Create new total spend and binary spend features
dataset['TotalSpend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
dataset['BinarySpend'] = dataset['TotalSpend'].apply(lambda x: 1.0 if x > 0 else 0.0)
# Split out the cabin field into deck, room and side - and add as new features. Deck and Side are categorical, Room is numeric
new_df = dataset['Cabin'].str.split('/', expand=True)
new_df.columns = ['Deck', 'Room', 'Side']
new_df['Room'] = new_df['Room'].astype(np.float64)
combined = pd.concat([dataset, new_df], axis=1)
# Perform one-hpt encoding on the categorical columns. Drop the features representing missing values
encoded = pd.get_dummies(combined, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], dtype=float)
encoded.drop(columns=['Cabin', 'Destination_Missing', 'HomePlanet_Missing'], inplace=True)
# Normalise the numeric columns
col_to_normalise = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']
for col in col_to_normalise:
    encoded[col] = (encoded[col] - encoded[col].min()) / (encoded[col].max() - encoded[col].min())
# Finally, drop any rows with NAN results
encoded.dropna(inplace=True)

# Split the dataset into training and validation sets
# Stratify option ensures that the categories are represented evenly across both sets
from sklearn.model_selection import train_test_split
train, val = train_test_split(encoded, test_size=0.2, shuffle=True, random_state=42, stratify=encoded['Transported'])
# Creating X, y for train set
X_train = train.drop(columns='Transported')
y_train = train['Transported']
# Creating X, y for validation set
X_val = val.drop(columns='Transported')
y_val = val['Transported']
'''

import xgboost as xgb
from sklearn import metrics

'''
# Feature selection : Method 1
# Using sklearn : Recursive feature elimination with cross-validation
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
clf = xgb.XGBClassifier()
cv = StratifiedKFold(10)
rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring="accuracy", min_features_to_select=1, n_jobs=2)
rfecv.fit(X_train, y_train)
print(f"Optimal number of features: {rfecv.n_features_}")
print('Params ' + str(rfecv.get_feature_names_out()))

# Plot out how the accuracy improves then plateaus as you add features
import matplotlib.pyplot as plt
import pandas as pd
cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()

# Feature selection : Method 2
# Generate all possible combinations of given number of features and try the first 500
import random
from itertools import combinations
best_score, counter = 0, 0
best_features = []

features = X_train.columns.values
random.shuffle(features)
comb_list = combinations(features, 6)
for example in comb_list:
    cols = list(example)
    #print('Example ' + str(example))
    #X_trial_train = X_train[list(example)]
    #print('Train ' + str(X_trial_train.head()))
    model = xgb.XGBClassifier().fit(X_train[cols], y_train)
    #X_trial_test = X_val[list(example)]
    #print('Test ' + str(X_trial_test.head()))
    y_hat = model.predict(X_val[cols])
    accuracy = metrics.accuracy_score(y_hat, y_val)
    if accuracy > best_score:
        print('New best : ' + str(accuracy) + ' : ' + str(list(example)))
        best_score = accuracy
        best_features = list(example)
    counter = counter + 1
    if counter > 500: break

# Hyper parameter optimisation with logging in mlflow

from mlflow.models.signature import infer_signature
from hyperopt import (fmin, hp, tpe, Trials, STATUS_OK)
from hyperopt.pyll.base import scope

import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Setting search space for xgboost model
search_space = {
    'objective': 'reg:squarederror',
    'eval_metric': 'logloss',
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

def train_model(params):
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        # Train xgboost classifier
        model = xgb.XGBClassifier(**params).fit(X_train, y_train)
        # Predict values for training and validation data & score accuracy
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
        val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
        # Log model params and results
        mlflow.log_param('params', params)
        mlflow.log_metric('training_accuracy', train_accuracy)
        mlflow.log_metric('validation_accuracy', val_accuracy)
        # Log model signature, class, and name
        signature = infer_signature(X_train, y_val_pred)
        mlflow.xgboost.log_model(model, 'model', signature=signature)
        mlflow.set_tag('estimator_name', model.__class__.__name__)
        mlflow.set_tag('estimator_class', model.__class__)
        # Set the loss to be -1 * validation accuracy so fmin maximizes it
        return {'status': STATUS_OK, 'loss': -1 * val_accuracy}

# Run fmin within an MLflow run context so that each hyperparam config is logged as a child run of a parent run called "xgboost_models"
trials = Trials()
with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name='xgboost_models'):
    xgboost_best_params = fmin(
        fn=train_model,
        space=search_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=10)

'''

# Get the best model
import mlflow
import mlflow.xgboost
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
EXPERIMENT_ID = dict(mlflow.get_experiment_by_name('SpaceshipTitanicParams'))['experiment_id']
runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.validation_accuracy DESC'])
best_run = runs_df.iloc[0]
best_run_id = best_run['run_id']
best_artifact_uri = best_run['artifact_uri']
best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')
print('Best Run Name ' + str(best_run['tags.mlflow.runName']))

# Read in the data
dataset = pd.read_csv('Data/Titanic/test.csv')
test_ids = dataset['PassengerId']

# Data preparation stage
# Drop the passenger ID and name as not needed
dataset.drop(columns=['PassengerId', 'Name'], inplace=True)
# Convert the boolean columns to 1's and 0's
#dataset['Transported'] = dataset['Transported'].astype(int)
dataset['CryoSleep'] = dataset['CryoSleep'].astype(float)
dataset['VIP'] = dataset['VIP'].astype(float)
# Fill in missing numeric data where makes sense : Age to an average, spend values to 0
mean_age = dataset['Age'].mean()
dataset.fillna({'Age': mean_age}, inplace=True)
dataset.fillna({'RoomService': 0, 'FoodCourt': 0, 'Spa': 0, 'VRDeck': 0, 'ShoppingMall': 0}, inplace=True)
# Fill in the missing planet category data
dataset.fillna({'HomePlanet': 'Missing', 'Destination': 'Missing'}, inplace=True)
# Create new total spend and binary spend features
dataset['TotalSpend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
dataset['BinarySpend'] = dataset['TotalSpend'].apply(lambda x: 1.0 if x > 0 else 0.0)
# Split out the cabin field into deck, room and side - and add as new features. Deck and Side are categorical, Room is numeric
new_df = dataset['Cabin'].str.split('/', expand=True)
new_df.columns = ['Deck', 'Room', 'Side']
new_df['Room'] = new_df['Room'].astype(np.float64)
combined = pd.concat([dataset, new_df], axis=1)
# Perform one-hpt encoding on the categorical columns. Drop the features representing missing values
encoded = pd.get_dummies(combined, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], dtype=float)
encoded.drop(columns=['Cabin', 'Destination_Missing', 'HomePlanet_Missing'], inplace=True)
# Normalise the numeric columns
col_to_normalise = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend']
for col in col_to_normalise:
    encoded[col] = (encoded[col] - encoded[col].min()) / (encoded[col].max() - encoded[col].min())
# Finally, drop any rows with NAN results
encoded.dropna(inplace=True)

# Predict the results for the testing dataset
test_pred = best_model.predict(encoded)
final = pd.concat([test_ids, pd.Series(test_pred)], axis=1)
final.columns = ['PassengerID', 'Transported']
final['Transported'] = final['Transported'].astype(bool)
print(final.head())
final.to_csv('submission_1.csv', index=False)

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