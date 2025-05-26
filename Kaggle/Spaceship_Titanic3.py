import pandas as pd
import numpy as np

def process_data(dataset):
    # Fill in missing numeric data where makes sense : Age to an average as normally distributed
    mean_age = dataset['Age'].mean()
    dataset.fillna({'Age': mean_age}, inplace=True)
    dataset.fillna({'HomePlanet': 'Missing', 'Destination': 'Missing', 'Cabin': 'M/0/M'}, inplace=True)
    # Create new total spend and binary spend anything features
    dataset['TotalSpend'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']
    dataset['BinarySpend'] = dataset['TotalSpend'].apply(lambda x: 1 if x > 0 else 0)
    # Convert the boolean columns to 1's and 0's
    dataset['CryoSleep'] = dataset['CryoSleep'].apply(lambda x: 2 if x == True else 1 if x == False else 0)
    dataset['VIP'] = dataset['VIP'].apply(lambda x: 2 if x == True else 1 if x == False else 0)
    # Split out the cabin field into deck/room/side and add as new features. Deck and Side are categorical, Room is numeric
    new_df = dataset['Cabin'].str.split('/', expand=True)
    new_df.columns = ['Deck', 'Room', 'Side']
    new_df['Room'] = new_df['Room'].astype(np.int32)
    combined = pd.concat([dataset, new_df], axis=1)
    # Perform one-hot encoding on the categorical columns. Drop the features representing missing values
    final = pd.get_dummies(combined, columns=['HomePlanet', 'Destination', 'Deck', 'Side'], dtype=int)
    return final

# Read in and process the training and testing data
raw_train = pd.read_csv('Data/Titanic/train.csv')
processed_train = process_data(raw_train)
processed_train['Transported'] = processed_train['Transported'].astype(int)
raw_test = pd.read_csv('Data/Titanic/test.csv')
processed_test = process_data(raw_test)

# Split the training dataset into training and validation sets
# Stratify option ensures that the categories are represented evenly across both sets
from sklearn.model_selection import train_test_split
train, val = train_test_split(processed_train, test_size=0.2, shuffle=True, random_state=42, stratify=processed_train['Transported'])
# Creating X, y for train set
X_train = train.drop(columns='Transported')
y_train = train['Transported']
# Creating X, y for validation set
X_val = val.drop(columns='Transported')
y_val = val['Transported']

import xgboost as xgb
from sklearn import metrics

print(X_train.columns)

print(X_train.info())

features = ['CryoSleep', 'Age', 'VIP', 'RoomService',
            'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpend',
            'BinarySpend', 'Room', 'HomePlanet_Earth', 'HomePlanet_Europa',
            'HomePlanet_Mars', 'HomePlanet_Missing', 'Destination_55 Cancri e',
            'Destination_Missing', 'Destination_PSO J318.5-22',
            'Destination_TRAPPIST-1e', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D',
            'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Deck_M', 'Side_P', 'Side_S', 'Side_M']
model = xgb.XGBClassifier(gamma=0.2, max_depth=12)
model.fit(X_train[features], y_train)
y_train_pred = model.predict(X_train[features])
y_val_pred = model.predict(X_val[features])
print('Training accuracy : ' + str(metrics.accuracy_score(y_train, y_train_pred)))
print('Validation accuracy : ' + str(metrics.accuracy_score(y_val, y_val_pred)))

# Predict the results for the testing dataset
test_pred = model.predict(processed_test[features])
final = pd.concat([processed_test['PassengerId'], pd.Series(test_pred)], axis=1)
final.columns = ['PassengerId', 'Transported']
final['Transported'] = final['Transported'].astype(bool)
print(final.head())
final.to_csv('submission_2csv', index=False)

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