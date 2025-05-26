import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

'''
# Read in the datasets
df = pd.read_csv('Data/StoreSales/train.csv', index_col='date')
events = pd.read_csv('Data/StoreSales/holidays_events.csv', index_col='date')
oil = pd.read_csv('Data/StoreSales/oil.csv', index_col='date')
transactions = pd.read_csv('Data/StoreSales/transactions.csv', index_col='date')
stores = pd.read_csv('Data/StoreSales/stores.csv', index_col='store_nbr')

# Join the oil data. Fill in the missing price values. Forward fill first, then backward fill so no blanks
df_2 = df.join(oil, how='left', on='date')
df_2['dcoilwtico'] = df_2['dcoilwtico'].ffill()
df_2['dcoilwtico'] = df_2['dcoilwtico'].bfill()

# Add the store data
df_3 = df_2.join(stores, how='left', on='store_nbr')

# Join the events data. Based on the store location, set the day type. Fill in the missing type values as Work Days
def label_holiday(x):
    day_type = x['type_event']
    if (x['transferred'] == True):
        day_type = 'Working Day'
    if (x['locale'] == 'Regional'):
        if (x['locale_name'] != x['state']):
            day_type = 'Working Day'
    if (x['locale'] == 'Local'):
        if (x['locale_name'] != x['city']):
            day_type = 'Working Day'
    return day_type

df_4 = df_3.join(events, how='left', on='date', lsuffix='_store', rsuffix='_event')
df_4['type_event'] = df_4.apply(label_holiday, axis=1)
df_4['type_event'] = df_4['type_event'].fillna('Work Day')

# Add the transaction data
df_5 = df_4.merge(transactions, on=['date', 'store_nbr'], how='left')
# Add a product on offer flag
df_5['offers_flag'] = np.where(df_5['onpromotion'] > 0, 1, 0)
# Create lagged variables
df_5['1d_lag'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(1))
df_5['2d_lag'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(2))
df_5['3d_lag'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(3))
df_5['5dMA'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df_5['10dMA'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
df_5['20dMA'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
df_5['40dMA'] = df_5.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=40, min_periods=1).mean())

# Drop the columns we no longer need
df_6 = df_5.drop(columns=['id', 'locale', 'locale_name', 'transferred', 'description'])
# One hot encode the categorical values
df_7 = pd.get_dummies(df_6, columns=['family', 'city', 'state', 'type_store', 'type_event'], dtype=int)
df_7.to_csv('Final_Dataset.csv')
'''

df = pd.read_csv('Final_Dataset.csv', index_col='date')
# Take just the last year for the 54 stores
df_2 = df.tail(365*54)
train_size = int(len(df_2) * 0.8)
train, test = df_2[1:train_size], df_2[train_size:]

from xgboost import XGBRegressor
store_array = []
num_stores = 54
for x in range(num_stores + 1):
    store_array.append(XGBRegressor())

for x in range(1, num_stores + 1):
    new_train = train.loc[train['store_nbr'] == x]
    X_train = new_train.drop(columns=['sales'])
    y_train = new_train['sales']
    store_array[x].fit(X_train, y_train)

def predict_function(x):
    store_num = int(x['store_nbr'])
    prediction = store_array[store_num].predict([x])[0]
    return prediction

X_test = test.drop(columns=['sales'])
y_test = test['sales']
y_hat = X_test.apply(predict_function, axis=1)

# Convert any negative predictions to 0
y_hat_final = pd.Series(y_hat.apply(lambda x: 0 if x < 0 else x))

from sklearn.metrics import root_mean_squared_log_error
print('Root Mean square log error : ' + str(root_mean_squared_log_error(y_test, y_hat_final)))

test_df = pd.read_csv('Data/StoreSales/test.csv', index_col='date')


