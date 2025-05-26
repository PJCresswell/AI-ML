import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True

# Read in the core datasets and combine together
train_df = pd.read_csv('Data/StoreSales/train.csv', parse_dates=True, index_col='date')
test_df = pd.read_csv('Data/StoreSales/test.csv', parse_dates=True, index_col='date')
raw = pd.concat([train_df, test_df])
# Take the last 2 years - then we lose one year for the lagged transactions
df_1 = raw.loc[raw.index > '2015-08-15']

# Join the oil data. Fill in the missing price values. Forward fill first, then backward fill so no blanks
oil = pd.read_csv('Data/StoreSales/oil.csv', parse_dates=True, index_col='date')
df_2 = df_1.join(oil, how='left', on='date')
df_2['dcoilwtico'] = df_2['dcoilwtico'].ffill()
df_2['dcoilwtico'] = df_2['dcoilwtico'].bfill()

# Add the store data
stores = pd.read_csv('Data/StoreSales/stores.csv', index_col='store_nbr')
df_3 = df_2.join(stores, how='left', on='store_nbr')

# Add the transaction data from one year ago
def hist_trans(x):
    try:
        date_rows = transactions.loc[x['Hist_date']]
        trans_rows = date_rows.loc[date_rows['store_nbr'] == x['store_nbr']]
        return trans_rows['transactions'][0]
    except:
        return 0

transactions = pd.read_csv('Data/StoreSales/transactions.csv', parse_dates=True, index_col='date')
transactions['Hist_date'] = transactions.index - pd.DateOffset(years=1)
transactions['Hist_trans'] = transactions.apply(hist_trans, axis=1)
df_4 = df_3.merge(transactions, on=['date', 'store_nbr'], how='left')

# Join the events data. Based on the store location set the day type. Fill in the missing type values as Work Days
def label_holiday(x):
    day_type = x['type_event']
    if (x['transferred'] == True):
        day_type = 'Work Day'
    if (x['locale'] == 'Regional'):
        if (x['locale_name'] != x['state']):
            day_type = 'Work Day'
    if (x['locale'] == 'Local'):
        if (x['locale_name'] != x['city']):
            day_type = 'Work Day'
    return day_type

events = pd.read_csv('Data/StoreSales/holidays_events.csv', parse_dates=True, index_col='date')
df_5 = df_4.join(events, how='left', on='date', lsuffix='_store', rsuffix='_event')
df_5['type_event'] = df_5.apply(label_holiday, axis=1)
df_5['type_event'] = df_5['type_event'].fillna('Work Day')

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
df_6 = df_5.drop(columns=['id', 'locale', 'locale_name', 'transferred', 'description', 'transactions', 'Hist_date'])
# One hot encode the categorical values
df_7 = pd.get_dummies(df_6, columns=['family', 'city', 'state', 'type_store', 'type_event'], dtype=int)
df_7.to_csv('Final_Dataset.csv')

# df = pd.read_csv('Final_Dataset.csv', index_col='date')

train = df_7.loc[df_7.index < '2017-08-16']
test = df_7.loc[df_7.index > '2017-08-15']

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
test_df['sales'] = X_test.apply(predict_function, axis=1)
test_df.to_csv('Results.csv', columns=['id', 'sales'])

# Convert any negative predictions to 0
# y_hat_final = pd.Series(y_hat.apply(lambda x: 0 if x < 0 else x))



