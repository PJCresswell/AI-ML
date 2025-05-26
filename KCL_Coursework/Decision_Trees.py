import pandas as pd

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.

def read_csv_1(data_file):
	raw = pd.read_csv(data_file, encoding='ISO-8859-1')
	raw2 = raw.drop(columns=['fnlwgt'])
	return raw2

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	rows = len(df)
	print(rows)
	return rows

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	attributes = df.keys()
	# print(attributes)
	return attributes

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	count_empty = 0
	attributes = column_names(df)
	for column in attributes:
		empty = pd.isnull(raw[column])
		empty_num = len(raw[empty])
		count_empty += empty_num
	# print(count_empty)
	return count_empty

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	empty_list = []
	attributes = column_names(df)
	for column in attributes:
		empty = pd.isnull(raw[column])
		empty_num = len(raw[empty])
		if empty_num > 0: empty_list.append(column)
	# print(empty_list)
	return empty_list

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	grouping = raw['education'].value_counts()
	total = len(df)
	count = grouping['Bachelors'] + grouping['Masters']
	result = count / total * 100
	result_1dp = round(result, 1)
	return result_1dp

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	cleaned = df.dropna()
	return cleaned

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	remove_target = df.drop(columns=['class'])
	# print(remove_target)
	one_hot = pd.get_dummies(remove_target)
	# print(one_hot)
	# one_hot.to_csv('testfile3.csv')
	return one_hot

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	from sklearn.preprocessing import LabelEncoder
	enc = LabelEncoder()
	labels = column_names(df)
	enc.fit(labels)
	result = enc.transform(labels)
	#print(result)
	return result

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	from sklearn import tree
	dtc = tree.DecisionTreeClassifier(random_state=0)
	dtc.fit(X_train, y_train)
	y_hat = dtc.predict(X_train)
	return y_hat

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred_new, y_true):
	count = 0
	y_true_new = y_true.to_numpy()
	for i in range(0, len(y_pred_new)):
		if (y_pred_new[i] == y_true_new[i]):
			count += 1
	score = count / len(y_pred_new)
	accuracy = score * 100
	error_rate = (1- score) * 100
	# print('Number of correct predictions = %d out of %d = %f%%' % (count, len(y_pred_new), accuracy))
	return error_rate

raw = read_csv_1('datasets/adult.csv')
rows = num_rows(raw)
print('Num rows ' + str(rows))
attributes = column_names(raw)
print('Attributes ' + str(attributes))
missing = missing_values(raw)
print('Num of missing values ' + str(missing))
empty_col = columns_with_missing_values(raw)
print('Cols with missing values ' + str(empty_col))
percentage = bachelors_masters_percentage(raw)
print('Percentage Bachelors or Masters ' + str(percentage))
remove_missing = data_frame_without_missing_values(raw)
print('DF where rows with missing values dropped ' + str(remove_missing))
one_hot = one_hot_encoding(raw)
print('One hot encoding ' + str(one_hot))
labels = label_encoding(raw)
print('Label encoding ' + str(labels))
raw2 = data_frame_without_missing_values(raw)
X = one_hot_encoding(raw2)
Y = raw2['class']
predict = dt_predict(X, Y)
error_rate = dt_error_rate(predict, Y)
print('Decision tree error rate ' + str(error_rate))