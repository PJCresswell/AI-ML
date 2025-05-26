import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	raw = pd.read_csv(data_file, encoding='ISO-8859-1')
	# print(raw.keys())
	new_data = raw.drop(columns=['Channel', 'Region'])
	# print(new_data.keys())
	return new_data

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	new_list = []
	for attrib in df.keys():
		mean = df[attrib].mean()
		std = df[attrib].std()
		min = df[attrib].min()
		max = df[attrib].max()
		new_list.append([round(mean), round(std), min, max])
	new_df = pd.DataFrame(new_list, index=df.keys(), columns=['mean', 'std', 'min', 'max'])
	# print(new_df)
	return new_df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	new_df = summary_statistics(df)
	df['Std_Fresh'] = df['Fresh'].apply(
		lambda x: (x - new_df.loc['Fresh']['mean']) / new_df.loc['Fresh']['std'])
	df['Std_Milk'] = df['Milk'].apply(
		lambda x: (x - new_df.loc['Milk']['mean']) / new_df.loc['Milk']['std'])
	df['Std_Grocery'] = df['Grocery'].apply(
		lambda x: (x - new_df.loc['Grocery']['mean']) / new_df.loc['Grocery']['std'])
	df['Std_Frozen'] = df['Frozen'].apply(
		lambda x: (x - new_df.loc['Frozen']['mean']) / new_df.loc['Frozen']['std'])
	df['Std_Detergents_Paper'] = df['Detergents_Paper'].apply(
		lambda x: (x - new_df.loc['Detergents_Paper']['mean']) / new_df.loc['Detergents_Paper']['std'])
	df['Std_Delicassen'] = df['Delicassen'].apply(
		lambda x: (x - new_df.loc['Delicassen']['mean']) / new_df.loc['Delicassen']['std'])
	new_frame = df.drop(columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
	final_frame = new_frame.rename(columns={'Std_Fresh' : 'Fresh', 'Std_Milk' : 'Milk', 'Std_Grocery' : 'Grocery', 'Std_Frozen' : 'Frozen', 'Std_Detergents_Paper' : 'Detergents_Paper', 'Std_Delicassen' : 'Delicassen'})
	# print(final_frame)
	return (final_frame)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=k, init='random', n_init='auto').fit(df)
	predict = kmeans.predict(df)
	return predict

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=k, init='k-means++').fit(df)
	predict = kmeans.predict(df)
	return predict

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	from sklearn.cluster import AgglomerativeClustering
	agg = AgglomerativeClustering(n_clusters=k).fit(df)
	predict = agg.fit_predict(df)
	return predict

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	from sklearn.metrics import silhouette_score
	score = silhouette_score(X, y)
	return score

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	std_df = standardize(df)
	results = []
	for i in (3, 5, 10):
		for j in range(0, 10):
			predict = kmeans(df, i)
			score = clustering_score(df, predict)
			results.append(['Kmeans', 'Original', i, score, predict])
			# print('Kmeans : Clusters ' + str(i) + ' score of ' + str(round(score, 3)))
	for i in (3, 5, 10):
		for j in range(0, 10):
			predict = kmeans(std_df, i)
			score = clustering_score(df, predict)
			results.append(['Kmeans', 'Standardized', i, score, predict])
			# print('Kmeans : Clusters ' + str(i) + ' score of ' + str(round(score, 3)))
	for i in (3, 5, 10):
		predict = agglomerative(df, i)
		score = clustering_score(df, predict)
		results.append(['Agglomerative', 'Original', i, score, predict])
		# print('Agglomerative : Clusters ' + str(i) + ' score of ' + str(round(score, 3)))
	for i in (3, 5, 10):
		predict = agglomerative(std_df, i)
		score = clustering_score(std_df, predict)
		results.append(['Agglomerative', 'Standardized', i, score, predict])
		# print('Agglomerative : Clusters ' + str(i) + ' score of ' + str(round(score, 3)))
	pd_results = pd.DataFrame(results, columns=['Algorithm', 'data', 'k', 'Silhouette Score', 'Results'])
	# print(pd_results)
	return pd_results

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	max_score = rdf['Silhouette Score'].max()
	return max_score

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	std_df = standardize(df)
	predict = kmeans(std_df, 3)
	keys = std_df.keys()
	plt.figure(figsize=(12, 6))
	plotcount = 0
	for i in range(0, 6):
		for j in range(i + 1, 6):
			plotcount += 1
			axs = plt.subplot(3, 5, plotcount)
			x = df[keys[i]].to_numpy()
			y = df[keys[j]].to_numpy()
			plt.scatter(x, y, c=predict)
			axs.set_xticks([])
			axs.set_yticks([])
			axs.set_xlabel(keys[i], fontsize=8)
			axs.set_ylabel(keys[j], fontsize=8)
	plt.savefig('Scatter_Plot_Results.pdf')
	plt.show()

raw = read_csv_2('datasets/wholesale_customers.csv')
summary_stats = summary_statistics(raw)
print('Summary stats ' + str(summary_stats))
std_data = standardize(raw)
print('Standardised data ' + str(std_data))
results = cluster_evaluation(raw)
print('Cluster evaluation ' + str(results))
best_score = best_clustering_score(results)
print('Best score ' + str(best_score))
scatter_plots(raw)