import random
import more_itertools as it
import pandas as pd
import nltk
import requests
from sklearn.naive_bayes import MultinomialNB

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	# raw = pd.read_csv('coronavirus_tweets.csv', encoding='ISO-8859-1', date_parser='')
	raw = pd.read_csv(data_file, encoding='latin-1', date_parser='')
	return raw

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	unique = pd.unique(df['Sentiment'])
	return unique

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	list = get_sentiments(df)
	freq = df['Sentiment'].value_counts()
	# print(freq)
	second = freq.index[1]

	# print(second)
	return second

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	EP = df.loc[raw['Sentiment'] == 'Extremely Positive', ['TweetAt']]
	# print(EP)
	counts = EP['TweetAt'].value_counts()
	# print(counts)
	return (counts.index[0])

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	to_remove = '!"#£$%&()*+,-./:;<=>?@[\]^_`{|}~0123456789'
	new_dict = dict()
	for char in to_remove: new_dict[char] = ' '
	#print(new_dict)
	table = str.maketrans(new_dict)
	df['OriginalTweet'] = df['OriginalTweet'].str.translate(table)
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	# I'm experimenting here with regular expressions - something not done before but learnt for this exercise
	# \s+ indicates one or more spaces. Is being replaced by a single space
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True)
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()
	#print(df['OriginalTweet'])
	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	# I also got lots of experience in this coursework with lambda expressioms
	# My first code attempt was often super slow and needed re-work in this way to perform
	tdf['Num_words'] = tdf['OriginalTweet'].apply(lambda i: len(i))
	total = tdf['Num_words'].sum()
	# print('All words ' + str(total))
	return total

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	distinct_words = set(word for tweet in tdf['OriginalTweet'] for word in tweet)
	total = len(distinct_words)
	# print('Distinct words ' + str(total))
	return total

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	all_words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
	series = pd.Series(all_words).value_counts()
	result = series.head(k).keys().tolist()
	# print(result)
	return result

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	stopwordlist = requests.get(
		'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').content.decode(
		'utf-8').split('\n')
	def process_tweet(word):
		# We do both operations here at once
		word = [w for w in word if (w not in set(stopwordlist)) and (len(w) > 2)]
		return word
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(process_tweet)
	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	# I got some experience in nltk in Practical 7 and so used it here as well
	from nltk.stem import PorterStemmer
	ps = PorterStemmer()
	def stem_word(word):
		word = [ps.stem(w) for w in word]
		return word
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(stem_word)
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):

	# First I want to create a balanced training data set
	# We saw at the start of the coursework that the original data is not at all balanced between sentiments
	# Our target for each group is the minimum number of instances within any group
	target = min(df['Sentiment'].value_counts())
	combined_list = []
	Positive = 0
	Negative = 0
	Neutral = 0
	Extremely_Positive = 0
	Extremely_Negative = 0
	# To pick the examples at random, I generate a list of random numbers to use as indices
	rnd_num_list = it.random_permutation(range(len(df['Sentiment'])))
	count = 0
	# We go through the list of random index numbers one by one - adding to our new list
	# We continue until all catagories are complete
	while (Positive < target) or (Negative < target) or (Neutral < target) or (Extremely_Positive < target) or (
			Extremely_Negative < target):
		if (df['Sentiment'].iloc[rnd_num_list[count]] == 'Positive'):
			if (Positive < target):
				combined_list.append(
					[df['OriginalTweet'].iloc[rnd_num_list[count]], df['Sentiment'].iloc[rnd_num_list[count]]])
				Positive += 1
		if (df['Sentiment'].iloc[rnd_num_list[count]] == 'Negative'):
			if (Negative < target):
				combined_list.append(
					[df['OriginalTweet'].iloc[rnd_num_list[count]], df['Sentiment'].iloc[rnd_num_list[count]]])
				Negative += 1
		if (df['Sentiment'].iloc[rnd_num_list[count]] == 'Neutral'):
			if (Neutral < target):
				combined_list.append(
					[df['OriginalTweet'].iloc[rnd_num_list[count]], df['Sentiment'].iloc[rnd_num_list[count]]])
				Neutral += 1
		if (df['Sentiment'].iloc[rnd_num_list[count]] == 'Extremely Positive'):
			if (Extremely_Positive < target):
				combined_list.append(
					[df['OriginalTweet'].iloc[rnd_num_list[count]], df['Sentiment'].iloc[rnd_num_list[count]]])
				Extremely_Positive += 1
		if (df['Sentiment'].iloc[rnd_num_list[count]] == 'Extremely Negative'):
			if (Extremely_Negative < target):
				combined_list.append(
					[df['OriginalTweet'].iloc[rnd_num_list[count]], df['Sentiment'].iloc[rnd_num_list[count]]])
				Extremely_Negative += 1
		count += 1
	final_list = pd.DataFrame(combined_list, columns=['OriginalTweet', 'Sentiment'])

	# Next, run the pre-processing steps on our balanced training dataset
	new_frame1 = lower_case(final_list)
	new_frame2 = remove_non_alphabetic_chars(new_frame1)
	new_frame3 = remove_multiple_consecutive_whitespaces(new_frame2)
	new_frame4 = tokenize(new_frame3)
	new_frame5 = remove_stop_words(new_frame4)
	new_frame6 = stemming(new_frame5)

	def join_up_words(x):
		return ' '. join(x)
	# Before I can send into the CountVectorizer, I need to recombine the tokens into a single string
	new_frame6['Final_String'] = new_frame6['OriginalTweet'].apply(join_up_words)

	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(max_features=1000, min_df=100)

	X = vectorizer.fit_transform(new_frame6['Final_String'])

	mnb = MultinomialNB()
	mnb.fit(X, new_frame6['Sentiment'])
	pred = mnb.predict(X)

	# I call the accuracy function here as I also need the balanced sentiment data from the training set
	accuracy = mnb_accuracy(pred, new_frame6['Sentiment'])
	print('Accuracy ' + str(accuracy))
	return pred

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	from sklearn import metrics
	accuracy = round(metrics.accuracy_score(y_true, y_pred), 3)
	return accuracy

raw = read_csv_3('datasets/coronavirus_tweets.csv')
sentiments = get_sentiments(raw)
print('Sentiments ' + str(sentiments))
second = second_most_popular_sentiment(raw)
print('Second most popular sentiment ' + str(second))
date = date_most_popular_tweets(raw)
print('Date of most popular tweet ' + str(date))
df1 = lower_case(raw)
df2 = remove_non_alphabetic_chars(df1)
df3 = remove_multiple_consecutive_whitespaces(df2)
print('Result of the preprocessing ' + str(df3))
df4 = tokenize(df3)
print('Result of the tokenising ' + str(df4))
processed = remove_stop_words(df4)
print('No stop words ' + str(processed))
stemmed = stemming(df4)
print('After stemming ' + str(stemmed))
count = count_words_with_repetitions(df4)
print('Word count with repetition ' + str(count))
distinct = count_words_without_repetitions(df4)
print('Word count without repetition ' + str(distinct))
list = frequent_words(df4, 10)
print('10 most frequent words ' + str(list))

predict = mnb_predict(raw)
