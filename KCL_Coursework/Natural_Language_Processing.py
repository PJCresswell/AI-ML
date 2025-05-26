import pandas as pd
import numpy as np
import nltk
import math
import textblob
import requests
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nltk.download('punkt')

# Load the data
raw = pd.read_csv('datasets/SMSSpamCollection.csv', encoding='ISO-8859-1', header=None)
raw.columns = ['type', 'text']

# Create the new features for this data
# After list of experimentation, this is best done as a list
# Tried to be smart here and only iterate over the data once
raw_list = raw.values.tolist()
punctuation = ['.', ',', '!', ':', ';']
capitals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
spam_words = ['winner', 'WINNER', 'buy', 'BUY', 'free', 'FREE', 'vouchers', 'VOUCHERS', 'bonus', 'BONUS']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ham_count = 0
spam_count = 0
for i in range(0, len(raw_list)):
    # Count how many ham and spam
    if (raw_list[i][0] == 'ham'):
        ham_count += 1
    else:
        spam_count += 1
    # Counts of punctuation characters
    count = 0
    for j in punctuation:
        count += raw_list[i][1].count(j)
    raw_list[i].append(count)
    # Counts of capital characters
    count = 0
    for j in capitals:
        count += raw_list[i][1].count(j)
    raw_list[i].append(count)
    # Counts of spam words
    count = 0
    for j in spam_words:
        count += raw_list[i][1].count(j)
    raw_list[i].append(count)
    # Counts of digits
    count = 0
    for j in digits:
        count += raw_list[i][1].count(j)
    raw_list[i].append(count)
    # Count the number of words
    token_set = nltk.word_tokenize(raw_list[i][1])
    raw_list[i].append(len(token_set))
# print(raw_list[0])

print('Length of ham ' + str(ham_count) + ' Length of spam ' + str(spam_count))

# Next, create the balanced data set
target = min(spam_count, ham_count)
combined_list = []
spams = 0
hams = 0
# Generate a list of random numbers to use an indices across the full dataset
rnd_num_list = np.random.permutation(len(raw_list))
count = 0
while (spams < target) or (hams < target):
    if (raw_list[rnd_num_list[count]][0] == 'ham'):
        if hams < target:
            combined_list.append(raw_list[rnd_num_list[count]])
            hams += 1
    else:
        if spams < target:
            combined_list.append(raw_list[rnd_num_list[count]])
            spams += 1
    count += 1

# Now we make into a pandas dataframe, split into test and train and pass into the model
df = pd.DataFrame(combined_list)
df.columns = ['type', 'text', 'punct', 'caps', 'obv_wds', 'digits', 'words']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['punct', 'caps', 'obv_wds', 'digits', 'words']], df['type'],
                                                    test_size=0.2)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred = mnb.predict(X_test)

print('Training score = ' + str(mnb.score(X_train, y_train)))
print('Test score = ' + str(mnb.score(X_test, y_test)))
print('Accuracy score = ' + str(metrics.accuracy_score(y_test, pred)))
cm = metrics.confusion_matrix(y_test, pred)
print('Confusion matrix ' + str(cm))
precision = metrics.precision_score(y_test, pred, average=None)
print('Precision ' + str(precision))
recall = metrics.recall_score(y_test, pred, average=None)
print('Recall ' + str(recall))
f1 = metrics.f1_score(y_test, pred, average=None)
print('f1 score ' + str(f1))

####################
# Clustering text
####################

# source the stopwords
stopwords = requests.get(
    'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').content.decode(
    'utf-8').split('\n')


def return_most_freq(verse, count):
    # make all lower case
    verse_lower = verse.lower()
    # remove stop words
    final_text = ' '
    for word in verse_lower.words:
        if word not in stopwords:
            final_text = final_text + word + ' '
    new_blob = textblob.TextBlob(final_text)
    dict = new_blob.word_counts
    # sort the word counts to get the top counts
    temp = dict.items()
    new_list = list(temp)
    new_list.sort(key=lambda a: a[1], reverse=True)
    top_counts = new_list[:count]
    return top_counts


def load_file(filename):
    # open data file containing verse
    f = open(file=filename, mode='r')
    raw_verse = f.read()
    f.close()
    # decode contents of text file and initialise a TextBlob object
    verse = textblob.TextBlob(raw_verse)
    return verse


three = load_file('datasets/a-3kittens.txt')
three_counts = return_most_freq(three, 10)
print(three_counts)
tom = load_file('datasets/bp-tom-kitten.txt')
tom_counts = return_most_freq(tom, 10)
print(tom_counts)
owl = load_file('datasets/el-owl-cat.txt')
owl_counts = return_most_freq(owl, 10)
print(owl_counts)
diddle = load_file('datasets/rc-cat-fiddle.txt')
diddle_counts = return_most_freq(diddle, 10)
print(diddle_counts)
cat = load_file('datasets/rk-cat.txt')
cat_counts = return_most_freq(cat, 10)
print(cat_counts)

keys = []
for i in range(0, 10):
    if (three_counts[i][0] not in keys):
        keys.append(three_counts[i][0])
    if (tom_counts[i][0] not in keys):
        keys.append(tom_counts[i][0])
    if (owl_counts[i][0] not in keys):
        keys.append(owl_counts[i][0])
    if (diddle_counts[i][0] not in keys):
        keys.append(diddle_counts[i][0])
    if (cat_counts[i][0] not in keys):
        keys.append(cat_counts[i][0])
print(keys)

bool_term = pd.DataFrame(0, columns=keys, index=['three', 'tom', 'owl', 'diddle', 'cat'])
freq_term = pd.DataFrame(0, columns=keys, index=['three', 'tom', 'owl', 'diddle', 'cat'])

for i in range(0, 10):
    bool_term.loc['three', three_counts[i][0]] = 1
    freq_term.loc['three', three_counts[i][0]] = three_counts[i][1]
    bool_term.loc['tom', tom_counts[i][0]] = 1
    freq_term.loc['tom', tom_counts[i][0]] = tom_counts[i][1]
    bool_term.loc['owl', owl_counts[i][0]] = 1
    freq_term.loc['owl', owl_counts[i][0]] = owl_counts[i][1]
    bool_term.loc['diddle', diddle_counts[i][0]] = 1
    freq_term.loc['diddle', diddle_counts[i][0]] = diddle_counts[i][1]
    bool_term.loc['cat', cat_counts[i][0]] = 1
    freq_term.loc['cat', cat_counts[i][0]] = cat_counts[i][1]


# print(bool_term)
# print(freq_term)

def cosine_dist(vector_u, vector_v):
    numerator = sum(vector_u * vector_v)
    u_sq = sum(vector_u ** 2)
    v_sq = sum(vector_v ** 2)
    root_u_sq = math.sqrt(u_sq)
    root_v_sq = math.sqrt(v_sq)
    denominator = root_u_sq * root_v_sq
    result = numerator / denominator
    return result


def euclidean_dist(vector_u, vector_v):
    sum_diff = sum((vector_v - vector_u) ** 2)
    result = math.sqrt(sum_diff)
    return result


for i in range(0, 5):
    for j in range(0, 5):
        if i != j:
            vec_1 = pd.Series(freq_term.iloc[i])
            vec_2 = pd.Series(freq_term.iloc[j])
            cos_dt = cosine_dist(vec_1, vec_2)
            # The larger the number the better - the more similar the documents are
            print('Cosine similarity between ' + str(i) + ' and ' + str(j) + ' is ' + str(cos_dt))

for i in range(0, 5):
    for j in range(0, 5):
        if i != j:
            vec_1 = pd.Series(freq_term.iloc[i])
            vec_2 = pd.Series(freq_term.iloc[j])
            euc_dt = euclidean_dist(vec_1, vec_2)
            # The smaller the number the better - the more similar the documents are
            print('Euclidean distance between ' + str(i) + ' and ' + str(j) + ' is ' + str(euc_dt))
