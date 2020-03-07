import pandas as pd
import csv
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# get files from stdin
data_set_file = sys.argv[1]
test_set_file = sys.argv[2]

# read files
train = pd.read_csv(data_set_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)
test = pd.read_csv(test_set_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)

# get sentences
train_sentence = np.array(train[1])
test_sentence = np.array(test[1])
test_id = np.array(test[0])


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words]
    return filtered_words


def stemming_words(words):
    ps = PorterStemmer()
    stem_words = [ps.stem(w) for w in words]
    return stem_words


def get_legal_sentence(raw_sentence):
    # rules from spec
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    illegal_char_pattern = re.compile(r'[^#@_$%\s\w\d]')
    words_pattern = re.compile(r'[#@_$%\w\d]{2,}')
    final_sentence = []
    for sentence in raw_sentence:
        # replace url with space
        s_remove_url = re.sub(url_pattern, ' ', sentence)
        # remove illegal char
        s_remove_illegal_char = re.sub(illegal_char_pattern, '', s_remove_url)
        # find all words
        words = re.findall(words_pattern, s_remove_illegal_char)
        # remove stop words
        words_without_stopwords = remove_stopwords(words)
        # stemming
        words_after_stemming = stemming_words(words_without_stopwords)
        # connect words_after_stemming
        final_sentence.append(' '.join(w for w in words_after_stemming))
    return final_sentence


# get legal train sentences
legal_train_sentence = np.array(get_legal_sentence(train_sentence))
# get legal test sentences
legal_test_sentence = np.array(get_legal_sentence(test_sentence))


# Create bag of words, use all features
count = CountVectorizer(lowercase=False, token_pattern='[#@_$%\w\d]{2,}')
bag_of_words = count.fit_transform(legal_train_sentence)
# get a train set
X_train = bag_of_words.toarray()
# get this train set's label
sentiment = np.array(train[3])
y_train = sentiment

# use the bag of words to create a test set
X_test = count.transform(legal_test_sentence).toarray()

# train the DT model
clf = MultinomialNB()
model = clf.fit(X_train, y_train)

predicted_y = model.predict(X_test)

# open file and write data into it
output_file = open('output.txt', 'w')
for i in range(len(test_id)):
    print(test_id[i], predicted_y[i])
