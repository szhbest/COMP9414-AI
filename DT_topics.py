import pandas as pd
import csv
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import sys

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
        # connect words
        final_sentence.append(' '.join(w for w in words))
    return final_sentence


# get legal train sentences and test sentences
legal_train_sentence = np.array(get_legal_sentence(train_sentence))
legal_test_sentence = np.array(get_legal_sentence(test_sentence))


# Create bag of words
count = CountVectorizer(lowercase=False, token_pattern='[#@_$%\w\d]{2,}', max_features=200)
bag_of_words = count.fit_transform(legal_train_sentence)
# get a train set
X_train = bag_of_words.toarray()
# get this train set's label
topic_id = np.array(train[2])
y_train = topic_id

# use the bag of words to create a test set
X_test = count.transform(legal_test_sentence).toarray()

# train the DT model
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0, min_samples_leaf=20)
model = clf.fit(X_train, y_train)

predicted_y = model.predict(X_test)

# open file and write data into it
output_file = open('output.txt', 'w')
for i in range(len(test_id)):
    print(test_id[i], predicted_y[i])
