import numpy as np
from scipy.io import loadmat
import nltk
import re


def process_email(email_contents, verbose=True):
    word_indices = []

    # Add code to strip headers here?

    email_contents = email_contents.lower()

    # Strip all HTML
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # Handle URLS
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # Handle punctuation and special ascii characters
    email_contents = re.sub(r'[@$/\\#,-:&*+=\[\]?!(){}\'\">_<;%]+', '',
                            email_contents)

    # Tokenize
    word_list = nltk.word_tokenize(email_contents)

    for i, word in enumerate(word_list):
        # Remove punctuation and non-alphanumeric characters.
        word = re.sub(r'[^a-zA-Z0-9]', '', word)

        # If remaining word length is zero, continue.
        if len(word) < 1:
            continue

        # Stem
        try:
            word = stemmer.stem(word)
        except:
            continue

        if verbose == True:
            print
            word,
            if (i + 1) % 13 == 0: print
            '\r'

        try:
            word_indices.append(vocab_dict[word])
        except:
            continue

    if verbose == True: print
    ""

    return word_indices


vocab_list = np.loadtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/vocab.txt', dtype='str')
vocab_index_dict = {row[1]: int(row[0]) for row in vocab_list}
index_vocab_dict = {int(row[0]): row[1] for row in vocab_list}

spam_train = loadmat('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/spamTrain.mat')
spam_test =  loadmat('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/spamTest.mat')

with open('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_6/emailSample1.txt', 'r') as f:
    email_contents_1 = f.read()

stemmer = nltk.PorterStemmer()