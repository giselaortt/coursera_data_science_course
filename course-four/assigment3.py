
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

def answer_one():
    
    t = spam_data['target']
    
    return len(t[ t==1 ])*100/len(spam_data)



answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

from sklearn.feature_extraction.text import CountVectorizer
import operator

def answer_two():
    vect = CountVectorizer().fit(X_train)
    return sorted([(token, len(token)) for token in vect.vocabulary_.keys()], key=operator.itemgetter(1), reverse=True)[0][0]

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    transformed_train = vect.transform(X_train)
    transformed_test = vect.transform(X_test)
    model = MultinomialNB(alpha=0.1)
    model = model.fit( X = transformed_train, y= y_train )
    predictions = model.predict( transformed_test)
    
    return roc_auc_score(y_test, predictions)

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    tfidf_score = X_train_vectorized.max(0).toarray()[0]
    feature_names = vect.get_feature_names()
    data = pd.Series( tfidf_score, index = feature_names )
    data = data.sort_index(ascending=True).sort_values(kind = 'mergesort')

    return (data[0:20], data[-20:].sort_index(ascending=True).sort_values(ascending = False, kind = 'mergesort'))


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

import sklearn
from sklearn.metrics import roc_auc_score

def answer_five():
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = sklearn.naive_bayes.MultinomialNB(alpha=0.1)
    model.fit( X_train_vectorized, y_train )
    X_test_vectorized = vect.transform( X_test )
    predictions = model.predict(X_test_vectorized)

    return roc_auc_score(y_test, predictions)

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

def answer_six():
    
    t = spam_data['target']
    spam = spam_data[ t==1 ]
    not_spam = spam_data[ t==0 ]
    
    return (not_spam['text'].str.len().mean(), spam['text'].str.len().mean())

answer_six()


# The following function has been provided to help you combine new features into the training data:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*


from sklearn.svm import SVC

def answer_seven():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    model = SVC( C=10000 )
    model.fit( X_train_vectorized, y_train )
    X_test_vectorized = vect.transform( X_test )
    X_test_vectorized = add_feature( X_test_vectorized, X_test.str.len() )
    predictions = model.predict(X_test_vectorized)

    return roc_auc_score(y_test, predictions)

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*


def answer_eight():
    
    t = spam_data['target']
    spam = spam_data['text'][ t==1 ]
    not_spam = spam_data['text'][ t==0 ]
    avg_non_character_spam = spam.str.count('\d').mean()
    avg_non_character_non_spam = not_spam.str.count('\d').mean()
    
    return (avg_non_character_non_spam, avg_non_character_spam)


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*


from sklearn.linear_model import LogisticRegression

def answer_nine():
    return


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

def answer_ten():
    
    t = spam_data['target']
    spam = spam_data['text'][ t==1 ]
    not_spam = spam_data['text'][ t==0 ]
    avg_non_character_spam = spam.str.count('\W').mean()
    avg_non_character_non_spam = not_spam.str.count('\W').mean()
    
    return (avg_non_character_non_spam, avg_non_character_spam)


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

from sklearn.linear_model import LogisticRegression

def answer_eleven():
    return

answer_eleven()

