#!/usr/bin/env python
# coding: utf-8


from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

import nltk
import string
import re

import sys
import tweepy
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.svm import SVC


print("Loadingsadad")
raw = pd.read_csv("neuron_main/data.csv")
raw['Tweet'] = raw.Tweet.apply(lambda x: x.replace('.', ''))
raw.head()

raw.shape[0]

# Create lists for tweets and label
Tweet = []
Labels = []
wpt = nltk.WordPunctTokenizer()
wordnet_lemmatizer = nltk.WordNetLemmatizer()

for row in raw["Tweet"]:
    # tokenize words
    words = wpt.tokenize(row)
    # remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    # remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''", '``', "rt", "https", "’", "“", "”", "\u200b", "--", "n't", "'s", "...", "//t.c"]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    # Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    # print(lemma_list)
    Tweet.append(lemma_list)

for label in raw["Text Label"]:
    Labels.append(label)

c = []

for x in Tweet:
    s = ''
    for y in x:
        s = s + ' ' + y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s = s.lstrip()
    c.append(s)

#print(c)


# from sklearn.feature_extraction.text import CountVectorizer
# 
# cv = CountVectorizer(min_df=0., max_df=1.)
# # have given min_document_freqn = 0.0 ->  which means ignore 
# # terms that appear in less than 1% of the documents 
# 
# # and max_document_freqn = 1.0 ->  which means ignore terms that appear 
# # in more than 100% of the documents".
# # In short nothing is to be ignored. all data values would be considered 
# 
# cv_matrix = cv.fit_transform(c)
# 
# cv_matrix = cv_matrix.toarray()
# 
# cv_matrix

# 
# 
# 
# # the final preprocessing step is to divide data into training and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(cv_matrix, Labels, test_size = 0.20,random_state=0)
# 
# # TYPE your Code here
# # Training the Algorithm. Here we would use simple SVM , i.e linear SVM
# #simple SVM
# 
# from sklearn.svm import SVC
# svclassifier=SVC(kernel='linear')
# 
# 
# 
# # classifying linear data
# 
# 
# # kernel can take many values like
# # Gaussian, polynomial, sigmoid, or computable kernel
# # fit the model over data
# 
# svclassifier.fit(X_train,y_train)
# 
# 
# # Making Predictions
# 
# y_pred=svclassifier.predict(X_test)
# 
# 
# # Evaluating the Algorithm
# 
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# 
# print(confusion_matrix(y_test,y_pred))
# 
# print(classification_report(y_test,y_pred))
#
# print(accuracy_score(y_test,y_pred))
# 
# # Remember : for evaluating classification-based ML algo use  
# # confusion_matrix, classification_report and accuracy_score.


tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(c)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()

lol = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

# In[ ]:

# the final preprocessing step is to divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tv_matrix, Labels, test_size=0.20, random_state=0)

# TYPE your Code here
# Training the Algorithm. Here we would use simple SVM , i.e linear SVM
# simple SVM

svclassifier = SVC(kernel='linear')

# classifying linear data


# kernel can take many values like
# Gaussian, polynomial, sigmoid, or computable kernel
# fit the model over data

svclassifier.fit(X_train, y_train)

# Making Predictions

y_pred = svclassifier.predict(X_test)


# Evaluating the Algorithm

#print("-------------------First conf matrix")
#print(confusion_matrix(y_test,y_pred))

#print("classification Report")
#print(classification_report(y_test,y_pred))


#print("accuracy_score")
#print(accuracy_score(y_test,y_pred))

# Remember : for evaluating classification-based ML algo use  
# confusion_matrix, classification_report and accuracy_score.
# And for evaluating regression-based ML Algo use Mean Squared Error(MSE), ...


# In[ ]:


model = GaussianNB()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

#print(confusion_matrix(y_test,y_pred))

#print(classification_report(y_test,y_pred))

#print(accuracy_score(y_test,y_pred))


# In[ ]:


results = model.predict_proba(X_test)[1]

#print("---------Print result")
#print(results)


svm = SVC(probability=True)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
LABEL = svm.predict(X_test)
#print("----------------LABEl[1:10]")

output_proba = svm.predict_proba(X_test)
#print("----------------output_proba")
#print(output_proba)





# credentials  --> put your credentials here
consumer_key = "EgjCgVxayfmnj68PXiVKu1PVj"
consumer_secret = "QFszM57LLN4GyIOmPBG9Q1bjWceDj7yGBIEuF2xWCyt93BbZ8Z"
access_token = "1199917326285426690-MWVeRP6oBVW7b6X5FEHJhwpNBB2bXf"
access_token_secret = "BqqWCmW1hUGlZhLLyXyjeFYVByDoJlPGIYwpGU949Y89x"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
L = []
author = []


class CustomStreamListener(tweepy.StreamListener):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 10

    def on_status(self, status):
        # print(status.text)
        global L
        L.append(status.text)
        author.append(status.user.screen_name)
        self.counter += 1
        if self.counter < self.limit:
            return True
        else:
            return False

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True  # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return False  # Don't kill the stream


sapi = tweepy.streaming.Stream(auth, CustomStreamListener())
sapi.filter(locations=[-74.36141, 40.55905, -73.704977, 41.01758])

a = [
    "Fuck Your mum",
    "I say to ypu somethink",
    "I want to die",
    "Suck my dick",
    "Lick my ass",
    "Today Is Monday",
    "Today Is Suthday",
    "My name is Alex",
    "but pls die fuck off and suck my dick",
]

main = pd.DataFrame(a, columns=['Tweet'])

# print("MAIN")
# MODELS.py["Tweet"][0] = "Fuck Your mum"
# print(MODELS.py["Tweet"])
# print(author)


posts = []
for row in main["Tweet"]:
    # tokenize words
    words = wpt.tokenize(row)
    # remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    # remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''", '``', "rt", "https", "’", "“", "”", "\u200b", "--", "n't", "'s", "...", "//t.c"]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    # Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    # print("ROW IN MAIN")
    # print(lemma_list)
    posts.append(lemma_list)

refine = []

for x in posts:
    s = ''
    for y in x:
        s = s + ' ' + y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s = s.lstrip()
    refine.append(s)

# print(refine)    

# re_tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
re_tv_matrix = tv.transform(refine)
re_tv_matrix = re_tv_matrix.toarray()

# print(re_tv_matrix)

OUTPUT = svm.predict(re_tv_matrix)

SEVERITY = svm.predict_proba(re_tv_matrix)
number = 0

#for i in MODELS.py["Tweet"]:
   # print("number " + str(number) + ": " + i + "  SEVERITY: " + str(SEVERITY[number][0]) + " THIS TEXT IS: " + OUTPUT[
     #   number] + ";")
   # number += 1


# print(OUTPUT)
# print(SEVERITY)
def isBuling(text):
    # tokenize words
    matrix = []
    words = wpt.tokenize(text)
    # remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    # remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''", '``', "rt", "https", "’", "“", "”", "\u200b", "--", "n't", "'s", "...", "//t.c"]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    # Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    s = ''
    for y in lemma_list:
        s = s + ' ' + y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s = s.lstrip()
    matrix.append(s)
    re_tv_matrix = tv.transform(matrix)
    re_tv_matrix = re_tv_matrix.toarray()
    OUTPUT = svm.predict(re_tv_matrix)
    SEVERITY = svm.predict_proba(re_tv_matrix)
    # print(re_tv_matrix)
    return SEVERITY[0][0]



