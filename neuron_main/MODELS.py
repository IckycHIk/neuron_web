import pickle

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
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from neuron.serializers import TrainTextSerializer

from neuron.models import TrainText

from neuron.models import MlModels

from mpl_toolkits import mplot3d

def learnModelSVC():
    c, Labels = getTrainData()
    print("Create TV_Matrix")
    tv, tv_matrix = initVc(c)
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

    print("SCV Classifier Train")
    svclassifier.fit(X_train, y_train)

    # Making Predictions

    y_pred = svclassifier.predict(X_test)

    # Evaluating the Algorithm

    # print("-------------------First conf matrix")
    # print(confusion_matrix(y_test,y_pred))

    # print("classification Report")
    # print(classification_report(y_test,y_pred))

    # print("accuracy_score")
    # print(accuracy_score(y_test,y_pred))

    # Remember : for evaluating classification-based ML algo use
    # confusion_matrix, classification_report and accuracy_score.
    # And for evaluating regression-based ML Algo use Mean Squared Error(MSE), ...

    # In[ ]:

    print("Model Train")
    model = GaussianNB()

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # print(confusion_matrix(y_test,y_pred))

    # print(classification_report(y_test,y_pred))

    # print(accuracy_score(y_test,y_pred))

    # In[ ]:

    results = model.predict_proba(X_test)[1]

    # print("---------Print result")
    # print(results)


    print("SVM Train")
    svm = SVC(probability=True)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    LABEL = svm.predict(X_test)

    pickle_data = pickle.dumps(svm)
    MlModels.objects.create(model=pickle_data)



    # print("----------------LABEl[1:10]")

    output_proba = svm.predict_proba(X_test)

    output_proba = svm.predict_proba(X_test)
    print("Lent Model end")
    return svm, tv


def getTrainData():
    raw = TrainText.objects.all()
    Text = []
    Label = []

    for i in raw:
        Text.append(i.text)
        if i.isBulling:
            Label.append("Bullying")
        else:
            Label.append("Non-Bullying")

    print(Text)
    print(Label)

    # raw['Tweet'] = raw.Tweet.apply(lambda x: x.replace('.', ''))

    number = 0
    print("Text From DB Get")
    # for i in raw["Text Label"]:
    #     data = {"text": raw["Tweet"][number]}
    #     if raw["Text Label"][number] == "Bullying":
    #         data["isBulling"] = True
    #     else:
    #         data["isBulling"] = False
    #
    #     serializer = TrainTextSerializer(data=data)
    #
    #
    #     if serializer.is_valid():
    #         print(number)
    #         serializer.save()
    #     else:
    #         print("ERROR")
    #     number += 1
    #
    #
    # raw.shape[0]
    print("Start Lern Model")
    Tweet = []
    Labels = []
    wpt = nltk.WordPunctTokenizer()
    wordnet_lemmatizer = nltk.WordNetLemmatizer()

    for row in Text:
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

    for label in Label:
        Labels.append(label)

    print("Create List")
    c = []

    for x in Tweet:
        s = ''
        for y in x:
            s = s + ' ' + y
        s = re.sub('[^A-Za-z0-9" "]+', '', s)
        s = s.lstrip()
        c.append(s)
    return c, Labels


def initVc(c):
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(c)
    tv_matrix = tv_matrix.toarray()

    vocab = tv.get_feature_names()

    lol = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
    return tv, tv_matrix


print("Loading")

myModel = MlModels.objects.all()
print(myModel)

if not myModel:
    svm, tv = learnModelSVC()



    print("Lent Model end")
else:
    print(myModel[0])

    c,Labels = getTrainData()

    tv, tv_matrix = initVc(c)

    svm = pickle.loads(myModel[0].model)

    print("Initialize Models")


def isBulling(text):
    # tokenize words
    wpt = nltk.WordPunctTokenizer()
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
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