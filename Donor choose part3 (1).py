#!/usr/bin/env python
# coding: utf-8

# In[83]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os

from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
import chart_studio
from scipy.sparse import hstack


# In[84]:


data=pd.read_csv("preprocessed_data1.csv")
data=data.drop(['project_tile'],axis=1)


# In[85]:


data.head(2)


# <h1>1. K Nearest Neighbor</h1>
# 

# Here we will be applying knnn to the model but before that we need to transform all the paramters in dataset to structured format, so to do these we will be using BOW,TFIDF,W2V for text data and for numerical values we will be using normalizer and other text values we will be using One hot encoder ,
# so we need to check every model accuracy using auc score , to find the best k value we can go with hyperparamter tunning or grid search tunning to find the best k value as you know auc score is determined by TPR and FPR 

# In[86]:


essays=data['essay'].values


# # BOW Representation

# In[87]:


y = data['project_is_approved'].values
X = data.drop(['project_is_approved'], axis=1)
X.head(1)


# <h2> Splitting data into Train and cross validation(or test): Stratified Sampling</h2>
# 

# In[88]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)


# # Encoding of Essay

# In[159]:


nbbow=[]

vectorizer = CountVectorizer()
print(X_train.shape, y_train.shape)
print(X_cv.shape, y_cv.shape)
print(X_test.shape, y_test.shape)

print("="*100)


vectorizer = CountVectorizer(min_df=10,ngram_range=(1,4))# 
#min_df=10(The words which are found atlest in 10 of the rows)

vectorizer.fit(X_train['essay'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_essay_bow = vectorizer.transform(X_train['essay'].values)
X_cv_essay_bow = vectorizer.transform(X_cv['essay'].values)
X_test_essay_bow = vectorizer.transform(X_test['essay'].values)

print("After vectorizations")
print(X_train_essay_bow.shape, y_train.shape)
print(X_cv_essay_bow.shape, y_cv.shape)
print(X_test_essay_bow.shape, y_test.shape)
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# <h3> encoding categorical features: School State</h3>

# In[160]:



vectorizer = CountVectorizer()
vectorizer.fit(X_train['school_state'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_state_ohe = vectorizer.transform(X_train['school_state'].values)
X_cv_state_ohe = vectorizer.transform(X_cv['school_state'].values)
X_test_state_ohe = vectorizer.transform(X_test['school_state'].values)

print("After vectorizations")
print(X_train_state_ohe.shape, y_train.shape)
print(X_cv_state_ohe.shape, y_cv.shape)
print(X_test_state_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# # For all the categorical features present below we are applying one hot encoders

# <h3> encoding categorical features:teacher_prefix </h3>

# In[161]:


vectorizer = CountVectorizer()
vectorizer.fit(X_train['teacher_prefix'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_teacher_ohe = vectorizer.transform(X_train['teacher_prefix'].values)
X_cv_teacher_ohe = vectorizer.transform(X_cv['teacher_prefix'].values)
X_test_teacher_ohe = vectorizer.transform(X_test['teacher_prefix'].values)

print("After vectorizations")
print(X_train_teacher_ohe.shape, y_train.shape)
print(X_cv_teacher_ohe.shape, y_cv.shape)
print(X_test_teacher_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# <h3> encoding categorical features:project_grade_category </h3>

# In[162]:


vectorizer = CountVectorizer()
vectorizer.fit(X_train['project_grade_category'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_pgc_ohe = vectorizer.transform(X_train['project_grade_category'].values)
X_cv_pgc_ohe = vectorizer.transform(X_cv['project_grade_category'].values)
X_test_pgc_ohe = vectorizer.transform(X_test['project_grade_category'].values)

print("After vectorizations")
print(X_train_pgc_ohe.shape, y_train.shape)
print(X_cv_pgc_ohe.shape, y_cv.shape)
print(X_test_pgc_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# <h3> encoding categorical features:clean_categories </h3>

# In[163]:


vectorizer = CountVectorizer()
vectorizer.fit(X_train['clean_categories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_clca_ohe = vectorizer.transform(X_train['clean_categories'].values)
X_cv_clca_ohe = vectorizer.transform(X_cv['clean_categories'].values)
X_test_clca_ohe = vectorizer.transform(X_test['clean_categories'].values)

print("After vectorizations")
print(X_train_clca_ohe.shape, y_train.shape)
print(X_cv_clca_ohe.shape, y_cv.shape)
print(X_test_clca_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# <h3> encoding categorical features:clean_subcategories </h3>

# In[164]:


vectorizer = CountVectorizer()
vectorizer.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_clsc_ohe = vectorizer.transform(X_train['clean_subcategories'].values)
X_cv_clsc_ohe = vectorizer.transform(X_cv['clean_subcategories'].values)
X_test_clsc_ohe = vectorizer.transform(X_test['clean_subcategories'].values)

print("After vectorizations")
print(X_train_clsc_ohe.shape, y_train.shape)
print(X_cv_clsc_ohe.shape, y_cv.shape)
print(X_test_clsc_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)
nbbow.extend(vectorizer.get_feature_names())


# # For the numerical features we are using the normalization technique

# <h3> encoding numerical feature:price </h3>

# In[167]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['price'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['price'].values.reshape(-1,1))

X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
X_cv_price_norm = normalizer.transform(X_cv['price'].values.reshape(-1,1))
X_test_price_norm = normalizer.transform(X_test['price'].values.reshape(-1,1))



print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
print(X_cv_price_norm.shape, y_cv.shape)
print(X_test_price_norm.shape, y_test.shape)
print("="*100)
nbbow.extend('price')


# <h3> encoding numerical feature:teacher_number_of_previously_posted_projects </h3>

# In[166]:


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['price'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))


X_train_tnpp_norm = normalizer.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_cv_tnpp_norm = normalizer.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
X_test_tnpp_norm = normalizer.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))



print("After vectorizations")
print(X_train_tnpp_norm.shape, y_train.shape)
print(X_cv_tnpp_norm.shape, y_cv.shape)
print(X_test_tnpp_norm.shape, y_test.shape)
print("="*100)

nbbow.extend('teacher_number_of_previously_posted_projects')


# #Here we will be concatenating all the features

# In[97]:


X_trbow = hstack((X_train_essay_bow, X_train_state_ohe, X_train_teacher_ohe, X_train_price_norm,X_train_tnpp_norm,X_train_clsc_ohe,X_train_clca_ohe,X_train_pgc_ohe)).tocsr()
X_crbow = hstack((X_cv_essay_bow, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_price_norm,X_cv_tnpp_norm,X_cv_clsc_ohe,X_cv_clca_ohe,X_cv_pgc_ohe)).tocsr()
X_tebow = hstack((X_test_essay_bow, X_test_state_ohe, X_test_teacher_ohe, X_test_price_norm,X_test_tnpp_norm,X_test_clsc_ohe,X_test_clca_ohe,X_test_pgc_ohe)).tocsr()


# In[98]:


print("Final Data matrix")
print(X_trbow.shape, y_train.shape)
print(X_crbow.shape, y_cv.shape)
print(X_tebow.shape, y_test.shape)
print("="*100)


# In[171]:


nbbwords=nbbow.copy()
nbtfidf=nbbow.copy()


# <h3> After Applying BOW for text data,One hot encoder for categorical features and Normalizing the numerical features we have divided the dataset in the ratio of 77 train and 33 % test we also used cross validation to make the accuracy better so we divided the training set again into 77 % train and 33 % cv so after that we used hstack and concatenated the columns to the three paramters , Now to make sure the model doen't overfit or underfit we need to find the right value of k, so we will be using tunning the model using random search or hyperparamter search or using loops and plot a graph to visually find the best value of k  </h3>

# We will be using simple loop first to get the best value of k and the K fold cross validation we are performing is 1 here

# In[17]:


#Basically I am creating a function to iterate and predict the probabilities batch wise.

def batch_predict(clf, data):
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    y_data_pred = []
    tr_loop = data.shape[0] - data.shape[0]%1000
    #consider you X_tr shape is 49041, then your tr_loop will be 49041 - 49041%1000 = 49000(Here we are actually taking the remainder 49041/1000=41 and 
    # subracting it from 49041 to get the multiplrs of 1000'''') )
    # in this for loop we will iterate unti the last 1000 multiplier
    for i in range(0, tr_loop, 1000): #(0,49000,1000)  # so here the loop will run for 49 times 
        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])
    # we will be predicting for the last data points
    if data.shape[0]%1000 !=0:
        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])
    
    return y_data_pred


# Here in the below code i am using auc score to determine the best k value

# In[18]:


# The loop is here running 490 times 

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
"""
y_true : array, shape = [n_samples] or [n_samples, n_classes]
True binary labels or binary label indicators.

y_score : array, shape = [n_samples] or [n_samples, n_classes]
Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of
decisions (as returned by “decision_function” on some classifiers). 
For binary y_true, y_score is supposed to be the score of the class with greater label.
"""
train_auc = []
cv_auc = []
K = [3, 15, 25, 35,45,55,71,81,91,101] # we can alos use range(1,50) but due to memory cnstrains we are only selecting 10 k values to choose
for i in tqdm(K):
    neigh = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    neigh.fit(X_trbow, y_train)

    y_train_pred = batch_predict(neigh, X_trbow)    
    y_cv_pred = batch_predict(neigh, X_crbow)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs        
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(K, train_auc, label='Train AUC')
plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, train_auc, label='Train AUC points')
plt.scatter(K, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


#  So from the above plot we can see the K=101 is the best we got as both the curves are converging at this point , we are selecting this point as both cvauc curve and  train auc curve converges at this point

# Here in the below code we will be using random search cv and the cv=3 so we are dividing the train dataset into 3 parts and using auc score to fidn the best k value
# 

# In[99]:


from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

neigh = KNeighborsClassifier(n_jobs=-1)
parameters = {'n_neighbors':[3, 15, 25, 51, 101]}
clf = RandomizedSearchCV(neigh, parameters, cv=3, scoring='roc_auc',return_train_score=True)
clf.fit(X_trbow, y_train)

results = pd.DataFrame.from_dict(clf.cv_results_)
results = results.sort_values(['param_n_neighbors'])


train_auc= results['mean_train_score']
train_auc_std= results['std_train_score']
cv_auc = results['mean_test_score'] 
cv_auc_std= results['std_test_score']
K =  results['param_n_neighbors']


plt.plot(K, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
#plt.gca().fill_between(K, train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(K, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(K, cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(K, train_auc, label='Train AUC points')
plt.scatter(K, cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("Hyper parameter Vs AUC plot")
plt.grid()
plt.show()

results.head()


# In[20]:


# from the error plot we choose K such that, we will have maximum AUC on cv data and gap between the train and cv is less
# Note: based on the method you use you might get different hyperparameter values as best one
# so, you choose according to the method you choose, you use gridsearch if you are having more computing power and note it will take more time
# if you increase the cv values in the GridSearchCV you will get more rebust results.

#here we are choosing the best_k based on forloop results
best_k = 101


# we have found out the best value of k so we will be plotting auc curve auc using TPR and FPR
# Auc represents the measure of seperability and Receiver operating characteristics represents the probability

# In[100]:


from sklearn.metrics import roc_curve, auc


neigh = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
neigh.fit(X_trbow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = batch_predict(neigh, X_trbow)    
y_test_pred = batch_predict(neigh, X_tebow)

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# As we know auc curve and roc curve can be used for bimary classification

# In[101]:


# we are writing our own function for predict, with defined thresould
# we will pick a threshold that will give the least fpr
def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[102]:


print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))


#  # Applying AW2V using KNN

# In[24]:


# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
with open('glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


# In[25]:


# average Word2Vec for train essay values
# compute average word2vec for each review.
avg_w2v_vectors_train = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_train['essay'].values): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_train.append(vector)

print(len(avg_w2v_vectors_train))
print(len(avg_w2v_vectors_train[0]))
print(avg_w2v_vectors_train[0])


# In[26]:



# avg w2v for cv data values
avg_w2v_vectors_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_cv['essay'].values): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_cv.append(vector)


# In[27]:


#avg w2v for test set.
avg_w2v_vectors_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(X_test['essay'].values): # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_test.append(vector)


# In[28]:


X_tr = hstack((avg_w2v_vectors_train, X_train_state_ohe, X_train_teacher_ohe, X_train_price_norm,X_train_tnpp_norm,X_train_clsc_ohe,X_train_clca_ohe,X_train_pgc_ohe)).tocsr()
X_cr = hstack((avg_w2v_vectors_cv, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_price_norm,X_cv_tnpp_norm,X_cv_clsc_ohe,X_cv_clca_ohe,X_cv_pgc_ohe)).tocsr()
X_te = hstack((avg_w2v_vectors_test,X_test_state_ohe, X_test_teacher_ohe, X_test_price_norm,X_test_tnpp_norm,X_test_clsc_ohe,X_test_clca_ohe,X_test_pgc_ohe)).tocsr()





print("Final Data matrix")
print(X_tr.shape, y_train.shape)
print(X_cr.shape, y_cv.shape)
print(X_te.shape, y_test.shape)
print("="*100)


# Hyper parameter tunning

# In[29]:


train_auc = []
cv_auc = []
K = [3, 15, 25, 51, 101]
for i in tqdm(K):
    neigh = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    neigh.fit(X_tr, y_train)

    y_train_pred = batch_predict(neigh, X_tr)    
    y_cv_pred = batch_predict(neigh, X_cr)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs        
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(K, train_auc, label='Train AUC')
plt.plot(K, cv_auc, label='CV AUC')

plt.scatter(K, train_auc, label='Train AUC points')
plt.scatter(K, cv_auc, label='CV AUC points')

plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
best_k=101


from sklearn.metrics import roc_curve, auc


neigh = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
neigh.fit(X_tr, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = batch_predict(neigh, X_tr)    
y_test_pred = batch_predict(neigh, X_te)

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("K: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))


# # Naive Bayes Assignment

# We will be using multinomial naive bayes here and use BOW and TFIDF for encoding Text Data.
# One thing we need to remember if auc score is greater then 0.5 it means model is able to make good classification

# In[144]:


# we will be using the below parameters to find the best K value.

from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
import warnings
import math
warnings.filterwarnings("ignore")




parameters={'alpha':[0.00001,0.0005, 0.0001,0.1,0.5,1,5,51,101,1001]}


nb = MultinomialNB(fit_prior=False)

clf = RandomizedSearchCV(nb,parameters , cv=3, scoring='roc_auc',return_train_score=True)
clf.fit(X_trbow, y_train)
results = pd.DataFrame.from_dict(clf.cv_results_)


train_auc= results['mean_train_score']

train_auc_std= results['std_train_score']
cv_auc = results['mean_test_score'] 
cv_auc_std= results['std_test_score']

parameters['alpha']=[math.log10(i) for i in parameters['alpha']]
print('Best score: ',clf.best_score_)
print('Alpha value with best score: ',clf.best_params_)


# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
#plt.gca().fill_between(K, train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')
plt.plot(parameters['alpha'], train_auc, label='CV AUC')

plt.plot(parameters['alpha'], cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(K, cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(parameters['alpha'], train_auc, label='Train AUC points')
plt.scatter(parameters['alpha'], cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("Alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("Hyper parameter Vs AUC plot")
plt.grid()
plt.show()

results




# 
# <br> We have used randomized search cv for finding best alpha so to avoid model overfitting and underfitting.<br>
# We have used 3 fold validation,cv=3 and trained over model <br>
# We have used log10(alpha) to make the plot simple and look good.<br>
# In the above plot we have used only 10 values of alpha to get the best model using roc score which is given by auc curve ,here the best values of alpha we got is 1 with 0.68 
# 
# 

# In[145]:


from sklearn.metrics import roc_curve, auc


nb = MultinomialNB(alpha=1,fit_prior=False)
nb.fit(X_trbow, y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

y_train_pred = batch_predict(neigh, X_trbow)    
y_test_pred = batch_predict(neigh, X_tebow)

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[146]:


def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[147]:


from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
cmtrain =confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t))
print(cmtrain)
print("Test confusion matrix")
cmtest=confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t))
print(cmtest)


# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt     
print("Train confusion matrix")

ax= plt.subplot();
sns.heatmap(cmtrain, annot=True,cmap='Blues',ax=ax,fmt="d");
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_ylim(2.0, 0)
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['False','Correct']); 
ax.yaxis.set_ticklabels(['False','Correct']);


# In[149]:


import seaborn as sns
import matplotlib.pyplot as plt     
print("Test confusion matrix")

ax= plt.subplot();
sns.heatmap(cmtest, annot=True,cmap='Blues',ax=ax,fmt="d");
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_ylim(2.0, 0)
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['False','Correct']); 
ax.yaxis.set_ticklabels(['False','Correct']);


# In[175]:


pos_class_prob_sorted = nb.feature_log_prob_[1].argsort()
print("The top 10 positive features are")
print(np.take(nbbwords,pos_class_prob_sorted[-10:]))

nag_class_prob_sorted = nb.feature_log_prob_[0].argsort()
print("The top 10 nagative features are")
print(np.take(nbbwords,nag_class_prob_sorted[-10:]))


# In[ ]:




