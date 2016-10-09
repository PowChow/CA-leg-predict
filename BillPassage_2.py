from pymongo import MongoClient
from datetime import datetime
#import sunlight
import json
import pprint
import pandas as pd
import numpy as np
import re
import math
import pickle

from patsy import dmatrices
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from gensim import corpora, models, similarities

import logging
import sys
import getpass
from time import time

# Display progress logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
client = MongoClient('mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates')
db = client.openstates

#============================================================================
# FUNCTIONS
#============================================================================

def legtext_process(lda_model):
    '''
    Function loads feature_list & lda_model
    Allocates topics to each document
    Return: bill_id with lda_topic & probabilites
    '''    

    url_feature = pickle.load(open('./saved_models/url_feature_list_tuple.p', 'rb'))
    leg_corpus = corpora.Dictionary.load('./saved_models/legtext.dict')

    bill_topics_dict = []

    for i in xrange(len(url_feature)):

        url, f = url_feature[i][0], url_feature[i][1]
        bill_id = re.findall(".*?bill_id=(.*)", url)[0]
        f_bow = leg_corpus.doc2bow(f)
        t_dict = dict(lda_model[f_bow])

        t_dict.update({'bill_id': bill_id})
        bill_topics_dict.append(t_dict)

    logging.info('returning dict with bill_id and topics')
    return bill_topics_dict

def billDuration(lst):
    start = lst.get('first')
    end  = lst.get('last')
    if start and end is not None:
        _start = _datetime(start)
        _end = _datetime(end)
        delta = _end  - _start
        val = (delta.total_seconds() *(1/3600.0)*(1/24.0))
    return val

def _datetime(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

def isBill(lst):
    key = ['bill']
    if any(x in key for x in lst):
        return True
    else:
        return False

#0 == bill dies else 1 == bill passed
# no == bill dies else no == bill passed
def billStatus(lst):
    str_lst = lst[-1]['action'].split()
    key_die = ['died','Died','without','Without', 'vetoed','Vetoed', 'From', 'printer']
    key_pass = ['chaptered', 'Chaptered']

    if any(x in key_die for x in str_lst):
        return 0
    if any(x in key_pass for x in str_lst):
        return 1

def primarySponsors(lst):
    i = 0
    for e in lst:
        if e['type'] == 'primary':
            i += 1
    return i

def coSponsors(lst):
    i = 0
    for e in lst:
        if e['type'] == 'cosponsor':
            i += 1
    return i

#============================================================================
# MAIN
#============================================================================

def main():
    
    logging.info('Started')
    
    #==================================================================================================
    #Query MongoDB to pull relevant data 
    #==================================================================================================
    
    bills_details = list(db.bills_details.find({'state':'ca', 'type': 'bill'}, 
        {'_id': 1, 'session':1, 'chamber': 1, 'sponsors': 1, 'sponsors.leg_id':1, 
           'scraped_subjects': 1, 'subjects':1, 'type': 1,
           'action_dates': 1, 'votes': 1, 'actions': 1, 'versions.url': 1}).limit(10) )
    legtext = list(db.legtext.find().limit(10))
    logging.info('Data succesfully obtained from MongoDB.\n')

    logging.info('Creating legis dataframe...........\n')
    df_bills_d = pd.DataFrame(bills_details)
    logging.info('Finished creating DataFrame........\n')

    ## Only need to load this dataframe once
    logging.info('Load model and get topics by URL...........\n')
    # lda_tfidf_model = models.LdaModel.load('./saved_models/lda_tfidf_model_100.pkl')
    # df_bill_topics = pd.DataFrame.from_dict( legtext_process(lda_tfidf_model) )
    # df_bill_topics.fillna(-0.0001)
    # df_bill_topics.to_csv('./saved_models/df_bill_topics.csv')
    df_bill_topics = pd.read_csv('./saved_models/df_bill_topics.csv')
    df_bill_topics.fillna(-0.0001, inplace=True)
    logging.info('bill topics dataframe loaded....\n')
    print(df_bill_topics.head())

    logging.info('Apply transformation to bills_details......\n')
    df_bills_d['bill_id'] = df_bills_d['versions'].map(lambda lst: re.findall(".*?bill_id=(.*)", str(lst[0]['url']))[0])
    df_bills_d['bill_duration'] = df_bills_d['action_dates'].apply(lambda lst: billDuration(lst))
    df_bills_d['bill_status'] = df_bills_d['actions'].map(lambda lst: billStatus(lst))
    df_bills_d['primary_sponsors'] = df_bills_d['sponsors'].map(lambda lst: primarySponsors(lst))
    df_bills_d['co_sponsors'] = df_bills_d['sponsors'].map(lambda lst: coSponsors(lst))
    df_bills_d['leg_id'] = df_bills_d['sponsors'].map(lambda lst: lst[0]['leg_id'])
    df_bills_d = df_bills_d.drop(['action_dates', 'actions', 'session', 'subjects', 
        'scraped_subjects', 'votes', 'type', 'sponsors'], axis = 1)
    df_bills_d.fillna(0, inplace = True)   
    df_bills_d_merged = pd.merge(df_bill_topics, 
                                df_bills_d[['bill_status', 'bill_id', 'actions']], 
                                on='bill_id', how='inner')
    df_bills_d_merged.to_csv('merged_df_bills_topics.csv')
    print 'Prints Merged Bill Details', df_bills_d_merged.head(), len(df_bills_d_merged)
    logging.info('Done applying transformation to DataFrame........\n')

    #===============================================================================
    # APPLY LOGISTIC REGRESSION MODEL TO DATAFRAME
    #===============================================================================

    # x = df_bills_d_merged.fillna(0).values
    # xScaled = preprocessing.StandardScaler().fit_transform(x)

    # y = df_bills_d_merged['bill_status'].values

    # X_train, X_test, y_train, y_test = train_test_split(xScaled, y, 
    #     test_size=0.33, random_state=42)

    # param_grid = [{'lr__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]

    # pipe_lr = Pipeline(steps=[ ('lr', LogisticRegression(random_state=1)) ])

    # #added statifiedKfolds for cv to decrease overfitting
    # cvs = StratifiedKFold(y_train, n_folds = 5, shuffle=True)
    
    # gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, 
    #                   scoring='accuracy', 
    #                   cv=cvs,n_jobs=-1)
    # gs = gs.fit(X_train, y_train)
    # print('Grid Search Best Score: %.4f' % gs.best_score_)
    # print('Grid Search Best Parameter for C: ')
    # print gs.best_params_
    
    # for params, mean_score, scores in gs.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() / 2, params))

    # for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
    # scores = cross_val_score(gs, X, y, cv=3, scoring=metric)
    # print('cross_val_score', metric, scores.mean(), scores.std())


# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
if __name__ == '__main__':
    main()
