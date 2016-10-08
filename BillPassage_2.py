from pymongo import MongoClient
from datetime import datetime
#import sunlight
import json
import pprint
import pandas as pd
import numpy as np
import re

import requests #used to import text from URL
from patsy import dmatrices
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

import math

import logging
import sys
import getpass
from time import time

# Display progress logs

logger = logging.getLogger('logger')
hdlr = logging.FileHandler('/Users/'+ getpass.getuser() + '/Desktop/logger.log')
formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

#Username and password are read only. Let me know if you have to write anything!
MONGOHQ_URL = 'mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates'

'''
@param param: None
@note: connect to MongoDB database and return database object
'''
def EstablishConnection():
    client  = MongoClient(MONGOHQ_URL)      #establish connection to database 
    db = client.openstates                  #connect to 'openstates' database
    logger.info('Yes, you have a connection to MongoDB')
    return db      

def main():
    
    logger.info('Started')
    #============================================================================
    #Establish connection and make database object
    #============================================================================
    db  = EstablishConnection()
    
    bill_table = db.ca_bills               #bill table
    bill_d_table = db.bills_details        #bill details table
    legislator_table = db.legislators      #legislator table
    committee_table = db.committees        #committee table
    
    #==================================================================================================
    #Query MongoDB to pull relevant data 
    #==================================================================================================

    # try:
    #     bills_details = list(db.bills_details.find({'state':'ca', 'type': 'bill'}, 
    #         {'_id': 1, 'session':1, 'chamber': 1, 'sponsors': 1, 'sponsors.leg_id':1, 'scraped_subjects': 1, 'subjects':1, 'type': 1,
    #         'action_dates': 1, 'votes': 1, 'actions': 1}).limit(10000) )

    #     legis_details = list(db.legislators.find({'state': 'ca','level':'state'}, 
    #         {'_id': 1,'leg_id': 1,'party': 1,'district': 1,'active': 1 ,'chamber': 1}).limit(10000) )

    #     logger.info('Data succesfully obtained from MongoDB.\n')
    # except:
    #     logger.info('Something went with wrong Querying MongoDB.\n')
    #     pass
    
    bills_details = list(db.bills_details.find({'state':'ca', 'type': 'bill'}, 
        {'_id': 1, 'session':1, 'chamber': 1, 'sponsors': 1, 'sponsors.leg_id':1, 
           'scraped_subjects': 1, 'subjects':1, 'type': 1,
           'action_dates': 1, 'votes': 1, 'actions': 1, 'versions.url': 1}).limit(5000) )
    legtext = list(db.legtext.find().limit(5000))
    logger.info('Data succesfully obtained from MongoDB.\n')

    logger.info('Creating legis dataframe...........\n')
    df_legis = pd.DataFrame(legis_details)
    df_bills_d = pd.DataFrame(bills_details)
    logger.info('Finished creating DataFrame........\n')

    logger.info('Load model and get topics by URL...........\n')
    lda_tfidf_model = models.LdaModel.load('./saved_models/lda_tfidf_model_100.pkl')
    df_bill_topics = pd.DataFrame.from_dict( legtext_process(model=lda_tfidf_model) )
    logger.info('Completed matching url and text...........\n')

    logger.info('Apply transformation to bills_details......\n')
    df_bills_d['bill_id'] = df_bills_d['versions'].map(lambda lst: re.findall(".*?bill_id=(.*)", str(lst[0]['url'])))
    df_bills_d['bill_duration'] = df_bills_d['action_dates'].apply(lambda lst: billDuration(lst))
    df_bills_d['bill_status'] = df_bills_d['actions'].map(lambda lst: billStatus(lst))
    df_bills_d['primary_sponsors'] = df_bills_d['sponsors'].map(lambda lst: primarySponsors(lst))
    df_bills_d['co_sponsors'] = df_bills_d['sponsors'].map(lambda lst: coSponsors(lst))
    df_bills_d['leg_id'] = df_bills_d['sponsors'].map(lambda lst: lst[0]['leg_id'])
    df_bills_d = df_bills_d.drop(['action_dates', 'actions', 'session', 'subjects', 'scraped_subjects', 'votes', 'type', 'sponsors'], axis = 1)
    df_bills_d.fillna(0, inplace = True)   
    df_bills_d_merged = pd.merge(df_bill_topics, df_bills_d, on='bill_id', how='inner')
    df_bills_d_merged.to_csv('merged_df_bills_topics.csv')
    print 'Prints Merged Bill Details', df_bills_d_merged.head(), len(df_bills_d_merged)
    logger.info('Done applying transformation to DataFrame........\n')


#     #===============================================================================
#     # APPLY NAIVE BAYES MODEL TO DATAFRAME
#     #===============================================================================
#     #df_bills_d_merged.describe()
#     #df_bills_d_merged[df_bills_d_merged['bill_status'] == 1 ].describe()
#     df_bills_d_merged.head()
#     y, X = dmatrices('bill_status ~ bill_duration + primary_sponsors + co_sponsors + locations + party - 1', 
#         data=df_bills_d_merged, return_type='dataframe')
#     yy = y['bill_status[yes]']

#     clf = BernoulliNB().fit(X, yy)
#     #clf = GaussianNB().fit(X,yy)
#     print clf.intercept_
#     print math.exp(clf.intercept_)
#     print 'NB Score/R2', clf.score(X,yy)

#     print "Coefs", clf.coef_[0]
    
#     top = np.argsort(clf.coef_[0])
#     print top
#     print clf.coef_[0][top]
#     print 'X.columns top', X.columns[top]

# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def legtext_process(lda_model):
    '''
    Function loads feature_list & lda_model
    Allocates topics to each document
    Return: bill_id with lda_topic & probabilites
    '''    

    url_feature = pickle.load(open('./saved_models/url_feature_list_tuple.p', 'rb')
    
    #sample: uf = [{'url': 'https:1', 1: .034, 7: .34, 8: 0}]
    uf_listdict = []

    for i in xrange(len(url_feature)):

        url, f = url_feature[i]['url'], url_feature[i]['text']
        #url to bill_id
        bill_id = re.findall(".*?bill_id=(.*)", s)
        f_bow = dict_load.doc2bow(f)
        t_dict = dict(lda_model(f_bow))
        t_dict.update{'bill_id': bill_id}

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
    return 'no'
  if any(x in key_pass for x in str_lst):
    return 'yes'

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

# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# #////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
# if __name__ == '__main__':
#     main()
