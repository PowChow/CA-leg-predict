from pymongo import MongoClient
from datetime import datetime
#import sunlight
import json
import pprint
import pandas as pd
import numpy as np
import requests #used to import text from URL
from patsy import dmatrices
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

import math

import logging
#from optparse import OptionParser
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
        {'_id': 1, 'session':1, 'chamber': 1, 'sponsors': 1, 'sponsors.leg_id':1, 'scraped_subjects': 1, 'subjects':1, 'type': 1,
           'action_dates': 1, 'votes': 1, 'actions': 1}).limit(10000) )

    legis_details = list(db.legislators.find({'state': 'ca','level':'state'}, 
            {'_id': 1,'leg_id': 1,'party': 1,'district': 1,'active': 1 ,'chamber': 1}).limit(5000) )

    logger.info('Data succesfully obtained from MongoDB.\n')

    logger.info('Creating legis dataframe...........\n')
    df_legis = pd.DataFrame(legis_details)
    df_bills_d = pd.DataFrame(bills_details)
    logger.info('Finished creating DataFrame........\n')

    logger.info('Uploading median income by district data')
    fnames = np.array(['locations', 'district', 'chamber', 'med_ann_income'])
    income_df = pd.read_csv('./Med_Family_Income_20082012.csv', names=fnames)
    legis_income_df = pd.merge(income_df, df_legis, on=['chamber', 'district'], how='right')
    legis_income_df = legis_income_df.drop(['_id', 'district', 'chamber'], axis=1)
    logger.info('Combined legislation and income dataframes')

    logger.info('Apply transformation to DataFrame......\n')
    df_bills_d['bill_duration'] = df_bills_d['action_dates'].apply(lambda lst: billDuration(lst))
    df_bills_d['bill_status'] = df_bills_d['actions'].map(lambda lst: billStatus(lst))
    df_bills_d['primary_sponsors'] = df_bills_d['sponsors'].map(lambda lst: primarySponsors(lst))
    df_bills_d['co_sponsors'] = df_bills_d['sponsors'].map(lambda lst: coSponsors(lst))
    df_bills_d['leg_id'] = df_bills_d['sponsors'].map(lambda lst: lst[0]['leg_id'])
    df_bills_d = df_bills_d.drop(['action_dates', 'actions', 'session', 'subjects', 'scraped_subjects', 'votes', 'type', 'sponsors'], axis = 1)
    df_bills_d.fillna(0, inplace = True)   
    df_bills_d_merged = pd.merge(legis_income_df, df_bills_d, on='leg_id', how='outer')
    print 'Prints Merged Dataframe', df_bills_d_merged
    logger.info('Done applying transformation to DataFrame........\n')


    #===============================================================================
    # APPLY NAIVE BAYES MODEL TO DATAFRAME
    #===============================================================================
    #df_bills_d_merged.describe()
    #df_bills_d_merged[df_bills_d_merged['bill_status'] == 1 ].describe()
    df_bills_d_merged.head()
    y, X = dmatrices('bill_status ~ bill_duration + primary_sponsors + co_sponsors + locations + party - 1', 
        data=df_bills_d_merged, return_type='dataframe')
    yy = y['bill_status[yes]']

    clf = BernoulliNB().fit(X, yy)
    #clf = GaussianNB().fit(X,yy)
    print clf.intercept_
    print math.exp(clf.intercept_)
    print 'NB Score/R2', clf.score(X,yy)

    print "Coefs", clf.coef_[0]
    
    top = np.argsort(clf.coef_[0])
    print top
    print clf.coef_[0][top]
    print 'X.columns top', X.columns[top]

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
if __name__ == '__main__':
    main()
