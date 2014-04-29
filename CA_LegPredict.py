from pymongo import MongoClient
import sunlight
import json
import pprint
import pandas as pd
import requests #used to import text from URL

import logging
from optparse import OptionParser
import sys
from time import time

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format '%(asctime)s %(levelname)s %(message)s')
#=======================================================================================
# CONNECTION & API KEY's
#=======================================================================================

sunlight.config.API_KEY = "somelettersandnumbers"
sunlight.config.API_SIGNUP_PAGE = 'http://sunlightfoundation.com/api/accounts/register/'
sunlight.config.KEY_ENVVAR = 'SUNLIGHT_API_KEY'
sunlight.config.KEY_LOCATION = '~/.sunlight.key'

#=======================================================================================
# Url with user and password to MongoHQ database
#=======================================================================================

MONGOHQ_URL = 'mongodb://USER:PASSWORD@oceanic.mongohq.com:10036/openstates'

'''
@param param: None
@note: connect to MongoDB database and return database object
'''
def EstablishConnection():
    
    client  = MongoClient(MONGOHQ_URL)      #establish connection to database 
    db = client.openstates                  #connect to 'openstates' database
    
    return db

#Check connection to MongoDB
#change to logging
logging.info('Yes, you have a connection to MongoDB')
         

#=======================================================================================
# CALL SUNLIGHT API & PUSH TO DATABASE
#=======================================================================================

#List of states for which data will be obtained

# STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
#           "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
#           "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
#           "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
#           "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

STATES = ["CA"]


def main():
    
    logging.basicsConfig(filename='CA_LegPredict.log', level=logging.DEBUG)
    logging.info('Started')
    #============================================================================
    #Establish connection and make database object
    #============================================================================
    db  = EstablishConnection()
    
    bill_table = db.ca_bills               #bill table
    bill_d_table = db.bills_details        #bill details table
    legislator_table = db.legislators      #legislator table
    committee_table = db.committees        #committee table
    

    #============================================================================
    #Get data from bills, legislators, and committees
    #============================================================================
    
    #Getting bills using fields: state, first_name, last_name, chamber, state
    #                            active (true or false, true default), term,
    #                            district, party 
    
    #When data from all states is necessary, loop over list of states
    #for state in STATES:
    #    bills_data = sunlight.openstates.bills(state = state, chamber = 'lower')

    #bills_data = sunlight.openstates.bills(state = state, chamber = 'lower')
    #bills_data = sunlight.openstates.bills(state = state, chamber = 'upper')
    #committee_data = sunlight.openstates.committees()
    #legislators_data = sunlight.openstates.legislators()
    #committee_data = sunlight.openstates.committees()
    
    #============================================================================
    #Send data to database
    #============================================================================
    
    #PostDB(bills_data,bill_table)
    #PostDB(legislators_data, legislator_table)
    #PostDB(committee_data, committee_table)


    # From MongoDB: Created list of dicts, where dict keys are bill_id & list of sessions  
    # Getting list of tuples for API call to bill details
    bill_session = bill_table.aggregate(
        { '$group': {
             '_id': '$bill_id', 
             'session': {'$push':'$session'}}
        }).values()[1]
 
  # list_bill_session = [ list_bill_session.append(str(bill_session[b].values()[1]), list(bill_session[b].values()[0])) for b in bill_session ]
    list_bill_session =[]

    for i, entry in enumerate(bill_session):
        b_id = str(bill_session[i].values()[1]) 
        for s in xrange(1, len(bill_session[i].values()[0])):
            b_session = str(bill_session[i].values()[0][s])
            list_bill_session.append((b_id, b_session))
        
    #API calls for bill details using unique list of bill_id & session
    #example: sunlight.openstates.bill_detail(state="CA",session='20092010', bill_id='SCR 2')
    # Question the API call here may timeout, suggestions for production / doing the call in chunks?
    list_bill_details = [ sunlight.openstates.bill_detail(state='CA',session=session, bill_id=bill) for bill, session in list_bill_session  ]
    bill_d_table.insert(bill_details)

    logging.info('Finished')

if __name__ == '__main__':
    main()

#============================================================================
#Query MongoDB to create set of dataframes - OK this might no work so well, trying something else
#============================================================================

#Create unique set of values for fields and convert to PD DataFrame
unique_s = pd.DataFrame(db.ca_bills.distinct('session'), columns = ['session'])
unique_b = pd.DataFrame(db.ca_bills.distinct('bill_id'), columns=['bill_id'])
subjects = pd.DataFrame(db.ca_bills.aggregate([
            {'$project': {'subjects':1}},
            {'$unwind': '$subjects'},
            {'$group': {'_id': '$subjects', 'count':{'$sum':1}}}
        ]).values()[1])

df_bills = pd.DataFrame(list(db.ca_bills.find()))
df_bills_d = pd.DataFrame(list(db.bills_details.find()))

#both committees and legislators MongoDB tables have problems converting to panda. Are these unicode problems?
#df_comm = pd.DataFrame(list(db.committees.find()))
#df_legis = pd.DataFrame(list(db.legislators.find()))

# The closest I can get is getting a list of dataframes
df_comm = []
for row in list(db.committees.find()): 
    df_comm.append(pd.DataFrame(row)) 
#results = pd.concat(frames) #this gets a messed up result too
 
df_legsl = []
for row in list(db.legislators.find()):
    df_legsl.append(pd.DataFrame(row))

#============================================================================
#Next Step - Get URL from bills details and texts
#============================================================================

#query database for session, bill, type, URL
url_q = db.bills_details.find({},
            {'_id': 0, 
             'session': 1, 'bill_id':1, 'title':1, 'summary':1, 'versions.url':1,    
             })
df_url = pd.DataFrame(list(url_q))
types = df_url.apply(lambda x: pd.lib.infer_dtype(x.values))

# ERRORS - attempts to convert to types 
for col in types[types=='unicode'].index:
    df_url[col] = df_url[col].astype(str)

#create table with session, bill, type, URL and text using packages: URLLIB & REQUESTS 
links = []
texts = []
for row in df_url['versions']:
    for r in row: 
        for key, value in r.iteritems(): 
            link = str(value) #create a list of dictionaries to put into panda dataframe then join
            print list


                
