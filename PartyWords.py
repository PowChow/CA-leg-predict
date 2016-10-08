from pymongo import MongoClient
from bson.objectid import ObjectId
import pprint
import nltk
from nltk.util import bigrams, ngrams
from bs4 import BeautifulSoup
import urllib
import pickle
from sklearn import linear_model, cross_validation, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pylab as party_linearSVC

import logging
from optparse import OptionParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
client = MongoClient('mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates')
db = client.openstates

# function - get party from MongoDB
def GetParty(db_id):
    # p_sponsor[0]['sponsors'][0]['leg_id']
    # party_query = list(db.legislators.find({'all_ids': 'CAL000077'}, {'_id': 0, 'party':1, 'active':1}))
    """ Get party of legislator from MongoDB ObjectId and returns party distinction with Error Handling when 
    the sponsor is not a member of the legislation
    """
    try: 
        party_query = list(db.legislators.find({'all_ids': db_id}, {'_id':0, 'party':1}))
        return str(party_query[0]['party'])
    except:
        logging.debug('not CA legislator')


def main():
    global X
    logging.info('Started')
    # pulling primary bill sponsor to match with party information 
    sponsors_query = db.bills_details.find({},
        {'_id': 1,'sponsors.leg_id':1,'sponsors.type':1,'sponsors.name':1, 
                  'action_dates.signed': 1}).limit(1000) #able to limit number of records for testing

    sponsors = list(sponsors_query)
    bill_party = []
    # sponsors[0]['sponsors'][0]
    # Creates list of dict: bill database ID, passed status, legislator ID and party 
    # This table should be created in MongoDB
    logging.info('get sponsor list')    

    for i in range(len(sponsors)):
        bill_dbid = sponsors[i]['_id']
        leg_id = sponsors[i]['sponsors'][0]['leg_id']
       
        if leg_id == None: 
            leg_id = 'CA0000'
            party = sponsors[i]['sponsors'][0]['name']
        else: 
            party = GetParty(leg_id)
            print party
            if party == None:
                party = sponsors[i]['sponsors'][0]['name']
       
        if sponsors[i]['action_dates']['signed'] == None:
            bill_signed = False
        else:
            bill_signed = True

        k = ['id', 'leg_id', 'party','passed']
        v = [bill_dbid, leg_id, party, bill_signed]
        bill_party.append(dict(zip(k,v)))

    logging.info('populated list of sponsor and party')    
    # note to self/presentation: show number of bills sponsored by non-legislators
    # graph bills by party that passed .....     

    # Do I need to create/ update a dictionary? This pulls MongoDB_Id and texts
    # all_legtext = list(db.legtext.find({}, {'text': 1}).limit(25))

    #adds vectorized features of bigrams using function
    # for i in range(len(bill_party)):
    #     vec = GetBigramsVector(bill_party[i]['id'])
    #     bill_party[i]['vec'] = vec
    # logging.info('loaded vectorized bigrams')

    bigram_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df =1)
    analyze = bigram_vectorizer.build_analyzer()


    #longest 
    for i in range(len(bill_party)):
        #oid = bill_party[i]['id']
        #print "Getting text for item", i, bill_party[i]['id']
        leg_text = list(db.legtext.find({'_id': bill_party[i]['id']}, {'text': 1}))[0]['text']
        #raw = nltk.clean_html(leg_text) function no longer available
        soup = BeautifulSoup(leg_text)
        raw = soup.get_text()
        # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        bigram_features = analyze(raw)
        bill_party[i]['features'] = bigram_features
        bill_party[i]['raw'] = raw
        # bill_party[i]['vec'] = bigram_vectorizer.fit_transform(bigram_features).toarray()
    
    party_options = {'democratic': 0, 'republican': 1}
    X = bigram_vectorizer.fit_transform([x['raw'] for x in bill_party if x['party'].lower() in party_options])
    logging.info('in place of bigram_vectorizer output')
    logging.info('loaded tfidf vectorized bigrams')

    # Creates numpy arrays, results = party and features = vectorized words  
    # party only = democrat or republican and vectorized text
    bp_target = []
    bp_data = []
    for i in range(len(bill_party)):
        if bill_party[i]['party'].lower() in ('democratic', 'republican'): 
            bp_target.append( party_options[bill_party[i]['party'].lower()] )            
        else:
            continue

    targets = np.array(bp_target)
    data = X.toarray()
    
    #=====================================================================================
    # Train different models - Linear, Logistic, Random Linear
    #=====================================================================================

    #  Supported Vector Classification
    logging.info('Linear Support Vector Classification')
    clf = LinearSVC(loss='l2')
    clf = clf.fit(data,targets)
    # print clf
    # print 'LinearSVC Coef', clf.coef_
    # print 'LinearSVC Intercept', clf.intercept_
    # print 'LinearSVC Score/R2', clf.score(data,targets)

    with open('party_linearSVC.pkl', 'wb') as mclf:
        pickle.dump(clf, mclf)
    logging.info('output LinearSVC to party_linearSVC.pkl')

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, targets, test_size=0.4, random_state=0)
    clfCV = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # print clfCV
    # print 'training shape', X_train.shape, y_train.shape
    # print 'testing shape', X_test.shape, y_test.shape
    # print 'Test Score', clfCV.score(X_test, y_test)
    # print 'Train Score', clfCV.score(X_train, y_train)

    # Logistic Regression 
    logging.info('Logistic Regression')
    # Insert GridSearch Here
    logreg_l1 = linear_model.LogisticRegression(C=1.0, penalty='l1')
    logreg_l2 = linear_model.LogisticRegression(C=1.0, penalty='l2')
    logreg_l1.fit(data,targets)
    logreg_l2.fit(data,targets)

    # print logreg_l1
    # print logreg_l2
    # print 'Pseudo-R2 penalty l1', logreg_l1.score(data,targets)
    # print 'Pseudo-R2 penalty l2', logreg_l2.score(data,targets)
    # print 'LogReg l1 Coef', logreg_l1.coef_
    # print 'LogReg l1 Intercept', logreg_l1.intercept_

    with open('party_logreg_l1.pkl', 'wb') as lr1:
        pickle.dump(logreg_l1, lr1)
    logging.info('output Logistic regression to party_logreg_l1.pkl')

    with open('party_logreg_l2.pkl', 'wb') as lr2:
        pickle.dump(logreg_l2, lr2)
    logging.info('output Logistic regression to party_logreg_l2.pkl')

    # Random Forests
    # See other python file

    logging.info('Finished')
  
if __name__ == '__main__':
    main()
