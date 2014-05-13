from pymongo import MongoClient
from bson.objectid import ObjectId
import pprint
import nltk
from nltk.util import bigrams, ngrams
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC

import logging
from optparse import OptionParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
client = MongoClient('mongodb://USER:PASSWORD@oceanic.mongohq.com:10036/openstates')
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

def GetBigramsVector(db_id):
    """ @param db_id: MongoDB Object Id 
    function: Returns vectorized bigrams. Cleans, tokenizes texts and creates a list of bigrams for texts """
    oid = db_id
    leg_text = list(db.legtext.find({'_id': oid}, {'text': 1}))[0]['text']

    # remove common words, words of one length and tokenize
    # when in the sequence should ngrams be completed // should I still go ahead with stopwords??
    # raw = nltk.clean_html(leg_text)
    # words = [w.lower() for w in nltk.wordpunct_tokenize(raw) if (w.isalpha() & (len(w) > 1)) ]
    # wnl = nltk.WordNetLemmatizer() 
    # words_lemmatize = [wnl.lemmatize(w) for w in words]  # removing word stems
    # bigrams = nltk.bigrams(words_lemmatize)
    # stopwords = nltk.corpus.stopwords.words('english')
    # pairs = [p for p in bigrams if p[0].lower() not in stopwords and p[1].lower() not in stopwords] 

    raw = nltk.clean_html(leg_text)
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    bigram_features = analyze(raw)

    X = bigram_vectorizer.fit_transform(bigram_features).toarray()
    
    return X

def main():
    
    logging.info('Started')
    # pulling primary bill sponsor to match with party information 
    sponsors_query = db.bills_details.find({},
        {'_id': 1,'sponsors.leg_id':1,'sponsors.type':1,'sponsors.name':1, 
                  'action_dates.signed': 1}).limit(5) #able to limit number of records for testing

    sponsors = list(sponsors_query)
    bill_party = []
    # sponsors[0]['sponsors'][0]
    # Creates list of dict: bill database ID, passed status, legislator ID and party 
    for i in range(len(sponsors)):
        bill_dbid = sponsors[i]['_id']
        leg_id = sponsors[i]['sponsors'][0]['leg_id']
       
        if leg_id == None: 
            leg_id = 'CA0000'
            party = sponsors[i]['sponsors'][0]['name']
        else: 
            party = GetParty(leg_id)
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

    # Do I need to create/ update a dictionary? This pulls MongoDB_Id and texts
    # all_legtext = list(db.legtext.find({}, {'text': 1}).limit(25))

    #adds vectorized features of legislative text with function
    # for i in range(len(bill_party)):
    #     vec = GetBigramsVector(bill_party[i]['id'])
    #     bill_party[i]['vec'] = vec
    # logging.info('loaded vectorized bigrams')

    for i in range(len(bill_party)):
        #oid = bill_party[i]['id']
        print "Getting text for item", i, bill_party[i]['id']
        leg_text = list(db.legtext.find({'_id': bill_party[i]['id']}, {'text': 1}))[0]['text']
        raw = nltk.clean_html(leg_text)
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        analyze = bigram_vectorizer.build_analyzer()
        bigram_features = analyze(raw)
        bill_party[i]['vec'] = bigram_vectorizer.fit_transform(bigram_features).toarray()
    logging.info('loaded vectorized bigrams')

    print bill_party[0]
    print bill_party[0].values()
    print bill_party[0].keys()

    logging.info('Finished')
  
if __name__ == '__main__':
    main()