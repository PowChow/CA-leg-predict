from pymongo import MongoClient
from bson.objectid import ObjectId
import pprint
import nltk
from nltk.util import bigrams, ngrams
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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

def GetVectorizedText(db_id):
    """ @param db_id: MongoDB Object Id 
    function: Cleans, tokenizes texts and creates a list of vectorized bigrams """
    oid = db_id
    leg_text = list(db.legtext.find({'_id': oid}, {'text': 1}))[0]['text']
    
    id_feature = []
    id_text = []
    corpus = []
    feature_list = []

    #remove common words, words of one length and tokenize
    # when in the sequence should ngrams be completed // should I still go ahead with stopwords??
    raw = nltk.clean_html(leg_text)
    words = [w.lower() for w in nltk.wordpunct_tokenize(raw) if (w.isalpha() & (len(w) > 1)) ]
    wnl = nltk.WordNetLemmatizer() 
    words_lemmatize = [wnl.lemmatize(w) for w in words]  # removing word stems
    bigrams = nltk.bigrams(words_lemmatize)
    stopwords = nltk.corpus.stopwords.words('english')
    pairs = [p for p in bigrams if p[0].lower() not in stopwords and p[1].lower() not in stopwords] 

    #vectorize bigram
    #return vectorized -     

def main():
    
    logging.info('Started')
    # pulling primary bill sponsor to match with party information 
    sponsors_query = db.bills_details.find({},
        {'_id': 1,'sponsors.leg_id':1,'sponsors.type':1,'sponsors.name':1, 'action_dates.signed': 1}).limit(25) #able to limit number of records for testing

    sponsors = list(sponsors_query)
    bill_party = []
    # sponsors[0]['sponsors'][0]
    # Creates list of dict: bill database ID, passed status, legislator ID and party 
    for i in range(len(sponsors)):
        print 'getting info for item', i
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

    # # I considered loading database ID with vectorized features from LegText.py output 
    # id_vector = pickle.load(open('/Users/ppchow/data_science/CA-leg-predict/id_vector.txt', 'rb'))

    # Create a dictionary of all texts for sklearn

    #adds vectorized features of legislative text 
    #[bill_party[i]['vec_text'] = GetVectorizedText(bill_party[i]['id']) in i for range(len(bill_party))]


    logging.info('Finished')
  
if __name__ == '__main__':
    main()