
'''
Create a list of dictionaries to perform analysis on text 
'''

from pymongo import MongoClient
#import sunlight
import json
import pprint
import pandas as pd
import requests #used to import text from URL
from lxml.html import fromstring
from lxml.html.clean import Cleaner
import nltk
import pickle
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
from optparse import OptionParser
import sys
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
client = MongoClient('mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates')
db = client.openstates

#============================================================================
# FUNCTIONS
#============================================================================

def GetLegText(link):
    '''
    Function fetches legislative text from url
    @para link: string type http://http://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id='session'+'bill_id'
    '''
    
    html = requests.get(link).text
    html_encode = html.encode('ascii','ignore') #convert text to string

    doc = fromstring(html_encode)

    tags = ['h1','h2','h3','h4','h5','h6',
           'div', 'span', 
           'img', 'area', 'map']
    args = {'meta':False, 'safe_attrs_only':False, 'page_structure':True, 
           'scripts':True, 'style':True, 'links':True, 'remove_tags':tags}
    cleaner = Cleaner(**args)

    path = '/html/body'
    body = doc.xpath(path)[0]

    clean_doc = (cleaner.clean_html(body).text_content().encode('ascii', 'ignore')).strip().split('Version:',1)[1]

    return clean_doc


def main():
    
    logging.info('Started')
    print 'Started'
    #============================================================================
    #Transform data - pull bills and url from MongoDB to get html text and clean
        #only need to do this once 
    #============================================================================
    #originally pulled from bill details data to populate legtext table
    # url_query = db.bills_details.find({},
    #     {'_id': 1, 
    #         'session': 1, 'bill_id':1, 'title':1, 'subjects':1, 'versions.url':1,    
    #     }) #able to limit number of records for testing
    # logging.debug('MongoDB query completed for url and bill data')

    # Adds string type URL and legislative text 
    #embedded url in 'versions' - str(lod_leg[0]['versions'][0].values()[0])
    # print "Getting text for ids"
    # for i in range(len(lod_leg)):
    #     #print "Getting text for item", i
    #     for x in lod_leg[i]['versions']:
    #         link = str(x.values()[0])
    #         lod_leg[i]['url'] = link
    #         lod_leg[i]['text'] = GetLegText(link)

    #============================================================================
    #Update database - upload html cleaned text and unique IDs to database
        #only need to do this once 
    #============================================================================
    # db.legtext.insert(lod_leg) #create a new table in MongoDB with bill text
    # print 'finished updating MongoDB'

    #============================================================================
    # Natural Language Processing - get text from database and prepare for models
    #============================================================================
    lod_leg = list(db.legtext.find()) 

    id_feature = []
    id_text = []
    corpus = []
    feature_list = []

    for i in range(len(lod_leg)):
        _id, text = lod_leg[i]['_id'], lod_leg[i]['text']

        #remove common words, words of one length and tokenize
        raw = nltk.clean_html(lod_leg[i]['text'])
        words = [w.lower() for w in nltk.wordpunct_tokenize(raw) if (w.isalpha() & (len(w) > 1)) ]
        words_filtered = [w for w in words if w not in nltk.corpus.stopwords.words('english')]
        
        # removing word stems and create a list of features 
        wnl = nltk.WordNetLemmatizer() 
        feature = [wnl.lemmatize(t) for t in words_filtered]

        #create list of tuples: (1) MongoDB Object_id and texts (2) MongoDB Object_id and features
        id_text.append((_id, text))
        id_feature.append((_id, feature))
        feature_list.append(feature)  
    
    # # CREATE DICTIONARY for all leg texts and store the dictionary - only need to do once
    # dictionary = corpora.Dictionary(feature_list)
    # dictionary.compactify() # remove gaps in id sequence after words that were removed
    # dictionary.save('/Users/ppchow/data_science/CA-leg-predict/tmp/legtext.dict') # save for future reference 
    # print 'saved dictionary'

    #============================================================================
    # RUNNING LDA and LSI models - load dictionary and corpus
    #============================================================================

    # LOAD DICTIONARY
    dictionary = corpora.Dictionary.load('/Users/ppchow/data_science/CA-leg-predict/tmp/legtext.dict')
    print dictionary

    #vectorize the features and add to a list of tuple (id, vector)
    id_vector = []
    [id_vector.append( (id_feature[i][0], dictionary.doc2bow(id_feature[i][1])) ) for i in range(len(id_feature))]

    # output IDs associated with features and vectors for each document
    with open('/tmp/id_feature_50.txt', 'wb') as ff:
        pickle.dump(id_feature, ff)  
    with open('/tmp/id_vector_50.txt', 'wb') as fv:
        pickle.dump(id_vector, fv)
    logging('output to files')  

    #vectorize: turn each document (list of works) into a vectorized bag of words
    # corpus = [dictionary.doc2bow(feature) for feature in feature_list]
    # corpora.MmCorpus.serialize('/Users/ppchow/data_science/CA-leg-predict/tmp/corpus.mm', corpus) 
    # print 'saved corpus'
    
    # LOAD corpus 
    corpus_mm = corpora.MmCorpus('/Users/ppchow/data_science/CA-leg-predict/tmp/corpus.mm')
    print corpus_mm

    # print 'step 1 - intializing model'
    #step 1 -- initialize a model. This learns document frequencies.
    tfidf = models.TfidfModel(corpus_mm) 

    # print 'step 2 - use model to transform bunch of vectors'
    # step 2 -- use the model to transform bunch of vectors.
    corpus_tfidf = tfidf[corpus_mm]

    # sunlight subject data about 45 categories
    logging('running LSI model')
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=50) # initialize an LSI transformation
    lsi.save('/Users/ppchow/data_science/CA-leg-predict/tmp/lsi_model_50.pkl')
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    corpus_lsi.save('/Users/ppchow/data_science/CA-leg-predict/tmp/corpus_lsi_model_50.pkl')
    print 'lsi model completed'

    logging('running LDA models')
    lda_tfidf = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=50, update_every=1, chunksize=1000, passes=1, iterations=500) #model easier to interpert results
    lda_tfidf.save('/Users/ppchow/data_science/CA-leg-predict/tmp/lsa_tfidf_model_50.pkl')
    print 'lda model completed'

    #============================================================================
    # ANALYSIS - Load corpus, dictionary, models and python lists
    #============================================================================

    # # LOAD DICTIONARY
    # dictionary = corpora.Dictionary.load('/Users/ppchow/data_science/CA-leg-predict/tmp/legtext.dict')
    
    # # LOAD CORPUS 
    # corpus_mm = corpora.MmCorpus('/Users/ppchow/data_science/CA-leg-predict/tmp/corpus.mm')
 
    # # LOAD MODELS
    # lsi_model = models.LsiModel.load('/Users/ppchow/data_science/CA-leg-predict/tmp/lsi_model.pkl')
    # # lda_corpus_model = models.LdaModel.load('/Users/ppchow/data_science/CA-leg-predict/tmp/lsa_model.pkl')
    # lda_tfidf_model = models.LdaModel.load('/Users/ppchow/data_science/CA-leg-predict/tmp/lsa_tfidf_model.pkl')
    
    # # PRINT MODEL TOPICS
    # print 'lsi model'
    # print lsi_model.print_topics(25)
    # print 'lda corpus model'
    # print lda_corpus_model.print_topics(25)
    # print 'lsi tfidf model'
    # print lda_tfidf_model.print_topics(25)


    logging.info('Finished')
  
if __name__ == '__main__':
    main()


