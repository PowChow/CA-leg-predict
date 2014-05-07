
'''
Create a list of dictionaries to perform analysis on text 
'''

from pymongo import MongoClient
import sunlight
import json
import pprint
import pandas as pd
import requests #used to import text from URL
from lxml.html import fromstring
from lxml.html.clean import Cleaner
import nltk
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
from optparse import OptionParser
import sys
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
client = MongoClient('mongodb://USER:PASSWORD@oceanic.mongohq.com:10036/openstates')
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

#============================================================================
#Next Step - Get URL from bills details and texts
#============================================================================

def main():
    
    logging.info('Started')

    #list of dictionaries
    url_query = db.bills_details.find({},
        {'_id': 'id', 
            'session': 1, 'bill_id':1, 'title':1, 'subjects':1, 'versions.url':1,    
        }).limit(20) #limits for testing
    
    lod_leg = list(url_query) #makes a list of URLs
    print lod_leg[0]['_id']

    #Adds string type URL and legislative text 
    #embedded url in 'versions' - str(lod_leg[0]['versions'][0].values()[0])
    
    # for l in lod_leg[:2]: #limited to [:2] for testing and not sure of call on html limit
    #   for x in l['versions']:
    #       link = str(x.values()[0])
    #       l['url'] = link
    #       l['text'] = GetLegText(link)

    for i in range(len(lod_leg)):
        print "Getting text for item", i
        for x in lod_leg[i]['versions']:
            link = str(x.values()[0])
            lod_leg[i]['url'] = link
            lod_leg[i]['text'] = GetLegText(link)
            print lod_leg[i]['_id']

    #db.legtext.insert(lod_leg) #create a new table in MongoDB with bill text
    
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
    
    # Create dictionary for all leg texts and store the dictionary 
    dictionary = corpora.Dictionary(feature_list)
    dictionary.save('/tmp/CA-legtext.dict') # save for future reference 

    #vectorize the features and add to a list of tuple (id, vector)
    id_vector = []
    [id_vector.append( (id_feature[i][0], dictionary.doc2bow(id_feature[i][1])) ) for i in range(len(id_feature))]

    corpus = [dictionary.doc2bow(feature) for feature in feature_list]
    #corpus_id = [dictionary.doc2bow(id_vector[i][0]) for i in id_vector] # this does not work, needs list and not tuple
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus) # save to memory to access one at a time
    
    corpus_mm = corpora.MmCorpus('/tmp/corpus.mm') # loads corpus iterator
    logging.debug('corpus loaded')

    # step 1 -- initialize a model. This learns document frequencies.
    #tfidf = models.TfidfModel(vec_corpus) 

    # step 2 -- use the model to transform bunch of vectors.
    #corpus_tfidf = tfidf[corpus]

    #lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20) # initialize an LSI transformation
    #corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

    #lda = models.LdaModel(corpus, id2word=dictionary, num_topics=20, passes=20, iterations=500)

    logging.info('Finished')
  
if __name__ == '__main__':
    main()


