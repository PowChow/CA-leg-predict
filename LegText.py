
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

import logging
from optparse import OptionParser
import sys
from time import time

logging.basicConfig(level=logging.INFO, format = '%(asctime)s %(levelname)s %(message)s')

#============================================================================
# FUNCTIONS
#============================================================================

client = MongoClient('mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates')
db = client.openstates

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
            {'_id': 0, 
             'session': 1, 'bill_id':1, 'title':1, 'subjects':1, 'versions.url':1,    
             }).limit(20)
    lod_leg = list(url_query)
    #add text  

    #lod_leg_url = {'text': str(GetLegText(lod_leg[0][''])) for k, v in lod_leg[0].iteritems()}
    one_instance = lod_leg[0]
    print "one instance:", one_instance
    lod_leg_url = one_instance['versions'][0]['url']
    text GetLegText(lod_leg_url)
    #Adds string type URL and legislative text 
    #embedded url in 'versions' - str(lod_leg[0]['versions'][0].values()[0])
    for l in lod_leg[:2]: #limited to [:2] for testing and not sure of call on html limit
        l['urls'] = [x['url'] for x in l['versions']]
        l['texts'] = [x['text'] for x in l['versions']]
            link = str(x.values()[0])
            l['url'] = link
            l['text'] = GetLegText(link)

    # -----------------using one legtext as an example---------------------------------------
    db.legtext.insert(lod_leg[:2]) #limited to [:2] for testing
    a = GetLegText('http://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=200920100AB123') #lengthy text
    raw = nltk.clean_html(a)
    words = [w.lower() for w in nltk.wordpunct_tokenize(raw) if (w.isalpha() & (len(w) > 1)) ]
    filtered_words = [w for w in words if w not in nltk.corpus.stopwords.words('english')]
    #take out words in a stoplist
    vocab = sorted(set(words)) # normalized words, build the vocabulary
    
    wnl = nltk.WordNetLemmatizer() # removing word stems that are only a dictionary
    [wnl.lemmatize(t) for t in words]

    #run through gensim to determine categories for each legislative text - lesson 15
    print "Done!"
    logging.info('Finished')

#============================================================================
#Send data to database
#============================================================================
    
if __name__ == '__main__':
    main()


