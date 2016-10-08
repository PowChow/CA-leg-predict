#================================================================================
# Modeling code for random trees adopted from SKLEARN example
#=================================================================================

print(__doc__)
from pymongo import MongoClient
from bson.objectid import ObjectId
import pprint
import nltk
from bs4 import BeautifulSoup
from nltk.util import bigrams, ngrams
import pickle
from sklearn import linear_model, cross_validation, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn import clone
from sklearn.externals.six.moves import xrange
import numpy as np
import pylab as pl

import logging
from optparse import OptionParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
client = MongoClient('mongodb://galaHELP:pythonisfun!@oceanic.mongohq.com:10036/openstates')
client2 = MongoClient('mongodb://admin:party000cat!@oceanic.mongohq.com:10036/openstates')
db = client2.openstates


def main():
    global X
    logging.info('Started')

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

    bill_party = []
    for i in range(db.bill_party.count()):
        print oid, i
        oid = list(db.bill_party.find({},{'id':1}))[i]['id']
        party = list(db.bill_party.find({},{'party':1}))[i]['party']
        #print "Getting text for item", i, bill_party[i]['id']
        leg_text = list(db.legtext.find({'_id': oid}, {'text': 1}))[0]['text']
        soup = BeautifulSoup(leg_text)
        raw = soup.get_text()  
        raw = raw.replace('\t','').replace('\r','').replace('\n','').replace('>>','')
        # bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        bigram_features = analyze(raw)

        k = ['id', 'features', 'party','raw']
        v = [oid, bigram_features, party, raw ]
        bill_party[i].append(dict(zip(k,v)))

        # bill_party[i]['vec'] = bigram_vectorizer.fit_transform(bigram_features).toarray()
    
    party_options = {'democratic': 0, 'republican': 1}
    X = bigram_vectorizer.fit_transform([x['raw'] for x in bill_party if x['party'].lower() in party_options])
    print bigram_vectorizer
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

    #====================================================================================
    # Random Forests Modeling and Plotting 
    #===================================================================================
    
    # Parameters
    n_classes = 2
    n_estimators = 30
    plot_colors = "ryb"
    cmap = pl.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    RANDOM_SEED = 9  # fix the seed on each iteration ???

    plot_idx = 1

    models = [DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(n_estimators=n_estimators),
              ExtraTreesClassifier(n_estimators=n_estimators),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                 n_estimators=n_estimators)]

   
    for model in models:
        # We use all the features where the SKLEARN example choose specific ones
        X = data
        y = targets

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        clf = clone(model)
        clf = model.fit(X, y)

        scores = clf.score(X, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print model_details + " with all features has a score of", scores

    ###################### Commented out plotting ############################################
    #     pl.subplot(3, 4, plot_idx)
    #     if plot_idx <= len(models):
    #         # Add a title at the top of each column
    #         pl.title(model_title)

    #     # Now plot the decision boundary using a fine mesh as input to a
    #     # filled contour plot
    #     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))

    #     # Plot either a single DecisionTreeClassifier or alpha blend the
    #     # decision surfaces of the ensemble of classifiers
    #     if isinstance(model, DecisionTreeClassifier):
    #         Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #         Z = Z.reshape(xx.shape)
    #         cs = pl.contourf(xx, yy, Z, cmap=cmap)
    #     else:
    #         # Choose alpha blend level with respect to the number of estimators
    #         # that are in use (noting that AdaBoost can use fewer estimators
    #         # than its maximum if it achieves a good enough fit early on)
    #         estimator_alpha = 1.0 / len(model.estimators_)
    #         for tree in model.estimators_:
    #             Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    #             Z = Z.reshape(xx.shape)
    #             cs = pl.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

    #     # Build a coarser grid to plot a set of ensemble classifications
    #     # to show how these are different to what we see in the decision
    #     # surfaces. These points are regularly space and do not have a black outline
    #     xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),
    #                                          np.arange(y_min, y_max, plot_step_coarser))
    #     Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
    #     cs_points = pl.scatter(xx_coarser, yy_coarser, s=15, c=Z_points_coarser, cmap=cmap, edgecolors="none")

    #     # Plot the training points, these are clustered together and have a
    #     # black outline
    #     for i, c in zip(xrange(n_classes), plot_colors):
    #         idx = np.where(y == i)
    #         pl.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
    #                    cmap=cmap)

    #     plot_idx += 1  # move on to the next plot in sequence

    # pl.suptitle("Classifiers on feature subsets of the Party Words dataset")
    # pl.axis("tight")

    # pl.show()

    logging.info('Finished')
  
if __name__ == '__main__':
    main()
