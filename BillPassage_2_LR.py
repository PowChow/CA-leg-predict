import pandas as pd
import numpy as np
import re
import math
import pickle

from patsy import dmatrices
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from gensim import corpora, models, similarities

def main():
    df = pd.read_csv('./saved_models/merged_df_bills_topic.csv')
    df2 = df.iloc[:, 2:-3] 
    print df2.head()
    x = df2.values
    xScaled = StandardScaler().fit_transform(x)

    y = df['bill_status'].values
    print 'starting starting'
    X_train, X_test, y_train, y_test = train_test_split(xScaled, y, 
    test_size=0.33, random_state=42)

    param_grid = [{'lr__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}]
    pipe_lr = Pipeline(steps=[ ('lr', LogisticRegression(random_state=1046)) ])
    
    #added statifiedKfolds for cv to decrease overfitting
    cvs = StratifiedKFold(y_train, n_folds = 5, shuffle=True)
    gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, 
                      scoring='accuracy', 
                      cv=cvs,n_jobs=-1)
    gs = gs.fit(X_train, y_train)

    print('Grid Search Best Score: %.4f' % gs.best_score_)
    print('Grid Search Best Parameter for C: ')
    print gs.best_params_
    print gs.best_score_
    print gs.best_estimator_

    for params, mean_score, scores in gs.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    lr = LogisticRegression(C=0.1, random_state=1036, n_jobs=-1)

    for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
         scores = cross_val_score(lr, x, y, cv=5, scoring=metric)
         print('cross_val_score', metric, scores.mean(), scores.std())

    # Print out coefficients
    lr.fit(x, y)

    df_coefs = pd.DataFrame({'features' : df2.columns, 
        'coef': lr.coef_[0,:]})
    df_coefs.sort('coef', ascending=False, inplace=True)
    df_coefs.to_csv('./saved_models/df_lr_coefs_finals.csv')
    print df_coefs
if __name__ == '__main__':
    main()
