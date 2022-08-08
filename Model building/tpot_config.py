# -*- coding: utf-8 -*-
"""
TPOT config used to search the best logistic regression model. Here a lot of other models can be searched but since we are interested in simple linear models,
only the parameters of logistic regression are optimized, but a lot more complex pipelines can be built. (https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier.py)

@author: BM387
"""

## Main  model building and tuning parameter search configuration
classifier_config_dict = {

    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers 
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    }
}



## Trying out other models and combinations
""" 
classifier_config_dict = {
    # Preprocesssors

    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers
     'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 200),
        'weights': ["distance"],
        'metric': ["cosine", "euclidean"]  
    },
}

classifier_config_dict = {

    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers
    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    },
}

  # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers
    'xgboost.XGBClassifier': {
        'n_estimators': [500],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05, .1],
        'subsample': [0.25, 0.5, 0.75, 1],
        'min_child_weight': [1, 3],
        'n_jobs': [-1],
        'verbosity': [0]
    },
}


# Check the TPOT documentation for information on the structure of config dicts
classifier_config_dict = {
    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers
    'xgboost.XGBClassifier': {
        'n_estimators': [500],
        'max_depth': [1, 2],
        'subsample': [1],
        'n_jobs': [-1],
        'verbosity': [0]}
}


classifier_config_dict = {

    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers 
    'xgboost.XGBClassifier': {
     }
}

print("Tunning the parameters of the logistic regression model")
classifier_config_dict = {

    # Preprocesssors
    'sklearn.preprocessing.MinMaxScaler': {
    },
    # Classifiers 
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    }
}

"""