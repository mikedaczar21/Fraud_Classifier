
import os
import pandas as pd
import numpy as np
import dill
from scipy.stats import uniform, randint
from sklearn.utils import class_weight
from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
from data_feature_functions import print_class_report_confusion_matrix


current_dir = os.path.abspath(os.path.curdir)
trained_ensemble_dir = os.path.abspath(os.path.join(current_dir, "Trained_Ensemble_Classifier"))
boosting_dir = os.path.abspath(os.path.join(trained_ensemble_dir, "Boosting"))
bagging_dir = os.path.abspath(os.path.join(trained_ensemble_dir, "Bagging"))

trained_boosting_pred_1layer = os.path.join(boosting_dir, "Predictions")
saved_model_boosting_dir = os.path.join(boosting_dir, "Saved_Models")

trained_bagging_pred_1layer = os.path.join(bagging_dir, "Predictions")
saved_model_bagging_dir = os.path.join(bagging_dir, "Saved_Models")

def train_boosting_ensemble(X_train, X_test, y_train, y_test, boosting_type, recreate_model, **kwargs):


    # D_train = xgb.DMatrix(data = X_train, label = y_train )
    # D_test = xgb.DMatrix(data = X_test, label = y_test )
    boost_type = kwargs['model_type']


    boosting_stored_file = "Trained_{}.pk".format(boosting_type)
    boosting_path = os.path.join(saved_model_boosting_dir, boosting_stored_file)

    if (recreate_model == True) and (os.path.exists(boosting_path) == True):
        os.remove(boosting_path)

    if not (os.path.exists(boosting_path)) :


        # param = {
        # 'eta': 0.3,
        # 'max_depth': 3,
        # 'objective': "multi:softprob",
        # 'num_class': 2}
        #
        # steps = 100 # The number of training iterations

        # trained_xgb = xgb.train(param, D_train, steps)

        # Best for original unbalanced data, with 12 times more non-fraud labels than fraud
        if boost_type == 'balanced':
            xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.7176729953278143,
                  gamma=3.281453078951093, learning_rate=0.22906765020245498,
                  max_delta_step=0, max_depth=3,
                  min_child_weight=1.0246449999452354, missing=None,
                  n_estimators=806, n_jobs=1, nthread=None,
                  objective='binary:logistic', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=5, seed=None, silent=None,
                  subsample=0.8118706905581867, verbosity=1)
        else:

            xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.917062693644885,
              gamma=0.9102921524711354, learning_rate=0.17314068288862322,
              max_delta_step=0, max_depth=5, min_child_weight=2.460763947393805,
              missing=None, n_estimators=331, n_jobs=1, nthread=None,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=12, seed=None, silent=None,
              subsample=0.7784280410686237, verbosity=1)


        # Balanced with SMOTE
        # xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        #                               colsample_bynode=1, colsample_bytree=0.9735452527944342,
        #                               gamma=4.207563963333058, learning_rate=0.07451154752168451,
        #                               max_delta_step=0, max_depth=4,
        #                               min_child_weight=1.7719494212307625, missing=None,
        #                               n_estimators=401, n_jobs=1, nthread=None,
        #                               objective='binary:logistic', random_state=0, reg_alpha=0,
        #                               reg_lambda=1, scale_pos_weight=2, seed=None, silent=None,
        #                               subsample=0.9725392055968518, verbosity=1)

        trained_xgb = xgb_model.fit(X_train, y_train)

        with open(boosting_path, 'wb') as boost_file:
            dill.dump(trained_xgb, boost_file)


    else:

        with open(boosting_path, 'rb') as boost_file:
            trained_xgb = dill.load(boost_file)

    xgb_pred = trained_xgb.predict( X_test)
    xgb_prob = trained_xgb.predict_proba( X_test)

    xgb_prob_fraud = [ round(float(elem[1]), 3 ) for elem in xgb_prob]



    return xgb_pred, xgb_prob, trained_xgb, trained_boosting_pred_1layer




def train_bagging_ensemble(X_train, X_test, y_train, y_test, bagging_type, recreate_model):

    bagging_stored_file = "Trained_{}.pk".format(bagging_type)
    bagging_path = os.path.join(saved_model_bagging_dir, bagging_stored_file)


    if (recreate_model == True) and (os.path.exists(bagging_path) == True):
        os.remove(bagging_path)

    if not (os.path.exists(bagging_path)):

        # Unbalanced orginal data
        etree = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=80, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=600,
                     n_jobs=None, oob_score=False, random_state=None, verbose=0,
                     warm_start=False)

        # Tuneed for balanced data with SMOTE
        # etree = ExtraTreesClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
        #              criterion='gini', max_depth=30, max_features='sqrt',
        #              max_leaf_nodes=None, max_samples=None,
        #              min_impurity_decrease=0.0, min_impurity_split=None,
        #              min_samples_leaf=1, min_samples_split=10,
        #              min_weight_fraction_leaf=0.0, n_estimators=600,
        #              n_jobs=None, oob_score=False, random_state=None, verbose=0,
        #              warm_start=False)

        trained_etree = etree.fit(X_train, y_train)

        with open(bagging_path, 'wb') as bag_file:
            dill.dump(trained_etree, bag_file)


    else:

        with open(bagging_path, 'rb') as bag_file:
            trained_etree = dill.load(bag_file)


    etree_pred = trained_etree.predict(X_test)
    etree_prob = trained_etree.predict_proba(X_test)

    etree_prob_fraud = [ round(float(elem[1]), 3 ) for elem in etree_prob]

    return etree_pred, etree_prob , trained_etree, trained_bagging_pred_1layer



def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


def perfrom_RandomSearch(X_train, X_test, y_train,  y_test, ensemble_type):


    if (ensemble_type == "bagging"):

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 500, stop = 3000, num = 15)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 15)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [int(x) for x in np.linspace(2, 15, num = 10)]
        # Minimum number of samples required at each leaf node
        min_samples_leaf =  [int(x) for x in np.linspace(1, 10, num = 10)]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        weight = ['balanced', 'balanced_subsample']


        param_grid =  {
              'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight' : weight
        }
        etree = RandomForestClassifier()
        search = RandomizedSearchCV(estimator = etree,  param_distributions=param_grid, random_state=21, n_iter=100, cv=5, scoring='roc_auc', verbose=1, n_jobs=4, return_train_score=True)
        search.fit(X_train, y_train)
        random_best = search.best_estimator_
        search_pred = random_best.predict(X_test)


        print("The best params: {}\n".format(search.best_params_) )
        print(" The best estimator: {}\n".format(search.best_estimator_) )

        print_class_report_confusion_matrix(y_test, search_pred, "RandForest Search", "Glove Vectors")


    else:

        colsample_bytree = [float(x) for x in np.linspace(start = 0.5, stop = 2.5, num = 10)]
        params = {

        'colsample_bytree': colsample_bytree,
        'min_child_weight': uniform(1, 8),
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0.5, 4),
        "learning_rate": uniform(0.03, 0.3), # default 0.1
        "max_depth": randint(2, 6), # default 3
        "n_estimators": randint(100, 1000), # default 100
        "subsample": uniform(0.6, 0.4),
        "scale_pos_weight": randint(0, 15)
        }

        xgb_model = xgb.XGBClassifier()
        # Use ROC for scoring if balanced, otherwise better to use precision-recall curve
        search = RandomizedSearchCV(estimator = xgb_model, param_distributions=params, random_state=56, n_iter=100, cv=5, scoring='roc_auc', verbose=1, n_jobs=4, return_train_score=True)
        search.fit(X_train, y_train)

        random_best = search.best_estimator_
        search_pred = random_best.predict(X_test)

        print("The best params: {}\n".format(search.best_params_) )
        print(" The best estimator: {}\n".format(search.best_estimator_) )

        print_class_report_confusion_matrix(y_test, search_pred, "XGBoost Search", "Glove Vectors")





    pass



def perfrom_GridSearch(X_train, X_test, y_train,  y_test, ensemble_type):




    if (ensemble_type == "bagging"):

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]


        param_grid =  {
              'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap
        }
        etree = ExtraTreesClassifier()
        grid = GridSearchCV(estimator = etree, param_grid = param_grid, refit=True, verbose=0, cv=3)
        grid.fit(X_train, y_train)
        best_grid = search.best_estimator_

        grid_pred = best_grid.predict(X_test)

        print("The best params: {}\n".format(grid.best_params_) )
        print(" The best estimator: {}\n".format(grid.best_estimator_) )


        print_class_report_confusion_matrix(y_test, grid_pred, "ETree GridSearch", "Glove Vectors")


    else:

        params = {

        'subsample': [0.8, 0.9, 1, 1.1],
        'colsample_bytree': [0.8, 0.9, 1.0, 1.1],
        'min_child_weight': [1.5, 1.6, 1.7, 1.8],
        "colsample_bytree": [0.8, 0.9, 1],
        "gamma": [4.0, 4.2, 4.4],
        "learning_rate": [0.05, 0.06, 0.07, 0.08], # default 0.1
        "max_depth": [3, 4, 5], # default 3
        "n_estimators": [350, 400, 450, 500], # default 100
        "scale_pos_weight": [0.5, 1, 2, 3]
        }

        xgb_model = xgb.XGBClassifier()

        search = GridSearchCV(estimator = xgb_model, param_grid =  params, refit=True, verbose=1, cv=3)
        search.fit(X_train, y_train)

        best_grid = search.best_estimator_

        grid_pred = best_grid.predict(X_test)

        print("The best params: {}\n".format(search.best_params_) )
        print(" The best estimator: {}\n".format(search.best_estimator_) )

        print_class_report_confusion_matrix(y_test, grid_pred, "XGBoost GridSearch", "Glove Vectors")


    pass
