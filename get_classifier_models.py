import os

current_dir = os.path.abspath(os.path.curdir)
classifier_dir = os.path.join(current_dir, "Trained_Classifiers")

from ensemble_classifier import train_bagging_ensemble, train_boosting_ensemble, perfrom_GridSearch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.pipeline import Pipeline
import dill

glove_dim = 100

def create_SVM_model(kernel, **kwargs):


    c_param = kwargs['c_param']
    gamma_param = kwargs['gamma_param']


    if (kernel.lower() == "sgd" ):
        svm_model = Pipeline([

            #('linear SVM', SVC(kernel="linear", C=c_param, gamma = gamma_param, probability=True))
            ('linear SVM', linear_model.SGDClassifier(max_iter=1000, tol=1e-3))
        ])
    elif (kernel.lower() == "radial" ):
        svm_model = Pipeline([

            ('radial SVM', SVC(kernel="rbf", class_weight='balanced', C=c_param, gamma = gamma_param, probability=True))
        ])


    else:
        svm_model = Pipeline([
            ('radial SVM', SVC(kernel="rbf", class_weight='balanced',C=c_param, gamma = gamma_param, probability=True))
        ])

    return svm_model



def get_classifier(feature_train, feature_test, label_train, **kwargs):


    word_vector_type = kwargs['word_Vector']
    classifier = kwargs['classifier']
    glove_dim = kwargs['word_vector_dim']

    classifier_file = "{}_{}_{}D.pk".format(classifier, word_vector_type, glove_dim)
    classifier_path = os.path.join(classifier_dir, classifier_file)


    if not os.path.exists(classifier_path):

        wordVec_pipeline = create_SVM_model("radial" , c_param = 0.005, gamma_param=1)

        if (classifier.lower() == "bayes"):


            wordVec_pipeline = Pipeline([
                ('multiNB', MultinomialNB() )
            ])


        elif (classifier.lower() == "svm"):

            if (word_vector_type.lower() == "glove"):
                wordVec_pipeline = create_SVM_model("radial", c_param = 0.005, gamma_param=1.8)

            elif (word_vector_type.lower() == "word2vec"):
                wordVec_pipeline = create_SVM_model("radial", c_param = 0.08, gamma_param=0.5)

        elif (classifier.lower() =="bagging" ):

            if (word_vector_type.lower() == "glove"):

                wordVec_pipeline = Pipeline([
                    ('etree',  ExtraTreesClassifier(
                                                   bootstrap=True, class_weight=None, criterion='gini',
                                                   max_depth=80, max_features=4, max_leaf_nodes=None,
                                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                                   min_samples_leaf=3, min_samples_split=12,
                                                   min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                                                   oob_score=False, random_state=None, verbose=0, warm_start=False
                                                   )
                    )
                ])
            elif (word_vector_type.lower() == "word2vec"):

                wordVec_pipeline = Pipeline([
                    ('etree',  ExtraTreesClassifier(
                           bootstrap=True,
                           max_depth=90, max_features=2,
                           min_samples_leaf=3, min_samples_split=8,
                           n_estimators=80
                           )
                    )
                ])
        elif (classifier.lower() =="boosting" ):

            if (word_vector_type.lower() == "glove"):

                wordVec_pipeline = Pipeline([
                    ('boost_glove',   xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                   colsample_bynode=1,
                                                   colsample_bytree=0.803144269805444,
                                                   gamma=4.003568280720304,
                                                   learning_rate=0.2492770588539289,
                                                   max_delta_step=0,
                                                   max_depth=2,
                                                   min_child_weight=4.463296743061923,
                                                   missing=None, n_estimators=128, n_jobs=1, nthread=None,
                                                   objective='binary:logistic', random_state=0, reg_alpha=0,
                                                   reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
                                                   subsample=0.8072204957465571, verbosity=1
                                                   )
                    )
                ])


            elif (word_vector_type.lower() == "word2vec"):

                wordVec_pipeline = Pipeline([
                    ('boost_word2vec',   xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                   colsample_bynode=1,
                                                   colsample_bytree=0.803144269805444,
                                                   gamma=4.003568280720304,
                                                   learning_rate=0.2492770588539289,
                                                   max_delta_step=0,
                                                   max_depth=2,
                                                   min_child_weight=4.463296743061923,
                                                   missing=None, n_estimators=128, n_jobs=1, nthread=None,
                                                   objective='binary:logistic', random_state=0, reg_alpha=0,
                                                   reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
                                                   subsample=0.8072204957465571, verbosity=1
                                                   )
                    )
                ])





        wordVec_pipeline.fit(feature_train, label_train)

        with open(classifier_path, 'wb') as write_file:
            dill.dump(wordVec_pipeline, write_file)

    else:
        with open(classifier_path, 'rb') as read_file:
            wordVec_pipeline = dill.load(read_file)


    wordVec_predictions = wordVec_pipeline.predict(feature_test)
    wordVec_probabilities = wordVec_pipeline.predict_proba(feature_test)

    return wordVec_predictions, wordVec_pipeline, wordVec_probabilities




def get_classifier_predictions_probabilities(feature_train, feature_test, label_train, **kwargs):

    recreate_models = kwargs['recreate_models']
    word_vector_dim = kwargs['word_vector_dim']


    # deleting all of the classifiers so can recreate new ones
    if recreate_models== True and os.path.exists(classifier_dir):
        for root, dirs, files in os.walk(classifier_dir):
            for file in files:
                os.remove(os.path.join(root,file))



    predict_wordVec_bayes_glove, model_bayes_glove, prob_wordVec_bayes_glove = get_classifier(
                                                                                              feature_train,
                                                                                              feature_test,
                                                                                              label_train,
                                                                                              classifier="bayes",
                                                                                              word_Vector = "glove",
                                                                                              word_vector_dim = word_vector_dim
                                                                                              )

    predict_wordVec_svm_glove,  model_svm_glove, prob_wordVec_svm_glove = get_classifier(
                                                                                         feature_train,
                                                                                         feature_test,
                                                                                         label_train,
                                                                                         classifier="svm",
                                                                                         word_Vector = "glove",
                                                                                         word_vector_dim = word_vector_dim
                                                                                         )

    predict_wordVec_randForest_glove, model_randForest_glove, prob_wordVec_randForest_glove  = get_classifier(
                                                                                                              feature_train,
                                                                                                              feature_test,
                                                                                                              label_train,
                                                                                                              classifier="tree",
                                                                                                              word_Vector= "glove",
                                                                                                              word_vector_dim = word_vector_dim
                                                                                                              )


    # predict_wordVec_svm_w2v, model_svm_w2v, prob_wordVec_svm_w2v  = get_classifier(feature_train, label_train, feature_test,
    #                                                                         glove_reg, word2vec,
    #                                                                         classifier="svm", vectors = "word2vec")
    #
    # predict_wordVec_randForest_w2v, model_randForest_w2v, prob_wordVec_randForest_w2v = get_classifier(feature_train, label_train,
    #                                                                                     feature_test, glove_reg, word2vec ,
    #                                                                                     classifier="tree", vectors = "word2vec")

    model_piplines = [
        {"bayes_glove": model_bayes_glove },
        {"svm_glove" :model_svm_glove},
        {"randForest_glove" : model_randForest_glove}
        # ,{"svm_w2v" : model_svm_w2v},
        # {"randForest_w2v": model_randForest_w2v}


    ]

    model_predictions = [
            {"bayes_glove":predict_wordVec_bayes_glove },
            {"svm_glove" :predict_wordVec_svm_glove},
            {"randForest_glove" : predict_wordVec_randForest_glove}
            # ,{"svm_w2v" : predict_wordVec_svm_w2v},
            # {"randForest_w2v": predict_wordVec_randForest_w2v}
    ]

    model_probabilities = [
        {"bayes_glove":prob_wordVec_bayes_glove },
        {"svm_glove" :prob_wordVec_svm_glove},
        {"randForest_glove" : prob_wordVec_randForest_glove}
        # ,{"svm_w2v" : prob_wordVec_svm_w2v},
        # {"randForest_w2v": prob_wordVec_randForest_w2v}
    ]


    return model_piplines, model_predictions, model_probabilities
