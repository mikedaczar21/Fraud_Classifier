import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.utils import class_weight
from collections import Counter
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import shap
import category_encoders as ce



from text_processing_cleanup import text_Processing, text_Processing_GloVe
from data_feature_functions import create_indicator_matrix, get_Fraud_Dataset, get_training_testing_data, print_class_report_confusion_matrix, get_combined_feature, export_predictions
from predict_CNN_1layer import get_cnn_pred_prob, evaluate_cnn
from ensemble_classifier import train_bagging_ensemble, train_boosting_ensemble, perfrom_GridSearch, perfrom_RandomSearch
from word_vector_functions import  load_word2vec, get_sentence_feature_values, read_glove_file, get_sentence_embeddings
# from get_classifier_models import get_classifier_predictions_probabilities
from data_visualization import plot_3d, get_feature_importance,  get_feature_importance_pred


test_val_size = 40 # out of 100 percent for train-test split
test_val_percent = float(test_val_size / 100.00)
rand_state = 47 # random state of train/test split
embedding_dim  = 100


text_type = "Clean_2char_NoNums"
glove_sum = "VecAvg"
recreate_full_xgb = True
fraud_type= 'refferal' # fraud_type= 'acceptance' or 'refferal', refferal has all 79k claims

if __name__ == '__main__':


    fraud_data = get_Fraud_Dataset(recreate_features = False, fraud_type= fraud_type, correct_spelling = False)

    text = fraud_data['Fraud_Text'] # features or inputs into model
    labels = fraud_data['Fraud_Label'] # labels

    labels_onehot = create_indicator_matrix(fraud_data['Fraud_Label'], check_index = True)

    cleaned_text = text.apply(text_Processing, numbers=False)

    text_word_vec = text.apply(text_Processing_GloVe, numbers=False)

    clean_list = cleaned_text.tolist()

    clean_wordVec_list = text_word_vec.tolist()


    glove_dict = read_glove_file(data_type = "refferal", dimension = embedding_dim,  vocab_size = "15k", glove_type = text_type)


    new_feat_avg = fraud_data.loc[:, 'Loss_PolicyEff':'Claim_Loss',] # Datediff features + fraud text
    new_feature_datediff = fraud_data.loc[:, 'Loss_PolicyEff':'Claim_Loss'] # Datediff features


    new_feature_labels = fraud_data['Fraud_Label']
    fraud_data['Main Cause'].fillna("", inplace=True)
    fraud_data['Sub Cause'].fillna("", inplace=True)
    fraud_data['Longitude'].fillna(0, inplace=True)
    fraud_data['Latitude'].fillna(0, inplace=True)

    fraud_data['New_Main'] = fraud_data['Main Cause']
    fraud_data['New_Sub'] = fraud_data['Sub Cause']
    fraud_data['Sub_Business'] = fraud_data['Subline_Business']
    fraud_data['Cause'] = fraud_data['Main Cause'] + " - " + fraud_data['Sub Cause']
    fraud_data['MainSubCause'] = fraud_data['Main Cause'] + " - " + fraud_data['Sub Cause']


    # Encoding main/sub cause columns
    encoder = ce.leave_one_out.LeaveOneOutEncoder(cols = ['Cause',  'Sub_Business', 'New_Main', 'New_Sub'])

    # encoder = ce.BackwardDifferenceEncoder(cols = ['New_Main', 'New_Sub'])

    fraud_data = encoder.fit_transform(X=fraud_data, y=fraud_data['Fraud_Label'])

    feature_cols = [

                   # 'Loss_PolicyEff',
                   'Loss_PolicyExp',
                    'Claim_PolicyEff',
                    'Claim_Loss',
                    'Longitude',
                    'Latitude',
                    'Sub_Business',
                    'Cause']

    new_feat_avg = fraud_data[feature_cols]

    loss_cleaned = fraud_data['Fraud_Text'].apply(text_Processing, numbers=False)


    sentence_embedding_avg = np.array( [get_sentence_feature_values(sentence = words, embedding = glove_dict, embedding_dim = 200) for words in loss_cleaned ])

    sent_orig_sum = sentence_embedding_avg.sum(axis = 1) # getting sum of sentece embeddings along row

    sent_embed_loss = get_sentence_embeddings(text = loss_cleaned, embedding = glove_dict, embedding_size = embedding_dim)

    sent_sum_loss = [sent.sum() for sent in sent_embed_loss]

    sent_avg_loss = [np.average(sent) for sent in sent_embed_loss]



    features_orig = [
                       # 'Loss_PolicyEff',
                       'Loss_PolicyExp',
                        'Claim_PolicyEff',
                        'Claim_Loss',
                        'Longitude',
                        'Latitude',
                        'Subline_Business',
                        'MainSubCause',
                        'Loss_Descrip_NoAddition'
                        ]
    new_feat_orig = fraud_data[features_orig]



    new_feat_avg['Loss_Descrip'] = sent_avg_loss
    # new_feat_avg.insert(-1, 'Loss_Descrip', sent_avg_loss, True)


    fraud_data['Loss_Glove_Avg'] = sent_avg_loss
    # fraud_data.insert(-1, 'Loss_Glove_Avg', sent_avg_loss, True)



    #### ======= CREATING DATAFRAME WITH 200 DIMENSIONAL GLOVE AVERAGE =======

    sent_embed_df = pd.DataFrame(sent_embed_loss, columns = ['Loss_Descrip_Feat_' + str(feat) for feat in range( len(sent_embed_loss[0]) ) ] )

    sent_embed_df.set_index(new_feature_datediff.index, inplace=True)


    new_feature_expand = pd.concat([new_feature_datediff, fraud_data['Cause'],  fraud_data['Longitude'], fraud_data['Latitude']], axis=1)

    new_feature_expand = new_feature_expand.join([sent_embed_df ], how="inner")



    ####  ====== TRAIN-TEST SPLIT   ======

    X_train_avg, X_test_val_avg, y_train_avg, y_test_val_avg = train_test_split(new_feat_avg, new_feature_labels, test_size = test_val_percent, random_state = rand_state)

    X_train_expand, X_test_val_expand, y_train_expand, y_test_val_expand = train_test_split(new_feature_expand, new_feature_labels, test_size = test_val_percent, random_state = rand_state)

    X_train_orig, X_test_val_orig, y_train_orig, y_test_val_orig = train_test_split(new_feat_orig, new_feature_labels, test_size = test_val_percent, random_state = rand_state)


    #### ===== ADDING SYNTHETIC FRAUD SAMPLES =====
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train_avg==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_avg==0)))

    over_sampling = SMOTE(random_state=777, k_neighbors=5)
    X_train_oversamp, y_train_oversamp = over_sampling.fit_sample(X_train_avg, y_train_avg)


    print('After OverSampling, the shape of train_X: {}'.format(X_train_oversamp.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_oversamp.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_oversamp==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_oversamp==0)))



    X_val_avg, X_test_avg, y_val_avg, y_test_avg = train_test_split(X_test_val_avg, y_test_val_avg, test_size = 0.5, random_state = 23)

    X_val_expand, X_test_expand, y_val_expand, y_test_expand = train_test_split(X_test_val_expand, y_test_val_expand, test_size =  0.5, random_state = 23)

    X_val_orig, X_test_orig, y_val_orig, y_test_val_orig = train_test_split(X_test_val_orig, y_test_val_orig, test_size =  0.5, random_state = 23)


    perfrom_RandomSearch(
                      X_train = X_train_avg, X_test = X_val_avg,
                      y_train = y_train_avg, y_test = y_val_avg,
                      ensemble_type = "boost"
                      )


    #### ====== TRAINING XGBOOST MODEL AND EXPORTING PREDICTIONS TO EXCEL FILE  ======

    boost_pred, boost_prob, boost_model, boost_pred_path = train_boosting_ensemble(X_train_avg, X_test_avg, y_train_avg, y_test_avg,
                                                                                  boosting_type= "xgboost_{}_FullData".format( 'VecAvg'),
                                                                                  recreate_model= recreate_full_xgb, model_type = 'imbalanced')

    boost_pred_expand, boost_prob_expand, boost_model_expand, boost_pred_path_expand = train_boosting_ensemble(X_train_expand, X_test_expand, y_train_expand, y_test_expand,
                                                                                  boosting_type= "xgboost_Glove{}_FullData".format( 'Expand'),
                                                                                  recreate_model= recreate_full_xgb, model_type = 'imbalanced')

    boost_pred_oversamp, boost_prob_oversamp, boost_model_oversamp, boost_pred_path_oversamp = train_boosting_ensemble(X_train_oversamp, X_test_avg, y_train_oversamp, y_test_avg,
                                                                                  boosting_type= "xgboost_Glove{}_FullData".format( 'Oversamp'),
                                                                                  recreate_model= recreate_full_xgb, model_type = 'balanced')


    # class_xgb =  print_class_report_confusion_matrix(y_test_avg, boost_pred, "XGBoost", "Glove Sum Full Testing Eval")
    #
    # class_xgb_expand =  print_class_report_confusion_matrix(y_test_expand, boost_pred_expand, "XGBoost", "Glove Expand Full Testing Eval")
    #
    # class_xgb_oversamp =  print_class_report_confusion_matrix(y_test_avg, boost_pred_oversamp, "XGBoost", "Glove Synthetic Oversampled Testing Eval")

    expand_out =  fraud_data.join(sent_embed_df, how="inner", lsuffix='_left')


    # perfrom_RandomSearch(
    #                   X_train = X_train_avg, X_test = X_val_avg,
    #                   y_train = y_train_avg, y_test = y_val_avg,
    #                   ensemble_type = "boost"
    #                   )
    #
    # perfrom_RandomSearch(
    #                   X_train = X_train_expand, X_test = X_val_expand,
    #                   y_train = y_train_expand, y_test = y_val_expand,
    #                   ensemble_type = "boost"
    #                   )



    boost_out = export_predictions(
                      fraud_data,
                      boost_prob,
                      boost_pred,
                      actual= y_test_avg,
                      recreate_ProbPreds = True,
                      pred_path = boost_pred_path,
                      file_name = 'XGBoost_{}_{}{}_FullData'.format(fraud_type, text_type, glove_sum),
                      model_type ='XGBoost')
    # #### ====== PLOTTING MODEL OUTPUTS =======
    # plot_data_points = { 'x':boost_out['Actual Label'] , 'y':boost_out['XGBoost_Predictions'] , 'z': boost_out['XGBoost_Confid_Prob'] * 100.00}
    # plot_3d(model_output= boost_out,  data_points = plot_data_points, fig_type = "Boost_FullData_Actual_Pred_Prob", model_type = "XGBoost",  z_label = "Probability Fraud (%)")
    #
    #
    # plot_data_points = { 'x':boost_out['Actual Label'] , 'y':boost_out['XGBoost_Predictions'] , 'z': boost_out['Claim_Loss'] }
    # plot_3d(model_output= boost_out,  data_points = plot_data_points, fig_type = "Boost_Full_Actual_Pred_ClaimLoss", model_type = "XGBoost", z_label = "Days Between Policy Loss - Claim")
    #
    #
    # # get_feature_importance(model = boost_model,
    # #                        features = X_test_avg,
    # #                        feature_names = list(X_test_avg.columns),
    # #                        orig_feat = X_test_orig,
    # #                        model_type = "XGBoost",
    # #                        plot = 'decision')
    #
    #
    #
    # get_feature_importance_pred(
    #                         model = boost_model,
    #                         features = new_feat_avg,
    #                         model_out = boost_out,
    #                         feature_names = list(X_test_avg.columns),
    #                         orig_feat = X_test_orig,
    #                         model_type = "XGBoost",
    #                         plot = 'decision')
    #
    #
    # cnn_pred, cnn_prob, cnn_out_path = get_cnn_pred_prob(cleaned_text = cleaned_text, labels = labels_onehot,
    #                                                      testing = True,
    #                                                      prob_type='{}test'.format(test_size),
    #                                                      test_size = test_size,
    #                                                      rand_state = rand_state,
    #                                                      glove_type = text_type,
    #                                                      embedding_dim = embedding_dim,
    #                                                      recreate_prob = False)
    #
    #
    # # evaluate_cnn(cleaned_text = cleaned_text, labels = labels_onehot)
    #
    # new_feature = get_combined_feature(
    #                                 features =  fraud_data,
    #                                 cnn_Prob = cnn_out_path,
    #                                 recreate_combined_features = True,
    #                                 feature_type = '{}test_{}_{}D'.format(test_size, text_type, embedding_dim)
    # )
    #
    #
    # # Getting rid of nan rows (policy_eff)
    # # null_dateDiff_rows = new_feature[ new_feature['Loss_PolicyEff'].isna() == True].index
    # # new_feature.drop(null_policyEff_rows, inplace = True)
    #
    # new_feature_datediff = new_feature.loc[:, 'Loss_PolicyEff':'Multi_Body_Parts_Injured'] # Datediff features
    # new_feat_avg = new_feature.loc[:, 'Loss_PolicyEff':'Multi_Body_Parts_Injured'] # Datediff features + fraud text
    # new_feature = new_feature.drop(['Unnamed: 0'], axis=1)
    #
    # cnn_features = [
    #                  # 'Loss_PolicyEff',
    #                  'Loss_PolicyExp',
    #                   'Claim_PolicyEff',
    #                   'Claim_Loss',
    #                   'New_Main',
    #                   'New_Sub',
    #                   'Longitude',
    #                   'Latitude',
    #                   'Sub_Business',
    #                   'CNN_Prob_Fraud'
    #             ]
    #
    # # new_feature_xgb = pd.concat([new_feature_datediff, new_feature['New_Main'],  new_feature['New_Sub'], new_feature['Longitude'],  new_feature['Latitude'], new_feature['CNN_Prob_Fraud']], axis=1)
    # new_feature_xgb = new_feature[cnn_features]
    #
    #
    #
    # new_feature_labels = new_feature['Fraud_Label']
    # new_feature['Main Cause'].fillna("", inplace=True)
    # new_feature['Sub Cause'].fillna("", inplace=True)
    # new_feature['Longitude'].fillna(0, inplace=True)
    # new_feature['Latitude'].fillna(0, inplace=True)
    #
    # loss_cleaned = new_feature['Loss_Descrip_NoAddition'].apply(text_Processing, numbers=False)
    #
    #
    # ## ===== GETTING SENTENCE GLOVE AVERAGES
    # sentence_embedding_avg = np.array( [get_sentence_feature_values(sentence = words, embedding = glove_dict, embedding_dim = 200) for words in loss_cleaned ])
    #
    # sent_orig_sum = sentence_embedding_avg.sum(axis = 1) # getting sum of sentece embeddings along row
    #
    # sent_embed_loss = get_sentence_embeddings(text = loss_cleaned, embedding = glove_dict, embedding_size = embedding_dim)
    #
    # sent_sum_loss = [sent.sum() for sent in sent_embed_loss]
    #
    #
    #
    #
    # new_feat_avg['Long'] = new_feature['Longitude']
    # new_feat_avg['Lat'] = new_feature['Latitude']
    #
    # features_orig = [
    #                    # 'Loss_PolicyEff',
    #                    'Loss_PolicyExp',
    #                     'Claim_PolicyEff',
    #                     'Claim_Loss',
    #                     'Main Cause',
    #                     'Sub Cause',
    #                     'Longitude',
    #                     'Latitude',
    #                     'Subline_Business',
    #                     'Loss_Description'
    #                     ]
    #
    # new_feat_orig = new_feature[features_orig]
    #
    # new_feat_avg['Glove_Avg'] = sent_sum_loss
    # new_feat_avg['Main'] = new_feature['New_Main']
    # new_feat_avg['Sub'] = new_feature['New_Sub']
    #
    #
    # new_feature['Loss_Descrip_Glove_Average'] = sent_sum_loss
    # # new_feature.insert(-1, 'Loss_Descrip_Glove_Average', sent_sum_loss, True)
    #
    # sent_embed_df = pd.DataFrame(sent_embed_loss, columns = ['Loss_Descrip_Feat_' + str(feat) for feat in range( len(sent_embed_loss[0]) ) ] )
    #
    # sent_embed_df.set_index(new_feature_datediff.index, inplace=True)
    #
    # expand_features = [
    #                  # 'Loss_PolicyEff',
    #                  'Loss_PolicyExp',
    #                   'Claim_PolicyEff',
    #                   'Claim_Loss',
    #                   'New_Main',
    #                   'New_Sub',
    #                   'Longitude',
    #                   'Latitude',
    #                   'Sub_Business',
    #             ]
    #
    # new_feature_expand = new_feature[expand_features]
    #
    #
    # new_feature_expand = new_feature_datediff.join([ sent_embed_df ], how="inner")
    #
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(new_feat_avg , new_feature_labels, test_size = 0.38, random_state = 121)
    #
    #
    # X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(new_feat_orig, new_feature_labels, test_size = 0.38, random_state = 121)
    #
    # X_train_expand, X_test_expand, y_train_expand, y_test_expand = train_test_split(new_feature_expand, new_feature_labels, test_size = 0.38, random_state = 121)
    #
    # X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(new_feature_xgb, new_feature_labels, test_size = 0.38, random_state = 121)
    #
    #
    #
    #
    # word2vec =  load_word2vec(text = clean_wordVec_list, dimension = 200, text_type =  text_type , clear_w2v = True)
    #
    #
    #
    # # print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    # # print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    # #
    # # over_sampling = SMOTE(random_state=777, k_neighbors=1)
    # # X_train_oversamp, y_train_oversamp = over_sampling.fit_sample(X_train, y_train)
    # #
    # #
    # # print('After OverSampling, the shape of train_X: {}'.format(X_train_oversamp.shape))
    # # print('After OverSampling, the shape of train_y: {} \n'.format(y_train_oversamp.shape))
    # #
    # # print("After OverSampling, counts of label '1': {}".format(sum(y_train_oversamp==1)))
    # # print("After OverSampling, counts of label '0': {}".format(sum(y_train_oversamp==0)))
    #
    # # colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in  y_train]
    # # kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
    # # plt.scatter(X_train.loc[:, 'Loss_PolicyEff':'claim_PolicyEff'], X_train.loc[:, 'CNN_Prob_Fraud'], c=colors, **kwarg_params)
    # # sns.despine()
    # # plt.suptitle(" Data After SMOTE OverSampling")
    #
    # boost_pred, boost_prob, boost_model, boost_pred_path = train_boosting_ensemble(X_train, X_test, y_train, y_test,
    #                                                                               boosting_type= "xgboost_{}test_GloveSum".format(test_size),
    #                                                                               recreate_model= True ,model_type = 'imbalanced')
    #
    # class_xgb =  print_class_report_confusion_matrix(y_test, boost_pred, "XGBoost", "Glove Vectors Sum")
    #
    #
    # boost_pred_cnn, boost_prob_cnn, boost_model_cnn, boost_pred_path_cnn = train_boosting_ensemble(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn,
    #                                                                               boosting_type= "xgboost_{}test_CNN".format(test_size),
    #                                                                               recreate_model= True, model_type = 'imbalanced')
    #
    # class_xgb_cnn =  print_class_report_confusion_matrix(y_test_cnn, boost_pred_cnn, "XGBoost", "Glove CNN Prob")
    #
    # boost_pred_expand, boost_prob_expand, boost_model_expand, boost_pred_path_expand = train_boosting_ensemble(X_train_expand, X_test_expand, y_train_expand, y_test_expand,
    #                                                                               boosting_type= "xgboost_{}test_GloveExpand".format(test_size),
    #                                                                               recreate_model= True, model_type = 'imbalanced')
    #
    # class_xgb_expand =  print_class_report_confusion_matrix(y_test_cnn, boost_pred_cnn, "XGBoost", "Glove Vectors Expand")
    #
    #
    # bagging_pred, bagging_prob, bagging_model, bagging_pred_path = train_bagging_ensemble(X_train, X_test, y_train, y_test,
    #                                                                                       bagging_type= "etree_{}test_GloveSum".format(test_size),
    #                                                                                       recreate_model=  True)
    #
    # class_etree = print_class_report_confusion_matrix(y_test, bagging_pred, "Etree", "Glove Vectors")
    #


    # model_pipelines, model_predictions, model_probabilities = get_classifier_predictions_probabilities(
    #                                                                                                    feature_train = X_train,
    #                                                                                                    feature_test =  X_test,
    #                                                                                                    label_train = y_train,
    #                                                                                                    word_vector_dim = embedding_dim,
    #                                                                                                    recreate_models = False
    #                                                                                                    )
    #
    #
    #
    #
    # class_bayes_glove = print_class_report_confusion_matrix(y_test , model_predictions[0]['bayes_glove'], "Multinominal Bayes", "Glove Vectors")

    # class_svm_glove = print_class_report_confusion_matrix(y_test , model_predictions[1]['svm_glove'], "SVM Weighted", "Glove Vectors")



    # boost_out = export_predictions(
    #                   new_feature,
    #                   boost_prob,
    #                   boost_pred,
    #                   actual= y_test,
    #                   recreate_ProbPreds = True,
    #                   pred_path = boost_pred_path,
    #                   file_name = 'XGBoost_Output_{}test_{}VecSum'.format(test_size, text_type),
    #                   model_type ='XGBoost')
    #
    # boost_out_cnn = export_predictions(
    #                   new_feature,
    #                   boost_prob_cnn,
    #                   boost_pred_cnn,
    #                   actual= y_test_cnn,
    #                   recreate_ProbPreds = True,
    #                   pred_path = boost_pred_path,
    #                   file_name = 'XGBoost_Output_{}test_{}CNN_Prob'.format(test_size, text_type),
    #                   model_type ='XGBoost')
    #
    # bag_out = export_predictions(
    #                   new_feature,
    #                   bagging_prob,
    #                   bagging_pred ,
    #                   actual= y_test,
    #                   recreate_ProbPreds = True,
    #                   pred_path = bagging_pred_path,
    #                   file_name = 'RandomForest_Output_{}test_{}VecSum'.format(test_size, text_type),
    #                   model_type = 'RandForest')



    # get_feature_importance(model = boost_model_cnn, features = X_test_cnn, feature_names = list(X_test_cnn.columns), orig_feat = X_test_orig, model_type = 'xgb', plot= 'decision')
    #
    # # plot_data_points = { 'x':boost_out['Actual Label'] , 'y':boost_out['XGBoost_Predictions'] , 'z': boost_out['XGBoost_Confid_Prob'] * 100.00}
    # plot_3d(model_output= boost_out,  data_points = plot_data_points, fig_type = "Boost_Small_Actual_Pred_Prob", model_type = "XGBoost", z_label = "Probability Fraud %")


    # plot_data_points = { 'x':bag_out['Loss_PolicyEff'] , 'y':bag_out['Claim_PolicyEff'] , 'z': bag_out['RandForest_Prob_Fraud'] * 100.00}
    # plot_3d(model_output= bag_out,  data_points = plot_data_points, fig_type = "Bag_Fraud_Prob_LossPolxClaimPol", model_type= "RandForest", y_label = "Days Between Policy Start - Claim")

    # new_feat_scaled =  StandardScaler().fit_transform(new_feat_avg)
    #
    # X_train, X_test, y_train, y_test = train_test_split(new_feature_scaled, new_feature_labels, test_size = 0.38, random_state = 47)
    #
    # boost_pred_scaled, boost_prob_scaled, boost_model_scaled, boost_pred_path_scaled = train_boosting_ensemble(X_train, X_test, y_train, y_test,
    #                                                                               boosting_type= "xgboost_{}test_Scaled".format(test_size),
    #                                                                               recreate_model= True)
    #
    #
    # plot_decision_regions(X =X_test , y = y_test , classifier = boost_model_scaled, test_idx=None, resolution=0.02)

    # all_models_prob_Y = [
    #
    #         ("bayes_glove",  np.average(prob_bayes_glove[0][1] )),
    #         ("svm_glove", np.average(prob_svm_glove[0][1] )),
    #
    #
    # ]
    #
    # all_models_prob_N = [
    #
    #         ("bayes_glove",np.average(  prob_bayes_glove[0][0]) ),
    #         ("svm_glove",  np.average( prob_svm_glove[0][0] ) ),
    #
    #
    # ]
    #
    # unsorted_scores = [(name, score) for name, score in all_models_prob_N]
    # scores = sorted(unsorted_scores, key=lambda x: -x[1])
    # print(tabulate(scores, floatfmt=".4f", headers=("model", 'probability - N')))
    # plt.figure(figsize=(15, 6))
    # sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
    #
    # unsorted_scores_crossVal = [(name, score) for name, score in all_models_prob_Y]
    # scores_cv = sorted(unsorted_scores_crossVal , key=lambda x: -x[1])
    # print(tabulate(scores_cv, floatfmt=".4f", headers=("model", 'probability - Y')))
    # plt.figure(figsize=(15, 6))
    # sns.barplot(x=[name for name, _ in scores_cv], y=[score for _, score in scores_cv])



    # perfrom_RandomSearch(
    #                   X_train = X_train_cnn, X_test = X_test_cnn,
    #                   y_train = y_train_cnn, y_test = y_test_cnn,
    #                   ensemble_type = "boost"
    #                   )
