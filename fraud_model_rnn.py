import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import shap

from text_processing_cleanup import text_Processing
from data_feature_functions import create_indicator_matrix, get_Fraud_Dataset, print_class_report_confusion_matrix
from word_vector_functions import  read_glove_file
from data_visualization import  plot_word_shap
from Text_RNN import KerasTextClassifier, prepare_explanation_words


test_size = 38 # out of 100 percent for train-test split
test_percentage = float(test_size / 100.00)
rand_state = 47 # random state of train/test split
embedding_dim  = 200
text_type = "Clean_Nums_2char"


glove_sum = "VecAvg"
recreate_full_xgb = False
fraud_type= 'acceptance'

if __name__ == '__main__':


    fraud_data = get_Fraud_Dataset(recreate_features = False, fraud_type= fraud_type, correct_spelling = False)


    text = fraud_data['Fraud_Text'] # features or inputs into model
    labels = fraud_data['Fraud_Label'] # labels

    labels_onehot = create_indicator_matrix(fraud_data['Fraud_Label'], check_index = True)

    cleaned_text = text.apply(text_Processing, numbers=False)


    X_train, X_test, y_train, y_test = train_test_split(cleaned_text, labels, test_size = test_percentage , random_state = rand_state)
    max_doc_length = max([len(words.split(",")) for words in cleaned_text]) # 200 otherwise

    rnn_text = KerasTextClassifier(
        max_word_input= max_doc_length, word_cnt=4421, word_embedding_dimension=200,
        labels=list(set(y_train)), batch_size=1024, epoch=12, validation_split=0.1)

    rnn_text.fit(X_train, y_train, retrain_RNN = False,  model_type = "Glove_NoNumbers_MoreEpochs")

    # print('Num of Word: %d' % len(rnn_text.tokenizer.word_index))


    encoded_x_train = rnn_text.transform(X_train)
    encoded_x_test = rnn_text.transform(X_test)

    rnn_pred = rnn_text.predict(X_test)
    class_rnn =  print_class_report_confusion_matrix(y_test, rnn_pred, "RNN", "Glove Vectors")


    attrib_data = encoded_x_train[0:4000]
    explainer = shap.DeepExplainer(rnn_text.model, attrib_data)
    num_explanations = 400
    start = 0
    testing_set = encoded_x_test[start: start + num_explanations]
    shap_vals = explainer.shap_values(testing_set)

    x_test_words = prepare_explanation_words(rnn_text, encoded_x_test, start, num_explanations)


    # shap.force_plot(explainer.expected_value[0], shap_vals[0][0], x_test_words[0],  matplotlib=True)

    # ===== DECISION PLOTS AND FORCE PLOTS
    # for row in range(num_explanations):
    #     shap.force_plot(explainer.expected_value[1], shap_vals[1][row], x_test_words[row],  matplotlib=True)
    #     shap.decision_plot(explainer.expected_value[1], shap_vals[1][row], x_test_words[row])
    #


    # words = rnn_text.tokenizer.word_index
    # word_lookup = list()
    # for i in words.keys():
    #   word_lookup.append(i)
    #
    # word_lookup = [''] + word_lookup
    #
    # word_dict = rnn_text.word2index
    # sorted_word = sorted(word_dict.items(), key = lambda kv: kv[1])

    decode_x_test = rnn_text._decode_feature(testing_set)
    all_words = set()
    indices = set()
    for sent in list(decode_x_test):
        split_sent = sent.split(', ')
        for word in split_sent:
            all_words.add(word)

    for sent in encoded_x_test[:num_explanations]:
        for word in sent:
            if word > 0:
                indices.add(word)


    index_shap_accept = {}

    for row, sent in enumerate(testing_set):

        for col, word in enumerate(sent):
            if word > 0:

                if word not in index_shap_accept.keys():
                    index_shap_accept[word] = [ shap_vals[1][row][col] ]
                else:
                    index_shap_accept[word].append(shap_vals[1][row][col])


    index_shap_reject = {}
    for row, sent in enumerate(testing_set):

        for col, word in enumerate(sent):
            if word > 0:

                if word not in index_shap_reject.keys():
                    index_shap_reject[word] = [ shap_vals[0][row][col] ]
                else:
                    index_shap_reject[word].append(shap_vals[0][row][col])

    word_shap_accept = {}
    max_impact = {}
    max_fraud = {}

    for index, shap_vals in index_shap_accept.items():

        if index in rnn_text.index2word:
            current_word = rnn_text.index2word[index]
            word_shap_accept[current_word] = shap_vals
            max_fraud[current_word] = sum(shap_vals)
            max_impact[current_word] = np.average([abs(elem) for elem in shap_vals])


    max_accept = dict([(key,elem) for key, elem in max_fraud.items() if elem > 0])

    max_reject = dict([(key, abs(elem)) for key, elem in max_fraud.items() if elem <= 0])

    impact_sorted = dict([(key,elem) for key, elem in max_impact.items() if elem > 0])

    max_accept_50 = dict( list(max_accept.items() )[:50] )

    max_reject_50 = dict( list(max_reject.items() )[:50] )

    max_impact_50 = dict( list(max_accept.items() )[:50] )

    plot_word_shap(word_shap_dict = word_shap_accept, words_to_plot = max_accept_50, plot_type = 'Fraud_Words_Full')

    plot_word_shap(word_shap_dict = word_shap_accept,words_to_plot = max_reject_50, plot_type = 'Non_Fruad Words_Full')

    plot_word_shap(word_shap_dict = word_shap_accept,words_to_plot = max_impact_50,  plot_type = 'Max_Impact_Full')

    # word_shap_reject = {}
    # max_non_fraud = {}
    #
    # for index, shap_vals in index_shap_reject.items():
    #
    #     if index in rnn_text.index2word:
    #         current_word = rnn_text.index2word[index]
    #         word_shap_reject[current_word] = shap_vals
    #         max_non_fraud[current_word] = sum(shap_vals)



    # shap.summary_plot(shap_vals[1], features = list(indices) ,feature_names=list(all_words), class_names=['Rejected', 'Accepted'])

    # shap.summary_plot(shap_vals[0], features = list(indices) ,feature_names=list(all_words), class_names=['Rejected', 'Accepted'])

    # cnn_pred, cnn_prob, cnn_out_path = get_cnn_pred_prob(cleaned_text = cleaned_text, labels = labels_onehot,
    #                                                      testing = True,
    #                                                      prob_type='{}test'.format(test_size),
    #                                                      test_size = test_size,
    #                                                      rand_state = rand_state,
    #                                                      glove_type = text_type,
    #                                                      embedding_dim = embedding_dim,
    #                                                      recreate_prob = True)
    #
    #
    # # evaluate_cnn(cleaned_text = cleaned_text, labels = labels_onehot)
    #
    # new_feature = get_combined_feature(
    #                                 features =  fraud_data,
    #                                 cnn_Prob = cnn_out_path,
    #                                 recreate_combined_features = False,
    #                                 feature_type = '{}test_{}_{}D'.format(test_size, text_type, embedding_dim)
    # )
    #
    #
    # # Getting rid of nan rows (policy_eff)
    # # null_dateDiff_rows = new_feature[ new_feature['Loss_PolicyEff_Diff'].isna() == True].index
    # # new_feature.drop(null_policyEff_rows, inplace = True)
    #
    # new_feature_datediff = new_feature.loc[:, 'Loss_PolicyEff_Diff':'Multi_Body_Parts_Injured'] # Datediff features
    # new_feature_loss = new_feature.loc[:, 'Loss_PolicyEff_Diff':'Multi_Body_Parts_Injured'] # Datediff features + fraud text
    #
    # new_feature_xgb = pd.concat([new_feature_datediff, new_feature['CNN_Prob_Fraud']], axis=1)
    #
    # new_feature_labels = new_feature['Fraud_Label']
    # new_feature['Main Cause'].fillna("", inplace=True)
    # new_feature['Sub Cause'].fillna("", inplace=True)
    # new_feature['New_City'].fillna("", inplace=True)
    # new_feature['Longitude'].fillna(0, inplace=True)
    # new_feature['Latitude'].fillna(0, inplace=True)
    #
    # loss_cleaned = new_feature['Loss_Descrip_NoAddition'].apply(text_Processing, numbers=False)
    # main_cleaned = new_feature['Main Cause'].apply(text_Processing, numbers=False)
    # sub_cleaned = new_feature['Sub Cause'].apply(text_Processing, numbers=False)
    #
    #
    # sentence_embedding_avg = np.array( [get_sentence_feature_values(sentence = words, embedding = glove_dict, embedding_dim = 200) for words in loss_cleaned ])
    #
    # sent_orig_sum = sentence_embedding_avg.sum(axis = 1) # getting sum of sentece embeddings along row
    #
    # sent_embed_loss = get_sentence_embeddings(text = loss_cleaned, embedding = glove_dict, embedding_size = embedding_dim)
    #
    # sent_sum_loss = [sent.sum() for sent in sent_embed_loss]
    #
    # sent_embed_main = get_sentence_embeddings(text = main_cleaned, embedding = glove_dict, embedding_size = embedding_dim)
    #
    # sent_sum_main = [sent.sum() for sent in sent_embed_main]
    #
    # sent_embed_sub = get_sentence_embeddings(text = sub_cleaned, embedding = glove_dict, embedding_size = embedding_dim)
    #
    # sent_sum_sub = [sent.sum() for sent in sent_embed_main]
    #
    #
    # new_feature_loss['Longitude'] = new_feature['Longitude']
    # new_feature_loss['Latitude'] = new_feature['Latitude']
    # new_feature_loss['Loss_Descrip_Glove_Average'] = sent_sum_loss
    # new_feature_loss['Main_Cause_Glove_Average'] = sent_sum_main
    # new_feature_loss['Sub_Cause_Glove_Average'] = sent_sum_sub
    #
    #
    # new_feature['Loss_Descrip_Glove_Average'] = sent_sum_loss
    # new_feature['Main_Cause_Glove_Average'] = sent_sum_main
    # new_feature['Sub_Cause_Glove_Average'] = sent_sum_sub
    #
    # sent_embed_df = pd.DataFrame(sent_embed_loss, columns = ['Loss_Descrip_Feat_' + str(feat) for feat in range( len(sent_embed_loss[0]) ) ] )
    #
    # sent_embed_df.set_index(new_feature_datediff.index, inplace=True)
    #
    # main_embed_df = pd.DataFrame(sent_embed_main, columns = ['Main_Cause_Feat_' + str(feat) for feat in range( len(sent_embed_main[0]) ) ] )
    #
    # main_embed_df.set_index(new_feature_datediff.index, inplace=True)
    #
    # sub_embed_df = pd.DataFrame(sent_embed_sub, columns = ['Sub_Cause_Feat_' + str(feat) for feat in range( len(sent_embed_sub[0]) ) ] )
    #
    # sub_embed_df.set_index(new_feature_datediff.index, inplace=True)
    #
    # new_feature_expand = new_feature_datediff.join([sent_embed_df, main_embed_df, sub_embed_df], how="inner")
    #
    #
    #
    #
    # X_train, X_test, y_train, y_test = train_test_split(new_feature_loss, new_feature_labels, test_size = 0.38, random_state = 47)
    #
    # X_train_expand, X_test_expand, y_train_expand, y_test_expand = train_test_split(new_feature_expand, new_feature_labels, test_size = 0.38, random_state = 47)
    #
    # X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(new_feature_xgb, new_feature_labels, test_size = 0.38, random_state = 47)
    #
    #
    # # X_train = X_train.loc[:, 'Loss_PolicyEff_Diff': 'CNN_Prob_Fraud']
    # #
    # # X_test = X_test.loc[:, 'Loss_PolicyEff_Diff': 'CNN_Prob_Fraud']
    #
    #
    # # sent_embed_df = pd.DataFrame(sent_embed_WR, columns = ['Loss_Descrip_Feat_' + str(feat) for feat in range( len(sent_embed_WR[0]) ) ] )
    # #
    # # sent_embed_df.set_index(X_train.index, inplace=True)
    # #
    # # train_embed_merge = pd.concat([X_train, sent_embed_df], axis = 1)
    #
    # # X_train, X_test, y_train, y_test = get_training_testing_data(features = new_feature_xgb,
    # #                                                             labels = new_feature_labels,
    # #                                                             test_percentage = float(test_size / 100.00),
    # #                                                             rand_state = rand_state,
    # #                                                             data_type = "XGBoost_GloveAvg",
    # #                                                             recreate_train_test_data = False)
    #
    #
    #
    #
    # # class_weights = list(class_weight.compute_class_weight('balanced',
    # #                                          np.unique(y_train),
    # #                                          y_train))
    # #
    # # w_array = np.ones(y_train.shape[0], dtype = 'float')
    # # for i, val in enumerate(y_train):
    # #     w_array[i] = class_weights[val]
    #
    # # colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in  y_train]
    # # kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
    # # plt.scatter(X_train.loc[:, 'Loss_PolicyEff_Diff':'claim_PolicyEff_diff'], X_train.loc[:, 'CNN_Prob_Fraud'], c=colors, **kwarg_params)
    # # sns.despine()
    # # plt.suptitle(" Data Before OverSampling")
    #
    # word2vec =  load_word2vec(text = clean_wordVec_list, dimension = 200, text_type =  text_type , clear_w2v = False)
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
    # # plt.scatter(X_train.loc[:, 'Loss_PolicyEff_Diff':'claim_PolicyEff_diff'], X_train.loc[:, 'CNN_Prob_Fraud'], c=colors, **kwarg_params)
    # # sns.despine()
    # # plt.suptitle(" Data After SMOTE OverSampling")
    #
    # boost_pred, boost_prob, boost_model, boost_pred_path = train_boosting_ensemble(X_train, X_test, y_train, y_test,
    #                                                                               boosting_type= "xgboost_{}test_GloveSum".format(test_size),
    #                                                                               recreate_model= False)
    #
    # class_xgb =  print_class_report_confusion_matrix(y_test, boost_pred, "XGBoost", "Glove Vectors Sum")
    #
    #
    # boost_pred_cnn, boost_prob_cnn, boost_model_cnn, boost_pred_path_cnn = train_boosting_ensemble(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn,
    #                                                                               boosting_type= "xgboost_{}test_CNN".format(test_size),
    #                                                                               recreate_model= False)
    #
    # class_xgb_cnn =  print_class_report_confusion_matrix(y_test_cnn, boost_pred_cnn, "XGBoost", "Glove CNN Prob")
    #
    # boost_pred_expand, boost_prob_expand, boost_model_expand, boost_pred_path_expand = train_boosting_ensemble(X_train_expand, X_test_expand, y_train_expand, y_test_expand,
    #                                                                               boosting_type= "xgboost_{}test_GloveExpand".format(test_size),
    #                                                                               recreate_model= False)
    #
    # class_xgb_expand =  print_class_report_confusion_matrix(y_test_cnn, boost_pred_cnn, "XGBoost", "Glove Vectors Expand")
    #
    #
    # bagging_pred, bagging_prob, bagging_model, bagging_pred_path = train_bagging_ensemble(X_train, X_test, y_train, y_test,
    #                                                                                       bagging_type= "etree_{}test_GloveSum".format(test_size),
    #                                                                                       recreate_model=  False)
    #
    # class_etree = print_class_report_confusion_matrix(y_test, bagging_pred, "Etree", "Glove Vectors")
    #
    #
    #
    # # model_pipelines, model_predictions, model_probabilities = get_classifier_predictions_probabilities(
    # #                                                                                                    feature_train = X_train,
    # #                                                                                                    feature_test =  X_test,
    # #                                                                                                    label_train = y_train,
    # #                                                                                                    word_vector_dim = embedding_dim,
    # #                                                                                                    recreate_models = False
    # #                                                                                                    )
    # #
    # #
    # #
    # #
    # # class_bayes_glove = print_class_report_confusion_matrix(y_test , model_predictions[0]['bayes_glove'], "Multinominal Bayes", "Glove Vectors")
    #
    # # class_svm_glove = print_class_report_confusion_matrix(y_test , model_predictions[1]['svm_glove'], "SVM Weighted", "Glove Vectors")
    #
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
    #
    #
    # # prob_bayes_glove = model_probabilities[0]['bayes_glove']
    # #
    # # prob_svm_glove = model_probabilities[1]['svm_glove']
    #
    #
    # get_feature_importance(model = boost_model, features = new_feature_loss, feature_names = list(new_feature_loss.columns))
    #
    # plot_data_points = { 'x':boost_out['Loss_PolicyEff_Diff'] , 'y':boost_out['Claim_Loss_diff'] , 'z': boost_out['XGBoost_Prob_Fraud'] * 100.00}
    # plot_3d(model_output= boost_out,  data_points = plot_data_points, fig_type = "Boost_Fraud_Prob_LossPolxClaimLoss", model_type = "XGBoost")
    #
    # plot_data_points = { 'x':boost_out['Loss_PolicyEff_Diff'] , 'y':boost_out['Claim_PolicyEff_diff'] , 'z': boost_out['XGBoost_Prob_Fraud'] * 100.00 }
    # plot_3d(model_output= boost_out,  data_points = plot_data_points, fig_type = "Boost_Fraud_Prob_LossPolxClaimPol", model_type = "XGBoost")
    #
    # plot_data_points = { 'x':bag_out['Loss_PolicyEff_Diff'] , 'y':bag_out['Claim_PolicyEff_diff'] , 'z': bag_out['RandForest_Prob_Fraud'] * 100.00}
    # plot_3d(model_output= bag_out,  data_points = plot_data_points, fig_type = "Bag_Fraud_Prob_LossPolxClaimPol", model_type= "RandForest")
    #
    #
    # plot_decision_regions(X =X_test.loc[:,['Loss_PolicyEff_Diff','Claim_Loss_diff']] , y = y_test , classifier = boost_model, test_idx=None, resolution=0.02)

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
    #                   X_train = X_train_oversamp, X_test = X_test,
    #                   y_train = y_train_oversamp, y_test = y_test,
    #                   ensemble_type = "bagging"
    #                   )
