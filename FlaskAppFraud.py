
from flask import Flask, render_template, request, url_for, Markup
import numpy as np

import dill

from math import sqrt
import os
import re
from collections import Counter, defaultdict

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix




import time
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow.contrib import learn
import category_encoders as ce
import matplotlib.pyplot as plt
import shap


from nltk.corpus import stopwords
from text_processing_cleanup import text_Processing, text_Processing_GloVe
from data_feature_functions import create_indicator_matrix, get_Fraud_Dataset, get_combined_feature
from predict_CNN_1layer import evaluate_cnn
from word_vector_functions import grab_batches, get_sentence_feature_values, read_glove_file, get_sentence_embeddings



batch_size = 1


# Evaluate on all training data

current_dir = os.path.abspath(os.path.curdir)

trained_ensemble_dir = os.path.abspath(os.path.join(current_dir, "Trained_Ensemble_Classifier"))
boosting_dir = os.path.abspath(os.path.join(trained_ensemble_dir, "Boosting"))
bagging_dir = os.path.abspath(os.path.join(trained_ensemble_dir, "Bagging"))
trained_boosting_pred_1layer = os.path.join(boosting_dir, "Predictions")
saved_model_boosting_dir = os.path.join(boosting_dir, "Saved_Models")



trained_boosting_pred_1layer = os.path.join(boosting_dir, "Predictions")
saved_model_boosting_dir = os.path.join(boosting_dir, "Saved_Models")
trained_cnn_dir = os.path.abspath(os.path.join(current_dir, "Trained_CNN_Text_Classifier"))
one_layer_cnn_dir = os.path.abspath(os.path.join(trained_cnn_dir, "1layer"))

trained_cnn_pred_1layer = os.path.join(one_layer_cnn_dir, "Predictions")
saved_model_1layer_dir = os.path.join(one_layer_cnn_dir, "Saved_Models")
current_model = ""

# Checkpoint directory from training run


model_dir = "1583812770_200d_0.75drop_100f_0.08L2_0.008lr_Accept_4421Vocab_Clean_2char_NoNums" # 89% precision 95% recall
current_model_path = os.path.join(saved_model_1layer_dir, model_dir)
vocab_path = os.path.join(current_model_path, "vocab.pk")


prediction_dir = os.path.join(one_layer_cnn_dir, "Predictions")
probability_dir = os.path.join(current_model_path, "probability_output")
pred_type = "CNN_1layers_GloVe"

checkpoint_path = os.path.join(current_model_path, "checkpoints")

# Evaluate on all training data
eval_train = False
prob_file_name = "probability_CNN_1layer_Acceptance.xlsx"
pred_file_name = "evalulation_CNN_1layer_Acceptance.xlsx"

output_prob_path = os.path.join(probability_dir, prob_file_name)
output_pred_path = os.path.join(prediction_dir,  pred_file_name)

demo_data_dir = os.path.join(current_dir, "DemoData")

image_dir = os.path.join(current_dir, "static")

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

test_size = 38 # out of 100 percent for train-test split
test_percentage = float(test_size / 100.00)
rand_state = 47 # random state of train/test split
embedding_dim  = 200


text_type = "Clean_Nums_2char"
glove_sum = "VecAvg"
recreate_full_xgb = False
fraud_type= 'acceptance' # fraud_type= 'acceptance' or 'refferal', refferal has all 79k claims

boosting_type= "xgboost_{}test_{}_FullData".format(test_size, 'VecAvg')
boosting_stored_file = "Trained_{}.pk".format(boosting_type)
boosting_path = os.path.join(saved_model_boosting_dir, boosting_stored_file)

def get_cnn_pred_demo(cleaned_text, labels):


    vocab_processing = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    X = np.array(list(vocab_processing.fit_transform(cleaned_text)))


    # Grabbing latest checkpoint and setting up TF graph

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)

    # Running graph by loading checkpointed TF graph
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=allow_soft_placement,
          log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]


            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            accuracy_scores = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            probability_scores = graph.get_operation_by_name("prediction_prob/prob").outputs[0]


            claim_prediction = []
            probability_list = []

            output_actual = np.argmax(labels, 1)
            batches = grab_batches(list( zip(X, labels) ), batch_size, 1, shuffle=False)



            feed_dict = {
              input_x: X,
              input_y: labels,
              dropout_keep_prob: 1.0
            }

            batch_pred, batch_prob = sess.run([predictions, probability_scores], feed_dict)

            for current_prob_out in batch_prob:
                    probability_list.append( float(current_prob_out[1]))

            claim_prediction = np.concatenate([claim_prediction, batch_pred])



            output_actual = np.argmax(labels, 1)

            output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]

            output_claim_pred = ['Accepted' if elem == 1 else  'Rejected' for elem in claim_prediction]

            prob_df = pd.DataFrame(probability_list, columns=['CNN_Probability_Output'])




    return output_claim_pred, prob_df


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def claimfield():
    return render_template('ClaimField.html')

@app.route('/predict', methods = ['POST'])
def predict():

    cnn_model_flag = False
    # claim = [request.form['Claim']] #get HTML input as string and change into list
    model_type = request.form['model']
    fraud_data = request.form['fraud_data']

    demo_data_path = os.path.join(demo_data_dir, fraud_data)

    fraud_data = pd.read_excel(demo_data_path)

    print("Fraud Data input:\n {}\n".format(fraud_data))


    text = fraud_data['Fraud_Text'] # features or inputs into model
    labels = fraud_data['Fraud_Label'] # labels

    cleaned_text = text.apply(text_Processing, numbers=False)

    glove_dict = read_glove_file(data_type = "accept", dimension = embedding_dim,  vocab_size = "4k", glove_type = text_type)

    clean_list = cleaned_text.tolist()

    # cnn_pred, cnn_prob = get_cnn_pred_demo( cleaned_text = cleaned_text, labels = labels_onehot)
    #
    # print("CNN pred: {}\n CNN Prob{}\n".format(cnn_pred, cnn_prob))


    new_feature_labels = fraud_data['Fraud_Label']



    print("Feature Cols {}".format(fraud_data.columns))
      # Encoding main/sub cause columns

    feature_cols = [

                   'Loss_PolicyExp',
                    'Claim_PolicyEff',
                    'Claim_Loss',
                    'Longitude',
                    'Latitude',
                    'Sub_Business',
                    'Cause',
                    'Loss_Glove_Avg'
                    ]

    new_feat_avg = fraud_data[feature_cols]

    new_feat_avg.rename(

                    columns = {
                        'Loss_Glove_Avg' : 'Loss_Descrip'
                    },
                    inplace=True
    )


    features_orig = [

                       'Loss_PolicyExp',
                        'Claim_PolicyEff',
                        'Claim_Loss',
                        'Longitude',
                        'Latitude',
                        'Subline_Business',
                        'Cause_Join',
                        'Loss_Descrip_NoAddition'
                        ]
    new_feat_orig = fraud_data[features_orig]




    # new_feature = get_combined_feature(
    #                                 cleaned_Text = cleaned_text,
    #                                 features =  features,
    #                                 cnn_Prob = cnn_prob,
    #                                 recreate_combined_features = False,
    #                                 feature_type = 'demo'
    # )
    #
    # new_feat_avg = new_feature.iloc[:, 1:] # Not using text for XGBoost
    #
    # print("XGB Feature {}\n".format(new_feat_avg))

    # if model_type == 'xgboost':
    #     model_predictions = list(trained_xgb.predict(new_feat_avg))
    #     model_prob = trained_xgb.predict_proba(new_feat_avg)
    #
    #     fraud_score = [round( (10 * prob[1] ), 2) for prob in model_prob]
    #
    # elif model_type == 'cnn_1layer':
    #     model_predictions = cnn_pred
    #     fraud_score = []
    #     for index, row in cnn_prob.iterrows():
    #         fraud_score.append( round((10 * row['CNN_Probability_Output']), 2))



    model_predictions = list(trained_xgb.predict(new_feat_avg))
    model_prob = trained_xgb.predict_proba(new_feat_avg)

    fraud_score = [round( (10 * prob[1] ), 2) for prob in model_prob]

    confidence_prob = []
    for index, elem in enumerate(model_predictions):

        # Fraud prection (or 'accepted' )
        if elem == 1:
            confidence_prob.append( round(100 * float(model_prob[index][1]), 2) )
        # Non-fraud prediction (or 'rejected')
        elif elem == 0:
            confidence_prob.append(round( 100 * float(model_prob[index][0]), 2) )


    output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in labels]
    output_pred = ['Accepted' if elem == 1 else 'Rejected' for elem in model_predictions]

    shap.initjs()
    shap_explainer = shap.TreeExplainer(trained_xgb)
    shap_values = shap_explainer.shap_values(new_feat_avg)

    dec_plot_list = []
    for row in range(len(new_feat_orig)):
        plt.clf()
        shap.decision_plot(shap_explainer.expected_value, shap_values[row,:], new_feat_orig.iloc[row,:], link='logit', show=False)
        fig_name =   "decision_plot_{}_{}.png".format(output_labels[row], new_feat_orig.iloc[row, 2])
        fig_path = os.path.join(image_dir, fig_name)

        if os.path.exists(fig_path) == False:
            # plt.savefig(fig_path, dpi=1000, bbox_inches='tight' )
            plt.savefig(fig_path, dpi=400, bbox_inches='tight')
            

        dec_plot_list.append(fig_name)


    print("Model Pred Len: {}\n Label Len: {} \n Clean_Text Len: {}\n Fraud Score Len: {}".format(len(model_predictions), len(labels), len(cleaned_text), len(fraud_score)) )

    [print("{}\n".format(elem) ) for elem in dec_plot_list]

    model_output = {'feature': cleaned_text, 'labels': output_labels, 'pred': model_predictions,  'prob': fraud_score}

    model_output_df = pd.DataFrame(model_output)

    print("\nModel Output {} \n".format(model_output))

    num_pred = len(model_predictions)

    return render_template('Predict.html',
                            feature = text.tolist() ,
                            dec_plots = dec_plot_list,
                            pred = output_pred,
                            prob = confidence_prob,
                            model_type = model_type,
                            num_pred = num_pred) #output prediction

    # return render_template('Predict.html', model_output = model_list, model_type = model_type) #output prediction

if __name__ == '__main__':


    with open(boosting_path, 'rb') as boost_file:
        trained_xgb = dill.load(boost_file)

    # with open('C:/Current_Projects/BayesClassifierProject/Stored_Classifiers_Acceptance/svm_word2vec_100d.pk', 'rb') as svm_word2vec:
    #     svm_word2vec_pipeline = dill.load(svm_word2vec)
    # with open('C:/Current_Projects/BayesClassifierProject/Stored_Classifiers_Acceptance/tree_glove_100d.pk', 'rb') as tree_glove:
    #     tree_glove_pipeline = dill.load(tree_glove)
    # with open('C:/Current_Projects/BayesClassifierProject/Stored_Classifiers_Acceptance/tree_word2vec_100d.pk', 'rb') as tree_word2vec:
    #     tree_word2vec_pipeline = dill.load(tree_word2vec)

    # claims_data = get_data() # This is from my data_glove_function
    #
    # X = claims_data['Claims_Description'] # features or inputs into model
    # y = claims_data['Fraud_Label'] # labels
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)




    app.run(debug=True, host='0.0.0.0', port=80)
