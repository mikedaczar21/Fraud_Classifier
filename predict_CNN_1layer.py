import os
import time
import datetime

import tensorflow as tf
import numpy as np
import pandas as pd
import shap

from pathlib import Path
import dill
import shap
from tensorflow.contrib import learn
from text_classifier_CNN_1layer import TextCNN

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from word_vector_functions import grab_batches, export_predictions, save_probabilities, save_prob_pred, load_trained_glove_embed, encode_text, decode_text
from data_feature_functions import create_indicator_matrix, get_Fraud_Dataset


batch_size = 1
embedding_dim = 200            # Word Vector Embedding Dimension

current_dir = os.path.abspath(os.path.curdir)
trained_cnn_dir = os.path.abspath(os.path.join(current_dir, "Trained_CNN_Text_Classifier"))
one_layer_cnn_dir = os.path.abspath(os.path.join(trained_cnn_dir, "1layer"))

trained_cnn_pred_1layer = os.path.join(one_layer_cnn_dir, "Predictions")
saved_model_1layer_dir = os.path.join(one_layer_cnn_dir, "Saved_Models")
current_model = ""

# Checkpoint directory from training run


model_dir = "1571448776_100d_0.7drop_100f_0.08L2_0.008learn_Accept"

# Different encoded vocab
model_dir = "1582930097_100d_0.7drop_100f_0.08L2_0.008learn_Accept_8kVocab"

#OverSampling used
model_dir = "1583023297_100d_0.7drop_100f_0.08L2_0.008learn_Accept_8kVocab_SMOTE"

# No numbers or dates in training
model_dir = "1583635119_200d_0.6drop_100f_0.08L2_0.008lr_Accept_4300Vocab_CleanNoNums" # 89% precision 95% recall

model_dir = "1583655446_200d_0.75drop_100f_0.08L2_0.008lr_Accept_4512Vocab_Clean_Nums" # 90% precision 91% recall

model_dir = "1583663241_200d_0.75drop_100f_0.08L2_0.008lr_Accept_4706Vocab_Clean_Nums_2char" # 92% precision 91% recall

# No numbers or dates in training
model_dir = "1583635119_200d_0.6drop_100f_0.08L2_0.008lr_Accept_4300Vocab_CleanNoNums" # 89% precision 95% recall

model_dir = "1583812770_200d_0.75drop_100f_0.08L2_0.008lr_Accept_4421Vocab_Clean_2char_NoNums" # 89% precision 95% recall

current_model_path = os.path.join(saved_model_1layer_dir, model_dir)
vocab_path = os.path.join(current_model_path, "vocab.pk")
encoded_vocab_path = os.path.join(current_model_path, "encoded_vocab.pk")

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


# Misc Parameters
allow_soft_placement = True
log_device_placement = False



def get_cnn_pred_prob(cleaned_text , labels, testing, **kwargs):

    cnn_prob_type = kwargs['prob_type']
    recreate_prob = kwargs['recreate_prob']
    test_size = kwargs['test_size']
    rand_state = kwargs['rand_state']
    glove_type = kwargs['glove_type']
    embedding_dim = kwargs['embedding_dim']

    test_percentage = float(test_size / 100.00)

    prob_path = os.path.join(probability_dir, "eval_CNN_1layer_Acceptance_{}_{}.xlsx".format(cnn_prob_type, glove_type))
    prob_path_serialized = os.path.join(probability_dir, "eval_CNN_1layer_Acceptance_{}_{}.pk".format(cnn_prob_type, glove_type))
    pred_path = os.path.join(probability_dir, "cnn_1layer_pred_{}.pk".format(cnn_prob_type))

    if ( os.path.exists(prob_path)) and (recreate_prob == True):
        os.remove(prob_path)
        os.remove(prob_path_serialized)


    if os.path.exists(prob_path) and  (recreate_prob == False):

        prob_pred_df = pd.read_excel(prob_path)
        prob_df = prob_pred_df.loc[:, 'CNN_Prob_Fraud']
        all_predictions_scores = prob_pred_df.loc[:, 'CNN_Predictions']


    else:

        vocab_processing = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        X = np.array(list(vocab_processing.fit_transform(cleaned_text)))

        glove_data_trained, word2index = load_trained_glove_embed(data_type = "accept", dimension = embedding_dim, glove_type = "Clean_2char_NoNums",
                                                                  vocab_size = "4k", recreate_embedding_word_dict = True)

        max_doc_length = max([len(words.split(",")) for words in cleaned_text])


        encoded_vocab = encode_text(text = cleaned_text, word_to_index =  word2index,  max_sentence_length = max_doc_length)


        with open(encoded_vocab_path, 'rb') as read:
            encoded_vocab = dill.load(read)

        decoded_vocab = decode_text(encoded_array = encoded_vocab, word_to_index = word2index)

        X_train, X_test, y_train, y_test = train_test_split(encoded_vocab, labels, test_size = test_percentage, random_state = rand_state)

        text_train, text_test, label_train, label_test = train_test_split(cleaned_text, labels, test_size = test_percentage, random_state = rand_state)

        # X_train, X_test, y_train, y_test = get_training_testing_data(features = encoded_vocab,
        #                                                             labels = labels,
        #                                                             test_percentage =  test_percentage,
        #                                                             rand_state = rand_state,
        #                                                             data_type = "CNN_Encoded",
        #                                                             recreate_train_test_data = False)
        #
        #
        #
        # text_train, text_test, label_train, label_test  = get_training_testing_data(features = cleaned_text,
        #                                                             labels = labels,
        #                                                             test_percentage =  test_percentage,
        #                                                             rand_state = rand_state,
        #                                                             data_type = "CNN_Text",
        #                                                             recreate_train_test_data = False)


        X_new = pd.Series(["ALLEGEDLY Customer was walking on tile when her cart skidded due to water on the floor causing her to fall to her knee"])
        fraud_labels_new = pd.Series([1])
        non_fraud_labels_new = pd.Series([0])

        X_test_new = np.array(list(vocab_processing.fit_transform(X_new)))
        y_test_new = np.array(pd.concat([fraud_labels_new, non_fraud_labels_new] ,axis = 1))


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

                # Generate batches for one epoch
                if (testing == True):
                    output_actual = np.argmax(y_test, 1)
                    batches = grab_batches(list( zip(X_test, y_test) ), batch_size, 1, shuffle=False)


                else:

                    output_actual = np.argmax(labels, 1)
                    batches = grab_batches(list( zip(X, labels) ), batch_size, 1, shuffle=False)


                # Collect the predictions here
                all_predictions_scores = []
                all_pred_accuracy = []

                claim_prediction = []
                probability_list = []
                all_prob_scores = []
                test_lbl = []


                # Running eval batches
                for test_batch in batches:
                    x_test_batch, y_test_batch = zip( *test_batch )
                    feed_dict = {
                      input_x: x_test_batch,
                      input_y: y_test_batch,
                      dropout_keep_prob: 1.0
                    }

                    batch_pred, batch_prob = sess.run([predictions, probability_scores], feed_dict)

                    for current_prob_out in batch_prob:
                        probability_list.append( round(float(current_prob_out[1]), 3 ) )

                    test_lbl.append( np.argmax(y_test_batch)  )

                    # all_prob_scores = np.concatenate([all_prob_scores, batch_prob])
                    all_predictions_scores = np.concatenate([all_predictions_scores, batch_pred])



                output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]
                output_pred = ['Accepted' if elem == 1 else 'Rejected' for elem in all_predictions_scores]
                lbl_text =  ['Accepted' if elem == 1 else 'Rejected' for elem in test_lbl]
                output_claim_pred = ['Accepted' if elem == 0 else 'Rejected' for elem in claim_prediction]

                total_fraud_labels = sum([1.0 if elem == 'Accepted' else 0.0 for elem in output_labels])
                total_non_fraud_labels = sum([1.0 if elem == 'Rejected' else 0.0  for elem in output_labels])
                precision_fraud_count = 0.0
                precision_non_fraud_count = 0.0

                for index in range(len(output_labels)):
                    if (output_labels[index] == 'Accepted' and output_pred[index] == 'Accepted'):
                            precision_fraud_count += 1
                    elif (output_labels[index] == 'Rejected' and output_pred[index] == 'Rejected'):
                            precision_non_fraud_count += 1

                precision_fraud = float(precision_fraud_count)/total_fraud_labels
                precision_non_fraud = float(precision_non_fraud_count)/total_non_fraud_labels

                # if(output_claim_pred[0] == 'Accepted'):
                #     print("Needs to go to fraud department")
                # elif(output_claim_pred[0] == 'No'):
                #     print("Legitimate claim")

                print("Precison for fraud labels: {}".format(precision_fraud))
                print("Precison for non fraud or legit labels:{}".format(precision_non_fraud))

                print(classification_report(output_labels, output_pred))
                print(confusion_matrix(output_labels, output_pred))

                print("Accuracy score {}".format(accuracy_score(output_labels, output_pred)))

                prob_df = pd.DataFrame(probability_list, columns=['CNN_Probability_Output'])

                # output_pred_path = os.path.join(prediction_dir,  "evalulation_CNN_1layer_Acceptance_{}.xlsx".format(cnn_prob_type ))




                save_prob_pred(model_probabilities = probability_list,
                                model_predictions = output_pred,
                                    features = text_test,
                                    labels = output_labels,
                                    output_directory_path = {'excel': prob_path, 'serial':prob_path_serialized})









    return all_predictions_scores, prob_df, prob_path



def evaluate_cnn(cleaned_text , labels):




    vocab_processing = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    X = np.array(list(vocab_processing.fit_transform(cleaned_text)))

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.38, random_state = 47)

    text_train, text_test, label_train, label_test = train_test_split(cleaned_text, labels, test_size = 0.38, random_state = 47)


    X_new = pd.Series(["ALLEGEDLY Customer was walking on tile when her cart skidded due to water on the floor causing her to fall to her knee"])
    fraud_labels_new = pd.Series([1])
    non_fraud_labels_new = pd.Series([0])

    X_test_new = np.array(list(vocab_processing.fit_transform(X_new)))
    y_test_new = np.array(pd.concat([fraud_labels_new, non_fraud_labels_new] ,axis = 1))


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

            # Generate batches for one epoch

            output_actual = np.argmax(y_test, 1)
            batches = grab_batches(list( zip(X_test, y_test) ), batch_size, 1, shuffle=False)


            # Collect the predictions here
            all_predictions_scores = []
            all_pred_accuracy = []

            claim_prediction = []
            probability_list = []
            all_prob_scores = []



            for test_batch in batches:
                x_test_batch, y_test_batch = zip( *test_batch )
                feed_dict = {
                  input_x: x_test_batch,
                  input_y: y_test_batch,
                  dropout_keep_prob: 1.0
                }

                batch_pred, batch_prob = sess.run([predictions, probability_scores], feed_dict)

                for current_prob_out in batch_prob:
                    probability_list.append( round(float(current_prob_out[1]), 3 ) )

                # all_prob_scores = np.concatenate([all_prob_scores, batch_prob])
                all_predictions_scores = np.concatenate([all_predictions_scores, batch_pred])





            output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]
            output_pred = ['Accepted' if elem == 1 else 'Rejected' for elem in all_predictions_scores]
            labels_text = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]
            # output_claim_pred = ['Accepted' if elem == 0 else 'Rejected' for elem in claim_prediction]

            total_fraud_labels = sum([1.0 if elem == 'Accepted' else 0.0 for elem in output_labels])
            total_non_fraud_labels = sum([1.0 if elem == 'Rejected' else 0.0  for elem in output_labels])
            precision_fraud_count = 0.0
            precision_non_fraud_count = 0.0

            for index in range(len(output_labels)):
                if (output_labels[index].upper() == 'ACCEPTED' and output_pred[index].upper() == 'ACCEPTED'):
                        precision_fraud_count += 1
                elif (output_labels[index].upper() == 'REJECTED' and output_pred[index].upper() == 'REJECTED'):
                        precision_non_fraud_count += 1

            precision_fraud = float(precision_fraud_count)/total_fraud_labels
            precision_non_fraud = float(precision_non_fraud_count)/total_non_fraud_labels

            # if(output_claim_pred[0] == 'Yes'):
            #     print("Needs to go to fraud department")
            # elif(output_claim_pred[0] == 'No'):
            #     print("Legitimate claim")

            print("Precison for fraud labels: {}".format(precision_fraud))
            print("Precison for non fraud or legit labels:{}".format(precision_non_fraud))

            print(classification_report(output_labels, output_pred))
            print(confusion_matrix(output_labels, output_pred))

            print("Accuracy score {}".format(accuracy_score(output_labels, output_pred)))

            prob_df = pd.DataFrame(probability_list, columns=['CNN_Prob_Output'])
            #
            # output_pred_path = os.path.join(prediction_dir,  "evalulation_CNN_1layer_Acceptance.xlsx")
            #
            # save_prob_pred(model_probabilities = probability_list,
            #                 model_predictions = output_pred,
            #                     features = text_test,
            #                     labels = output_labels,
            #                     output_directory_path = output_pred_path)






    return all_predictions_scores


def evaluate_features_cnn(cleaned_text , labels):




    vocab_processing = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    X = np.array(list(vocab_processing.fit_transform(cleaned_text)))

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.38, random_state = 47)

    text_train, text_test, label_train, label_test = train_test_split(cleaned_text, labels, test_size = 0.38, random_state = 47)


    X_new = pd.Series(["ALLEGEDLY Customer was walking on tile when her cart skidded due to water on the floor causing her to fall to her knee"])
    fraud_labels_new = pd.Series([1])
    non_fraud_labels_new = pd.Series([0])

    X_test_new = np.array(list(vocab_processing.fit_transform(X_new)))
    y_test_new = np.array(pd.concat([fraud_labels_new, non_fraud_labels_new] ,axis = 1))


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

            # Generate batches for one epoch

            output_actual = np.argmax(y_test, 1)
            batches = grab_batches(list( zip(X_test, y_test) ), batch_size, 1, shuffle=False)




            # Collect the predictions here
            all_predictions_scores = []
            all_pred_accuracy = []

            claim_prediction = []
            probability_list = []
            all_prob_scores = []



            for test_batch in batches:
                x_test_batch, y_test_batch = zip( *test_batch )
                feed_dict = {
                  input_x: x_test_batch,
                  input_y: y_test_batch,
                  dropout_keep_prob: 1.0
                }

                batch_pred, batch_prob = sess.run([predictions, probability_scores], feed_dict)

                for current_prob_out in batch_prob:
                    probability_list.append( round(float(current_prob_out[0]), 3 ) )

                # all_prob_scores = np.concatenate([all_prob_scores, batch_prob])
                all_predictions_scores = np.concatenate([all_predictions_scores, batch_pred])





            output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]
            output_pred = ['Accepted' if elem == 1 else 'Rejected' for elem in all_predictions_scores]
            labels_text = ['Accepted' if elem == 1 else 'Rejected' for elem in output_actual]
            # output_claim_pred = ['Accepted' if elem == 0 else 'Rejected' for elem in claim_prediction]

            total_fraud_labels = sum([1.0 if elem == 'Accepted' else 0.0 for elem in output_labels])
            total_non_fraud_labels = sum([1.0 if elem == 'Rejected' else 0.0  for elem in output_labels])
            precision_fraud_count = 0.0
            precision_non_fraud_count = 0.0

            for index in range(len(output_labels)):
                if (output_labels[index].upper() == 'ACCEPTED' and output_pred[index].upper() == 'ACCEPTED'):
                        precision_fraud_count += 1
                elif (output_labels[index].upper() == 'REJECTED' and output_pred[index].upper() == 'REJECTED'):
                        precision_non_fraud_count += 1

            precision_fraud = float(precision_fraud_count)/total_fraud_labels
            precision_non_fraud = float(precision_non_fraud_count)/total_non_fraud_labels

            # if(output_claim_pred[0] == 'Yes'):
            #     print("Needs to go to fraud department")
            # elif(output_claim_pred[0] == 'No'):
            #     print("Legitimate claim")

            print("Precison for fraud labels: {}".format(precision_fraud))
            print("Precison for non fraud or legit labels:{}".format(precision_non_fraud))

            print(classification_report(output_labels, output_pred))
            print(confusion_matrix(output_labels, output_pred))

            print("Accuracy score {}".format(accuracy_score(output_labels, output_pred)))

            # prob_df = pd.DataFrame(probability_list, columns=['CNN_Prob_Output'])
            #
            # output_pred_path = os.path.join(prediction_dir,  "evalulation_CNN_1layer_Acceptance.xlsx")
            #
            # save_prob_pred(model_probabilities = probability_list,
            #                 model_predictions = output_pred,
            #                     features = text_test,
            #                     labels = output_labels,
            #                     output_directory_path = output_pred_path)






    return all_predictions_scores, prob_df
