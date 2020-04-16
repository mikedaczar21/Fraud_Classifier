import os
import time
import datetime


import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

from text_classifier_CNN_1layer import TextCNN
from data_feature_functions import create_indicator_matrix, get_Fraud_Dataset
from word_vector_functions import read_glove_file, load_trained_glove_word, load_trained_glove_embed, encode_text
from text_processing_cleanup import  text_Processing, get_Vocab_Set, find_Max_Sentence_Length, text_Processing_GloVe


current_dir = os.path.abspath(os.path.curdir)
trained_cnn_dir = os.path.abspath(os.path.join(current_dir, "Trained_CNN_Text_Classifier"))
one_layer_cnn_dir = os.path.abspath(os.path.join(trained_cnn_dir, "1layer"))

trained_cnn_pred_1layer = os.path.join(one_layer_cnn_dir, "Predictions")
saved_model_1layer_dir = os.path.join(one_layer_cnn_dir, "Saved_Models")




# Parameterslets
# ==================================================

# Data loading params
valid_sample_percentage = 0.3 # Percentage of the training data to use for validation

label_data =  "Accept"

data_type = "accept"
vocab_size = "4k"
glove_type =   "Clean_2char_NoNums"

# Model Hyperparameters
embedding_dim = 200            # Word Vector Embedding Dimension
filter_sizes =  "2,3,4,5,6,7,8"         # Filter sizes denoted in string
num_filters = 100               # Number of filters per filter size
dropout_keep_prob = 0.75       # Dropout keep probability (probability that neurons are dropped for training)
l2_reg_lambda = 0.08            # L2 Regularization Lambda (parameter for equation)
learning_rate = 8e-3            # Learning rate for optimizer function (AdamOptimizer in this case)
init_learn_rate = 0.05
max_unique_words = 3079         # Vocab size after data cleaning
glove_dim = "{}d".format(embedding_dim)

# Training parameters
batch_size = 1200               # 7% of total data size
num_epochs = 250               # Number of training epochs
evaluate_every = 200           # Evaluate model after this number of steps
checkpoint_every = 400      # Save model after this many steps
num_checkpoints = 6             # Max number of checkpoints to store



allow_soft_placement = True # Allow device soft device placement
log_device_placement = False # Log placement of ops on devices




# Grabbing next round of batches
def grab_batches(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int( (data_size-1) /batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]







if __name__ == '__main__':



    fraud_data = get_Fraud_Dataset(recreate_features = False, fraud_type= 'acceptance')

    null_policyEff_rows = fraud_data[ fraud_data['Policy Eff Date'].isna() == True].index
    fraud_data.drop(null_policyEff_rows, inplace = True)

    text = fraud_data['Fraud_Text'] # features or inputs into model
    labels = fraud_data['Fraud_Label'] # labels






    cleaned_text = text.apply(text_Processing, numbers=False)

    # vocab_set = create_Vocab_Set(text)

    # vocab_set = set([])
    # for sentence in list(text):
    #     words = sentence.split(" ")
    #     for each_word in words:
    #         if each_word.isalpha() == True:
    #             vocab_set.add(each_word.lower())
    #
    #     # [vocab_set.add(elem.lower()) if elem.isalpha() == True else print("") for elem in words]
    #
    # vocab_size = len(vocab_set)
    # print("Vocab size: " + str(vocab_size))
    #
    # vocab_list = list(vocab_set)

    # glove_data =  generate_embedding_TF( "embedding_array_tf100", embedding_dim)
    # vocab_size = np.size(glove_data, 0)
    # glove_word = read_glove_file(vocab_set, glove_dim, "regular_CNN")

    #glove_word = load_trained_glove_word(data_type = "acceptance", dimension = embedding_dim)



    glove_data_trained, word2index = load_trained_glove_embed(data_type = "accept", dimension = embedding_dim, glove_type =   "Clean_2char_NoNums",
                                                              vocab_size = "4k", recreate_embedding_word_dict = True)


    max_doc_length = max([len(words.split(",")) for words in cleaned_text])


    vocab_processing = learn.preprocessing.VocabularyProcessor(max_doc_length)
    vocab_dict = vocab_processing.fit_transform(cleaned_text)
    X = np.array(list(vocab_processing.fit_transform(cleaned_text)))
    y = np.array( create_indicator_matrix(labels, check_index = True) )

    encoded_vocab = encode_text(text = cleaned_text , word_to_index =  word2index ,  max_sentence_length = max_doc_length)
    vocab_size = np.size(glove_data_trained, 0)
    #vocab_size = len(vocab_processing.vocabulary_)

    X_train, X_test, y_train, y_test = train_test_split(encoded_vocab, y, test_size = 0.38, random_state = 47)

    # y_test =  np.array(create_indicator_matrix(pd.Series(y_test)))
    #
    # print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    # print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    #
    # over_sampling = SMOTE(random_state=777, k_neighbors=1)
    # X_train_oversamp, y_train_oversamp = over_sampling.fit_sample(X_train, y_train)
    #
    #
    # print('After OverSampling, the shape of train_X: {}'.format(X_train_oversamp.shape))
    # print('After OverSampling, the shape of train_y: {} \n'.format(y_train_oversamp.shape))
    #
    # print("After OverSampling, counts of label '1': {}".format(sum(y_train_oversamp==1)))
    # print("After OverSampling, counts of label '0': {}".format(sum(y_train_oversamp==0)))
    #
    # y_train_oversamp = np.array(create_indicator_matrix(y_train_oversamp))

    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(encoded_vocab, y, test_size = 0.38, random_state = 121)

    train_size = y_train.shape[0]

    #============================================================================
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 47)
    # shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    # x_shuff = X_train[shuffle_indices]
    # y_shuff = y_train[shuffle_indices]
    #
    # # split again into train and dev data sets
    # X_train_dev, X_test+_dev, y_train_dev, y_test_dev = train_test_split(x_shuff, y_shuff, test_size=0.40)
    #============================================================================


    # Build TF Graph and CNN object
    main_graph = tf.Graph()
    with main_graph.as_default():

        session_conf = tf.ConfigProto( allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)

        sess = tf.Session(config=session_conf)
        filter_size_list = list( map(int, filter_sizes.split(",") ) )

        with sess.as_default():
            # creating CNN object with params defined at top of script
            cnn = TextCNN(
                max_sentence_length= max_doc_length,
                num_classes= 2,
                vocab_size= vocab_size,
                embedding_size=embedding_dim,
                embedding_shape = glove_data_trained.shape ,
                filter_sizes= filter_size_list,
                num_filters=num_filters,
                l2_reg_lambda= l2_reg_lambda)


            global_step = tf.Variable(0, name="global_step", trainable=False)
            # Using decaying learning rate
            # learn_rate = tf.train.exponential_decay(init_learn_rate, global_step,  y_train.shape[0], 0.96, staircase=True)

            # Training operation with AdamOptimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            #train_op = tf.train.AdamOptimizer(learning_rate).minimize(cnn.loss, global_step = global_step)


            # Output directory for models and summaries
            timestamp =  str(int(time.time()))
            train_params = timestamp + "_{}d_{}drop_{}f_{}L2_{}lr_{}_{}Vocab_{}".format(embedding_dim,dropout_keep_prob, num_filters,
                                                                                l2_reg_lambda, learning_rate, label_data, vocab_size, glove_type
                                                                                )
            out_dir = os.path.join(saved_model_1layer_dir , train_params)


            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation Summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Need to create one for TF
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "CNN_model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            probability_dir = os.path.abspath(os.path.join(out_dir, "probability_output"))
            if not os.path.exists(probability_dir):
                os.makedirs(probability_dir)

            saver = tf.train.Saver(tf.global_variables())


            # One training step (train model with one batch)
            def train_step(x_batch, y_batch):

                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                # Running the training session
                _, step, summaries, loss, accuracy = sess.run(
                                                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                                                      feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("Training {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # One evaluation step (evaluate model with one batch)
            def test_step(x_batch, y_batch, writer=None):

                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                # Running the testing/evaluation session
                step, summaries, loss, accuracy, num_correct = sess.run(
                                                                    [
                                                                     global_step,
                                                                     test_summary_op,
                                                                     cnn.loss,
                                                                     cnn.accuracy,
                                                                     cnn.num_correct
                                                                     ],
                                                                feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("Evaluating {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

                return num_correct


            # Save the word_to_id map so can load to make predicts on new data
            vocab_processing.save(os.path.join(out_dir, "vocab.pk") )

            with open(os.path.join(out_dir, "encoded_vocab.pk"), 'wb') as write:
                dill.dump(encoded_vocab, write)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Initialize word embeddings
            sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: glove_data_trained})


            #==================================================================================
            # Start training by generating some batches
            train_batches = grab_batches( list( zip(X_train, y_train)),batch_size, num_epochs)
            best_test_accuracy = 0
            best_accuracy_at_step = 0, 0

            # Train CNN with x_train and y_train batch by batch (looping through batches)
            for train_batch in train_batches :
                # grabbing current batch
                x_train_batch, y_train_batch = zip( *train_batch )
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Based on evaluate_every param, will use test data (x_test, y_test) to evaluate CNN for this batch
                if current_step % evaluate_every == 0:
                    # Grabbing batches to test model and evaluate it's accuracy
                    dev_batches = grab_batches( list( zip( X_test, y_test )) , batch_size, 1)
                    total_dev_correct_pred = 0
                    for dev_batch in dev_batches:
                        X_dev_batch, y_dev_batch = zip( *dev_batch )
                        num_dev_correct_pred =  test_step(X_dev_batch, y_dev_batch, writer=test_summary_writer)
                        total_dev_correct_pred += num_dev_correct_pred

                    # Getting dev accuracy
                    dev_accuracy = float( total_dev_correct_pred ) / len(y_test)
                    print("Accuracy on test set: {} ".format(dev_accuracy))

                    # Save CNN if it is the best based on accuracy from dev set
                    if dev_accuracy >= best_test_accuracy:
                        best_test_accuracy, best_accuracy_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model at {} at step {}.".format(path, best_accuracy_at_step))
                        print("Best accuracy is {} at setp {}.".format(best_test_accuracy, best_accuracy_at_step))

                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


            # # Evaluating with test set (larger set of split data)
            test_batches = grab_batches( list( zip( X_test_val, y_test_val)), batch_size, 1)
            total_test_correct = 0
            for test_batch in test_batches:
                X_test_batch, y_test_batch = zip ( *test_batch )
                num_test_correct_pred = test_step(X_test_batch, y_test_batch )
                total_test_correct += num_test_correct_pred

            test_accuracy = float( total_test_correct )/ len( y_test)

            print("Accuracy on test set is {} based on the best model ".format(test_accuracy))
