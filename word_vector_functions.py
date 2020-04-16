# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import dill
import os
import nltk
from joblib import dump, load
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from langdetect import detect_langs
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, word2vec
import re


# Importing functions from other scripts
from text_processing_cleanup import get_Vocab_Set


current_dir = os.path.abspath(os.path.curdir)
word_vec_dir = os.path.abspath(os.path.join(current_dir, "Word_Vectors"))

# Glove paths
glove_dir = os.path.abspath(os.path.join(word_vec_dir, "GloVe"))
trained_glove_dir = os.path.abspath(os.path.join(glove_dir, "Trained"))

# Word2vec paths
word2vec_dir = os.path.abspath(os.path.join(word_vec_dir, "Word2Vec"))

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


def encode_text(text, word_to_index,  max_sentence_length):


    encoded_list = []
    for sentence in list(text):

        words_split = sentence.split(', ')
        new_sentence = [0] * max_sentence_length
        for word_index, word in enumerate(words_split):

            if word in word_to_index:
                new_sentence[word_index] = word_to_index[word]
            else:
                new_sentence[word_index] = 0

        encoded_list.append(new_sentence)


    encoded_array = np.array(encoded_list)

    return encoded_array
    

def decode_text(encoded_array, word_to_index):


    decode_list = []

    index2word = {index: word for word, index in word_to_index.items()}

    for sent in encoded_array:

        sent_text = []

        for word_idx in sent:

            if word_idx in index2word:
                current_text = index2word[word_idx]
                sent_text.append(current_text)

        decode_list.append(", ".join(sent_text))


    return pd.Series(decode_list)


def read_glove_file(data_type,  dimension, vocab_size, glove_type):

    glove_data = {}


    word_dict_name = '{}d_{}_{}_{}_Word_Dict'.format(dimension, data_type, vocab_size, glove_type)
    word_dict_xls = os.path.join(trained_glove_dir , word_dict_name + '.xlsx')
    word_dict_serial = os.path.join(trained_glove_dir , word_dict_name + '.pk')

    glove_name = '{}d_{}_{}_Vocab_{}.pk'.format(dimension, data_type, vocab_size,  glove_type)
    glove_path = os.path.join(trained_glove_dir , glove_name )


    if not os.path.exists(word_dict_serial):

        with open(glove_path, 'rb') as inputFile:
            glove_load = dill.load(inputFile)


            for word, nums in glove_load.items():
                nums = np.array(nums, dtype=np.float32)
                glove_data[word] = nums




        with open(word_dict_serial, 'wb') as write:
            dill.dump(glove_data, write)

        glove_df = pd.DataFrame.from_dict(glove_data, orient="index")

        with pd.ExcelWriter(word_dict_xls) as xlsx_writer:
            glove_df.to_excel(xlsx_writer, "Vocab_Dict{}_{}".format(vocab_size, dimension), header=True, index_label= False)
            xlsx_writer.save()

    else:

        with open(word_dict_serial, 'rb') as read:
            glove_data = dill.load(read)


    return glove_data



def get_sentence_feature_values(sentence, embedding, embedding_dim):

    stopWords = set(stopwords.words('english'))

    sentence_split = sentence.split(", ")

    sent_stopWords = [word for word in sentence_split if not word in stopWords if word != '' and word != r'\s']

    sent_found = [word for word in sent_stopWords if word.isalnum() and word in embedding]

    if len(sent_found) == 0:
        return np.hstack([np.zeros(embedding_dim)])

    sent_glove_vals = np.array([embedding[word] for word in sent_found])

    # getting the average for each column
    glove_avg = sent_glove_vals.mean(axis = 0)


    return glove_avg

def get_sentence_vector(sentence, embedding, embedding_dim):


    sentence_split = sentence.split(", ")

    sent_found = [word for word in sentence_split if word.isalnum() and word in embedding]

    if len(sent_found) == 0:
        return np.hstack([np.zeros(embedding_dim)])

    vs = np.zeros(embedding_dim)
    sentence_len = len(sent_found)
    for word in sent_found:
        vs = np.add(vs, embedding[word])

    vs = np.divide(vs, sentence_len)


    return vs



def get_sentence_embeddings(text, embedding, embedding_size):


    # sentence_embedding_avg = [get_sentence_feature_values(sentence = words, embedding = embedding, embedding_dim = embedding_size) for words in text ]

    sentence_embedding_avg = [get_sentence_vector(sentence = words, embedding = embedding, embedding_dim = embedding_size) for words in text ]

    # calculate PCA of sentences
    pca = PCA()
    pca.fit(np.array(sentence_embedding_avg))
    pca_vec = pca.components_[0]
    pca_vec = np.multiply(pca_vec, np.transpose(pca_vec))  # pca_vec x pca_vec_transpose (element wise)


    if len(pca_vec) < embedding_size:
        for i in range(embedding_size - len(pca_vec)):
            pca_vec = np.append(pca_vec, 0)  # add needed extension for multiplication below

    # resulting sentence vectors, sent_vecs = sent_avg - (pca_vec x pca_vec_transpose x sent_avg)
    sentence_vecs_out = []
    for sent_vec in sentence_embedding_avg:
        sub = np.multiply(pca_vec, sent_vec)
        sentence_vecs_out.append(np.subtract(sent_vec, sub))


    return sentence_vecs_out





def load_trained_glove_embed(data_type, dimension, vocab_size, glove_type, recreate_embedding_word_dict):

    embedding_array = []

    glove_name = 'Trained{}d_{}_{}_Vocab.pk'.format(dimension, "acceptance", "6k")
    glove_name = '{}d_{}_{}_Vocab_{}.pk'.format(dimension, data_type, vocab_size, glove_type)

    glove_trained = os.path.join(trained_glove_dir , glove_name)

    glove_embed_name = '{}d_{}_{}_index2embedding_{}.pk'.format(dimension, data_type,vocab_size, glove_type)
    glove_embed_path = os.path.join(trained_glove_dir , glove_embed_name)

    word_dict_name = '{}d_{}_{}_word2index_{}'.format(dimension, data_type,vocab_size, glove_type)
    word_dict_serial = os.path.join(trained_glove_dir ,word_dict_name + '.pk')
    word_dict_xls = os.path.join(trained_glove_dir , word_dict_name + '.xlsx')

    word_to_index_dict = dict()

    if (os.path.exists(glove_embed_path)) and (os.path.exists(word_dict_serial)) and recreate_embedding_word_dict == True:
        os.remove(glove_embed_path)
        os.remove(word_dict_serial)
        os.remove(word_dict_xls)

    if not os.path.exists(glove_embed_path):

        with open(glove_trained, 'rb') as read_file:
            glove_data = dill.load(read_file)

            for index, (word, vector_repres) in enumerate(glove_data.items(), start = 1):

                nums = np.array(
                        [float(elem) for elem in vector_repres]
                )
                word_to_index_dict[word] = index
                embedding_array.append(nums)

        last_index = index + 1
        unkown_word = [0.0] * len(nums)
        word_to_index_dict = defaultdict( lambda: last_index, word_to_index_dict)
        embedding_array = np.array(embedding_array + [unkown_word])

        word_dict_series = pd.Series(word_to_index_dict)



        with open(glove_embed_path, 'wb') as write_file:
            dill.dump(embedding_array, write_file)

        with open(word_dict_serial, 'wb') as write:
            dill.dump(word_to_index_dict, write)

        with pd.ExcelWriter(word_dict_xls) as xlsx_writer:
            word_dict_series.to_excel(xlsx_writer, "Word_Dict{}_{}".format(vocab_size, dimension), header=True, index_label= False)
            xlsx_writer.save()

    else:
         with open(glove_embed_path, 'rb') as read_file:
             embedding_array = dill.load(read_file)

         with open(word_dict_serial, 'rb') as read_file:
             word_to_index_dict = dill.load(read_file)


    return embedding_array, word_to_index_dict



def load_trained_glove_word(data_type, dimension):

    trained_glove_name = '{}d_{}.pk'.format(dimension, data_type)

    glove_trained = os.path.join(trained_glove_dir , trained_glove_name)

    if os.path.exists(glove_stored_path):
        with open(glove_stored_path, 'rb') as read_file:
            glove_data = dill.load(read_file)

        return glove_data

def load_embedding_TF( glove_input, glove_dimen, vocab_size):

    embedding_array = []
    word_index_dict = dict()
    glove_stored = '{}_{}.joblib'.format(glove_input, glove_dimen)
    glove_store_path = os.path.join(mydir, glove_output_dir , glove_stored)

    embedding_array = load(glove_store_path)
    output_array =  np.zeros((size, dimensions), dtype=np.float32)

    index = 0
    for index in range(vocab_size):
        output_array[index] = embedding_array[index]


    return output_array


def load_word2vec(text, dimension, text_type, clear_w2v):
    """
    Takes in training features and builds word2vec word vectors using gensim library along with the parameters below.
    Also to save memory on training model each time, if we are using the same model then will just load it.
    Returns: the word2vec word vectors
    """

    num_features = 200 # dimensions of word vectors
    min_word_count = 2
    num_workers = 8 # Number of threads to run in parallel
    context = 7 # Context window size
    downsampling = 1e-3 # downsampling for frequent words

    word2vec_name = 'accept_{}d_{}'.format(dimension, text_type)


    model_path = os.path.join(word2vec_dir, word2vec_name)

    if clear_w2v == True and os.path.exists(word2vec_dir):
        for root, dirs, files in os.walk(word2vec_dir):
            for file in files:
                os.remove(os.path.join(root,file))

    if not os.path.exists(model_path):
        # Initialize and train model
        model = word2vec.Word2Vec(text, workers = num_workers, size=dimension, min_count = min_word_count, window = context, sample= downsampling)
        model.init_sims(replace= True)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)



    w2v_vectors = { words : vector for words, vector in zip(model.wv.index2word, model.wv.vectors)}

    return w2v_vectors


def export_predictions(model_predictions, test_labels, test_features, prediction_type, create_again, prob_directory_path):

    """
        Takes in a list of dictionaries containing the predictions made by the
        different classifier configurations and outputs those results to an
        excel file and will recreate it if the create_again flag is true.
    """


    df_pred_export = pd.DataFrame( {'Text Feature ': test_features, 'Actual Labels (Test)': test_labels, 'Predicted': model_predictions})


    pred_file_test = "predictions_{}.xlsx".format(prediction_type)
    pred_path_test = os.path.join(pred_directory_path, pred_file_test)

    print(df_pred_export)

    # deleting old file and reloading new one
    if create_again == True and os.path.exists(pred_path_test):
            os.remove(pred_path_test)

    with pd.ExcelWriter(pred_path_test) as pred_writer:
        df_pred_export.to_excel(pred_writer, sheet_name='Evaluation Results')
        pred_writer.save()


    return df_pred_export



def save_probabilities(model_probabilities, features, labels,   output_directory_path):

    """
        Takes in a list of dictionaries containing the predictions made by the
        different classifier configurations and outputs those results to an
        excel file and will recreate it if the create_again flag is true.
    """


    df_prob_export = pd.DataFrame( {' Feature ': features, 'Labels': labels, 'Probability': model_probabilities})






    with pd.ExcelWriter(prob_path_test) as prob_writer:
        df_prob_export.to_excel(prob_writer, sheet_name='Evaluation Results')
        prob_writer.save()


    return df_prob_export


def save_prob_pred(model_probabilities, model_predictions, features, labels,  output_directory_path):

    """
        Takes in a list of dictionaries containing the predictions made by the
        different classifier configurations and outputs those results to an
        excel file and will recreate it if the create_again flag is true.
    """


    df_prob_export = pd.DataFrame( {'Feature': features, 'Actual': labels, 'CNN_Predictions': model_predictions,  'CNN_Prob_Fraud': model_probabilities})

    excel_path = output_directory_path['excel']
    serial_path = output_directory_path['serial']

    with pd.ExcelWriter(excel_path) as prob_writer:
        df_prob_export.to_excel(prob_writer, sheet_name='Evaluation Results')
        prob_writer.save()

    with open(serial_path, 'wb') as write:
        dill.dump(df_prob_export, write)


    return df_prob_export
