import numpy as np
import os
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
# from tensorflow import keras
# from tensorflow.python.keras.callbacks import ModelCheckpoint
#from keras import backend as K
# import tensorflow.keras.backend as K
import keras.backend as K




from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, concatenate, Input

from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from word_vector_functions import load_trained_glove_embed, encode_text, decode_text
from text_processing_cleanup import text_Processing
current_dir = os.path.abspath(os.path.curdir)
trained_rnn_dir = os.path.abspath(os.path.join(current_dir, "Trained_RNN_Classifier"))
save_dir = os.path.join(trained_rnn_dir, "Saved_Models")

# allow_soft_placement = True # Allow device soft device placement
# log_device_placement = False # Log placement of ops on devices
# session_conf = tf.ConfigProto( allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
#
# sess = tf.Session(config=session_conf)
# tf.keras.backend.set_session(sess)

class KerasTextClassifier:


    OOV_TOKEN = "UnknownUnknown"
    # K.set_session(K.tf.Session(config=cfg))
    NUM_PARALLEL_EXEC_UNITS = 2 # Number of cores per socket (use lscpu to get that number)
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True,
                        device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})

    K.set_session(tf.Session(config=config))


    def __init__(self,
                 max_word_input, word_cnt, word_embedding_dimension, labels,
                 batch_size, epoch, validation_split,
                 verbose=0):
        self.verbose = verbose
        self.max_word_input = max_word_input
        self.word_cnt = word_cnt
        self.word_embedding_dimension = word_embedding_dimension
        self.labels = labels
        self.batch_size = batch_size
        self.epoch = epoch
        self.validation_split = validation_split

        self.label_encoder = None
        self.classes_ = None
        self.tokenizer = None
        self.word2index = None
        self.index2word = None

        self.model = self._init_model()
        self._init_label_encoder(y=labels)
        self._init_tokenizer()

    def _init_model(self):



        input_layer = Input((self.max_word_input,))

        # text_embedding = Embedding(
        #     input_dim=self.word_cnt+2, output_dim=self.word_embedding_dimension,
        #     input_length=self.max_word_input, mask_zero=False)(input_layer)
        #
        # text_embedding = SpatialDropout1D(0.5)(text_embedding)

        index2vec, word2index = load_trained_glove_embed(data_type = "accept", dimension = self.word_embedding_dimension, glove_type =   "Clean_2char_NoNums",
                                                        vocab_size = "4k", recreate_embedding_word_dict = False)

        self.word2index = word2index

        index2word = {index: word for word, index in word2index.items()}

        self.index2word = index2word

        text_embedding = Embedding(
            input_dim=self.word_cnt, output_dim=self.word_embedding_dimension,
            input_length=self.max_word_input,  weights=[index2vec], trainable=False )(input_layer)




        bilstm = Bidirectional(LSTM(units=256, return_sequences=True, recurrent_dropout=0.5))(text_embedding)

        max_pool = MaxPooling1D(self.max_word_input, strides=1)(bilstm)
        max_pool = Lambda(lambda s: K.squeeze(s, axis=1))(max_pool)

        avg_pool = AveragePooling1D(self.max_word_input, strides=1)(bilstm)
        avg_pool = Lambda(lambda s: K.squeeze(s, axis=1))(avg_pool)

        # x = concatenate([GlobalAveragePooling1D()(bilstm), GlobalMaxPooling1D()(bilstm)])
        x = concatenate([avg_pool, max_pool])
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)

        output_layer = Dense(units=len(self.labels), activation="softmax")(x)
        model = Model(input_layer, output_layer)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])

        return model

    def _init_tokenizer(self):
        self.tokenizer = Tokenizer(
            num_words=self.word_cnt+1, split=', ', oov_token=self.OOV_TOKEN)

    def _init_label_encoder(self, y):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_

    def _encode_label(self, y):
        return self.label_encoder.transform(y)

    def _decode_label(self, y):
        return self.label_encoder.inverse_transform(y)

    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_word_input, value=0)

    def _preprocess(self, texts):
        # Placeholder only.
        return [text for text in texts]

    def _encode_feature(self, x):
        self.tokenizer.fit_on_texts(self._preprocess(x))
        self.tokenizer.word_index = {e: i for e,i in self.tokenizer.word_index.items() if i <= self.word_cnt}
        self.tokenizer.word_index[self.tokenizer.oov_token] = self.word_cnt + 1
        # return self._get_sequences(self._preprocess(x))

        encoded_vocab = encode_text(text = x , word_to_index =  self.word2index ,  max_sentence_length = self.max_word_input)


        return encoded_vocab

    def _decode_feature(self, X):
        decoded_vocab = decode_text(encoded_array = X, word_to_index = self.word2index)

        return decoded_vocab


    def fit(self, X, y, retrain_RNN, model_type):
        """
            Train the model by providing x as feature, y as label

            :params x: List of sentence
            :params y: List of label
        """

        encoded_x = self._encode_feature(X)
        encoded_y = self._encode_label(y)

        save_path = os.path.join(save_dir, "LSTM_{}Vocab_{}D_{}.h5".format(self.word_cnt, self.word_embedding_dimension, model_type))

        if (retrain_RNN == True) and os.path.exists(save_path):
            os.remove(save_path)

        if (os.path.exists(save_path)) and (retrain_RNN == False):
            self.model = tf.keras.models.load_model(save_path)

        else:

            self.model.fit(encoded_x, encoded_y,
                           batch_size=self.batch_size, epochs=self.epoch,
                           validation_split=self.validation_split)

            self.model.save(save_path)


    def transform(self, X):
        # return self._get_sequences(self._preprocess(X))

        return self._encode_feature(X)


    def predict_proba(self, X, y=None):
        encoded_x = self.transform(X)
        return self.model.predict(encoded_x)

    def predict(self, X, y=None):
        y_pred = np.argmax(self.predict_proba(X), axis=1)
        return self._decode_label(y_pred)





def prepare_explanation_words(rnn, encoded_x, start,  num_explanations):
    # words = pipeline.tokenizer.word_index
    # num2word = {}
    # for w in words.keys():
    #     num2word[words[w]] = w
    # x_test_words = np.stack([
    #     np.array(list(map(
    #         lambda x: num2word.get(x, "NONE"), encoded_x[i]) ) ) for i in range(10)])

    x_test_words = np.stack([
        np.array(list(map(
            lambda x: rnn.index2word.get(x, "NONE"), encoded_x[i]) ) ) for i in range(start, start+num_explanations)])

    return x_test_words
