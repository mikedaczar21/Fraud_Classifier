#importing the glove library
from glove import Corpus, Glove
import os
import dill
from gensim.models import Word2Vec, word2vec
import pandas as pd
import numpy as np

# Importing helper functions from other scripts
from data_feature_functions import get_Fraud_Dataset
from text_processing_cleanup import   text_Processing_GloVe, text_Processing

current_dir = os.path.abspath(os.path.curdir)
word_vec_dir = os.path.abspath(os.path.join(current_dir, "Word_Vectors"))
glove_dir = os.path.abspath(os.path.join(word_vec_dir, "GloVe"))
trained_glove_dir = os.path.abspath(os.path.join(glove_dir, "Trained"))

# Param for GlovE for number of features
num_components = 200
vocab_size = '4k'
glove_type = "Clean_2char_NoNums"

def train_glove():


    fraud_data = get_Fraud_Dataset(recreate_features = False)

    X = fraud_data['Fraud_Text'] # features or inputs into model
    y = fraud_data['Fraud_Label'] # labels

    #labels = create_indicator_matrix(fraud_data['Fraud_Label'])


    # vocab = create_Vocab_Set(X)

    cleaned_text = X.apply(text_Processing_GloVe)

    cleaned_list = cleaned_text.tolist()


    # creating a corpus object
    corpus = Corpus()
    #training the corpus to generate the co occurence matrix which is used in GloVe
    corpus.fit(cleaned_list, window=6)
    #creating a Glove object which will use the matrix created in the above lines to create embeddings
    #We can set the learning rate as it uses Gradient Descent and number of components
    glove = Glove(no_components= num_components, learning_rate=0.05)

    trained_glove_name = '{}d_accept_train_{}_Vocab_{}.model'.format(num_components,vocab_size, glove_type)
    glove_train_path =  os.path.abspath(os.path.join(trained_glove_dir, trained_glove_name))

    glove.fit(corpus.matrix, epochs=200, no_threads=8, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save(glove_train_path)

    print("Dictionary")
    print(glove.dictionary)
    print("Word Vectors")
    print(glove.word_vectors)


def load_glove_accept():
    trained_glove_name = '{}d_accept_train_{}_Vocab_{}.model'.format(num_components,vocab_size, glove_type)
    glove_train_path = os.path.join( trained_glove_dir, trained_glove_name)

    new_glove = Glove.load(glove_train_path)

    combined_dict = {}

    for word, index in new_glove.dictionary.items():
        combined_dict[word] =  new_glove.word_vectors[index]



    for word, word_vec in combined_dict.items():
        print(word)
        # print(word_vec)
        print("Dimension {}".format(len(word_vec)))
        print("\n")
    print("Total number of words: {}".format( len(combined_dict) ))
    num_vocab =  len(combined_dict)

    glove_serial = os.path.join(trained_glove_dir, "{}d_accept_{}_Vocab_{}.pk".format(num_components, vocab_size, glove_type))

    with open(glove_serial, 'wb') as write_file:
        dill.dump( combined_dict, write_file)



    glove_excel = os.path.join(trained_glove_dir ,"{}d_accept_{}_Vocab_{}.xlsx".format(num_components, vocab_size, glove_type))

    glove_df = pd.DataFrame.from_dict(combined_dict, orient="index")

    with pd.ExcelWriter(glove_excel) as xlsx_writer:
        glove_df.to_excel(xlsx_writer, "Glove_{}_{}_{}".format(vocab_size, num_components, glove_type), header=True, index_label= False)
        xlsx_writer.save()




train_glove()
load_glove_accept()
