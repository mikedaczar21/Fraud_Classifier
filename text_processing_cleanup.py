# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import dill
import os
import nltk
import random
from joblib import dump, load
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from uszipcode import SearchEngine   # Used to get geolocation information from zipcode
from symspellpy.symspellpy import SymSpell  # Library to clean up misspelled words

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import re
from collections import OrderedDict

#from data_feature_functions import get_Fraud_Dataset

current_dir = os.path.abspath(os.path.curdir)
initial_data_dir = os.path.abspath(os.path.join(current_dir, "Initial_Datasets"))
feature_gen_data_dir = os.path.abspath(os.path.join(current_dir, "Feature_Generated_Datasets"))
train_test_dir = os.path.abspath(os.path.join(current_dir, "Train_Test_Data"))
prediction_dir = os.path.abspath(os.path.join(current_dir, "Predictions"))
word_vec_dir = os.path.abspath(os.path.join(current_dir, "Word_Vectors"))
vocab_dir = os.path.abspath(os.path.join(word_vec_dir, "Vocab"))

dictionary_path = os.path.join(vocab_dir, "frequency_dictionary_english.txt")
text_feature_file_serialized = "text_feature_fraud.pk"
initial_data_file = "MasterFraudData_more datav2.xlsx"
new_feature_gen_file = "Fraud_Data_Features_Generated.xlsx"

# Parameters for SysSpell
max_edit_distance_dict = 2
prefix_length = 7
dict_term_index = 0
dict_count_index = 1

replace_acronym_dict = {
                            'O/D': 'other driver',
                            'O/V': 'other vehicle',
                            'O/P': 'other person',
                            'I/V': 'insured vehicle',
                            'OVD': 'other vehicle driver',
                            'T/P/O': 'third party other',
                            'TOTAL LOSS': 'vehicle completely totalled',
                            'PI': 'person insured',
                            'P2': 'person two',
                            'CLMT': 'claimant',
                            'CLMT''S': 'claimant''s',
                            'clmts'.upper() : 'claimant''s',
                            'DOL': 'date of loss is',
                            'P/S': 'person site',
                            'D/S': 'damaged site',
                            'INJURYWHEN': 'injury when',
                            'claimantonly'.upper(): 'claimant only',
                            'Claimantand'.upper(): 'claimant and',
                            'themold.the'.upper(): 'the mold the',
                            'moldwas'.upper() : 'mold was',
                            'wearinghis'.upper(): 'wearing his',
                            'disregardedtraffic'.upper(): 'disregarded traffic',
                            'thepickup'.upper(): 'the pickup',
                            'laneswhen'.upper(): 'lanes when',
                            'theright'.upper(): 'the right',
                            'intoa'.upper(): 'into a',
                            'laid-off'.upper(): 'discharged',
                            'lay - off'.upper(): 'discharge',
                            'ANDFELL': 'and fell',
                            'HOSP': 'hospital',
                            'HOSP.': 'hospital',
                            'MRI': 'magnetic resonance imaging',
                            'MEN''S': 'men''s',
                            'WOMEN''S': 'women''s',
                            'PLTF': 'plaintiff',
                            'PLAINTIF': 'plaintiff',
                            'TPV': 'third party vehicle',
                            'TPV1': 'third party vehicle one',
                            'TPV2': 'third party vehicle two',
                            'TPV3': 'third party vehicle three',
                            'CV' : 'claimant vehicle',
                            'CV1': 'first claimant vehicle',
                            'CV2': 'second claimant vehicle',
                            'CD' : 'claimant driver',
                            'CVD' : 'claimant vehicle driver',
                            'CUST': 'customer',
                            'IV' : 'insured vehicle',
                            'IVs': 'insured vehicle',
                            'IVwas'.upper(): 'insured vehicle was',
                            'IV1' : 'insured vehicle one',
                            'IV2' : 'insured vehicle two',
                            'IV3' : 'insured vehicle three',
                            'IVD': 'insured vehicle driver',
                            'INSD' : 'insured person',
                            'MPH': 'miles per hour',
                            'BREKE': 'brake',
                            'LITE' : 'light',
                            'TEH': 'the',
                            'SHAW': 'saw',
                            'OV' : 'other vehicle',
                            'OV1': 'first other vehicle',
                            'OV2': 'second other vehicle',
                            'OV3': 'third other vehicle',
                            '1ST': 'first',
                            '2ND': 'second',
                            '3RD': 'third',
                            '4TH': 'third',
                            'ID': 'insured driver',
                            'LR': 'left rear',
                            'FLR': 'floor',
                            'DIV': 'driver of insured vehicle',
                            'DMGS':  'Damages',
                            'DUI': 'driving under the influence',
                            'TP' : 'thirdy party',
                            'BI' : 'bodily injury',
                            'IP' : 'insured person',
                            'IVD' : 'insured vehicle driver',
                            'REPO''D': 'repossessed',
                            'REPO': 'repossessed',
                            'EC' : '',
                            'HWY': 'highway',
                            'LOC:': 'location is ',
                            'ATV' : 'all terrain vehicle',
                            'APT': 'apartment',
                            'N.E.': 'north east',
                            'R/T': 'right turn',
                            'L/T': 'left turn',
                            'NYCC': 'New York Communities for Change',
                            'PVC': 'polyvinyl chloride',
                            'W/B': 'westbound',
                            'WB': 'westbound',
                            'EB': 'eastbound',
                            'E/B': 'eastbound',
                            'SB': 'southbound',
                            'NB': 'northbound',
                            'N/B': 'northbound',
                            'EMP': 'employee',
                            'EE' : 'employed employee',
                            'CLMT': 'claimant',
                            'CLMNT' : 'claimant',
                            'CLT' : 'claimant',
                            'DIV' : 'driver',
                            'REPO''D': 'repossessed',
                            'RD': 'road',
                            'VEH': 'vehicle',
                            'VEHS': 'vehicles',
                            'VEH1' : 'vehicle one',
                            'V1': 'vehicle one',
                            'V#1': 'vehicle one',
                            'CV#1': 'claimant vehicle one',
                            'V2': 'vehicle two',
                            'V#2': 'vehicle two',
                            'CV#2': 'claimant vehicle two',
                            'V3': 'vehicle three',
                            'CV#3': 'claimant vehicle three',
                            'V4': ' vehicle four',
                            'V5': 'vehicle five',
                            'V6': 'vehicle six',
                            'V7': 'vehicle seven',
                            'V8': 'vehicle eight',
                            'D1': 'driver one',
                            'D2': 'driver two',
                            'D3': 'driver three',
                            'SUV': 'sport utility vehicles',
                            'VEH2': 'vehicle two',
                            'VEH3': 'vehicle three',
                            'RE' : 'rear ended',
                            'R/E': 'rear ended',
                            'TYRE': 'tire',
                            'MI': 'miles',
                            'TT' : '',
                            'TPD': 'third party driver',
                            'INC': 'incorporated',
                            'IC': '',
                            'FNOL': '',
                            'ADP' : '',
                            'ASAP': 'as soon as possible',
                            'PD': 'police department',
                            'III': 'three',
                            'ASST ': 'assissant',
                            'MGR': 'manager',
                            'DOT': 'department of transportation',
                            'FT': 'feet',
                            'yd': 'yard',
                            'I-5': 'interstate 5',
                            'I-9': 'interstate 9',
                            'I-10': 'interstate 10',
                            'I20': 'interstate 20',
                            'I-20': 'interstate 20',
                            'I-55': 'interstate 55',
                            'I75': 'interstate 75',
                            'I-25': 'interstate 25',
                            'I-45': 'interstate 45',
                            'I495': 'interstate 495',
                            'I75': 'interstate 75',
                            'I78': 'interstate 78',
                            'I-80': 'interstate 80',
                            'I-95': 'interstate 95',
                            'I95': 'interstate 95',
                            '49TH': 'forty-nineth',
                            'yr': 'year',
                            'yrs': 'years',
                            'NaT': '',
                            'L': 'left',
                            'R': 'right',
                            'E': 'east',
                            'W': 'west'


}


def create_SysSpell_Object():

    # creating SymSpell object to correct spelling errors
    sys_spell = SymSpell(max_edit_distance_dict, prefix_length)

    # load dictionary
    if not sys_spell.load_dictionary(dictionary_path, dict_term_index, dict_count_index):
        print("Dictionary not found\n.")
        return


    return sys_spell





def correcting_Spelling(input_Sentence):

    symbols_to_remove = '''!()-[]{};,:'"\,<>.?@#$%^&*_~'''

    sys_spell = create_SysSpell_Object()

    sentence_split = input_Sentence.lower().split(" ")
    corrected_words_list = []
    print("Letter split {}".format(sentence_split))
    for word in sentence_split:


        if (word.isalpha() == True) and (len(word) > 2):

            word = word.replace(symbols_to_remove, "").replace("nan", "")

            correct_spell_suggestions = sys_spell.lookup_compound(word, 2)
            corrected_word = correct_spell_suggestions[0]

            print("\nCurrent word: {0}".format(word))
            for correct_sugg in correct_spell_suggestions:
                print(" and Suggested Word {1}\n".format(word, correct_sugg))

            corrected_words_list.append(str(corrected_word))

        else:

            corrected_words_list.append(word)


    return corrected_words_list



# Helper function to replace acronyms with the words they represent
def replace_str(mainStr):


    for oldStr, newStr in replace_acronym_dict.items():

        if oldStr in mainStr:

            mainStr = re.sub(r'\b{0}\b'.format(oldStr), newStr, mainStr)


    return mainStr

# Helper function for replacing with any given dict
def replace_str_general(mainStr, acronym_dict):


    for oldStr, newStr in acronym_dict.items():

        if oldStr in mainStr:

            mainStr = re.sub(r'\b{0}\b'.format(oldStr), newStr, mainStr)


    return mainStr



def get_Vocab_Set(text, recreate_Vocab_Set):

    current_vocab_path = os.path.join(vocab_dir, "current_Vocab_Set.pk")
    current_vocab_xlsx_path =  os.path.join(vocab_dir, "current_Vocab_Set.xlsx")

    # creating SymSpell object to correct spelling errors
    sys_spell = create_SysSpell_Object()


    if (os.path.exists(current_vocab_path)) and (recreate_Vocab_Set == False):
        with open(current_vocab_path, 'rb') as read:
            vocab_set = dill.load(read)
    else:

        if (os.path.exists(current_vocab_path)) and (recreate_Vocab_Set == True):
            os.remove(current_vocab_path)

        vocab_set = set()

        # loading feature text data
        text_serialized_path = os.path.join(text_feature_path, text_feature_file_name)


        if (os.path.exists(text_serialized_path)):
            # Loading serialized text feature
            with open(text_serialized_path, 'rb') as read:
                text = dill.load(read)
        else:
            fraud_data = text_processing_cleanup.get_Fraud_Dataset(recreate_features=False)
            text = fraud_data['Fraud_Text']

        symbols_to_remove = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        for sentence in list(text):
            words_replaced_acryonms = replace_str(sentence)

            print("\n Before Correction string: {} \n".format(words_replaced_acryonms))
            correct_spelling_result = sys_spell.word_segmentation(sentence)
            words_correct_spelling = correct_spelling_result.corrected_string
            print("\n Corrected string:  {}\n".format(words_correct_spelling))

            words_list = words_correct_spelling.split(" ")
            words_list_symbol_replaced = [elem.replace(symbols_to_remove, " ") for elem in words_list]




            [
                vocab_set.add(elem.lower())
                    for elem in words_list_symbol_replaced
                        if elem not in vocab_set and elem.isalpha() == True and len(elem) > 2
            ]

        text_feature_df = pd.DataFrame(vocab_set, columns= ['New_Fraud_Text'])
        with pd.ExcelWriter(current_vocab_xlsx_path) as xlsx_writer:
            text_feature_df.to_excel(xlsx_writer, 'Vocab Set', header=True, index_label= False)
            xlsx_writer.save()

        # Serializnig the vocab set
        with open(current_vocab_path, 'wb') as write:
            dill.dump(vocab_set, write)


    return vocab_set



# Helper function which looks for words within words and splits them by adding a space in between
# At the moment this function will split two words, but it can be called multiple times to split multiple words

# Helper function which looks for words within words and splits them by adding a space in between
# At the moment this function will split two words, but it can be called multiple times to split multiple words
def split_words_within_words(vocab_List, **kwargs):

    vocab_Set = kwargs['input_Vocab_Set']

    if len(vocab_Set) > 0:
        current_Vocab_Set = vocab_Set
        vocab_List = list(vocab_Set)
    else:
        current_Vocab_Set = set([])

    asc_Vocab_List =  sorted(vocab_List, key=len)

    # soring the list based on length (so the words with the least number of characters go first)
    dsc_Vocab_List =  sorted(vocab_List, key=len, reverse=True)

    dict = {}
    # loop to create ordered dictionary
    for index, elem in enumerate(vocab_List):
        dict[index] = elem

    dsc_Orderd_Dict = OrderedDict(sorted(dict.items(), key = lambda val : len(val[1]), reverse = True))
    starting_small_word_index = len(dsc_Orderd_Dict)

    for key, elem in dsc_Orderd_Dict.items():


        current_smaller_word = dsc_Orderd_Dict[starting_small_word_index-key]


        if len(current_smaller_word) > 2 and (current_smaller_word.isalpha() == True):
            wordIndex = elem.find(current_smaller_word)
            current_word_len = len(current_smaller_word)

            if wordIndex != -1:

                # match is made for word on the left
                if (wordIndex > 0) and  (elem != current_smaller_word):

                    # checking if there is alphabetical character to left of the index of the word which was found
                    if ( elem[wordIndex - 1].isalpha() == True):

                        left_word = elem[0:wordIndex]
                        right_word = elem[wordIndex:len(elem)+1]

                        current_Vocab_Set.add(left_word)
                        current_Vocab_Set.add(right_word)
                        dsc_Orderd_Dict[key] = right_word

                        dsc_Last_Index = len(dsc_Orderd_Dict)+1
                        if (right_word not in current_Vocab_Set):
                            dsc_Orderd_Dict[dsc_Last_Index] = left_word

                        # dsc_Orderd_Dict = OrderedDict(sorted(dsc_Orderd_Dict.items(), key = lambda val : len(val[1]), reverse = True))




                # Match is made for word on the left
                elif (wordIndex == 0) and (elem != word) :

                    if (elem[wordIndex+current_word_len].isalpha() == True):


                        left_word = elem[0:current_word_len]
                        right_word = elem[current_word_len:len(elem)+1]

                        current_Vocab_Set.add(left_word)
                        current_Vocab_Set.add(right_word)

                        dsc_Orderd_Dict[key] = left_word
                        dsc_Last_Index = len(dsc_Orderd_Dict)+1

                        if (right_word not in current_Vocab_Set):
                            dsc_Orderd_Dict[dsc_Last_Index] = right_word

                        # dsc_Orderd_Dict = OrderedDict(sorted(dsc_Orderd_Dict.items(), key = lambda val : len(val[1]), reverse = True))







    return_dict = set(  word for word in dsc_Orderd_Dict.values())

    return current_Vocab_Set, return_dict


def find_Max_Sentence_Length(input_2d_TextArray):

    sentence_length = []
    sentence_max = 0
    longest_sentence = dict([])
    for sentence in input_2d_TextArray:

        words = sentence.split(",")
        sentence_max = 0
        for current_word in words:

            if ( current_word.isalpha() == True):
                sentence_max += 1

            if (current_word == words[-1]):
                sentence_length.append(sentence_max)
                longest_sentence[sentence_max] = words

    # --- Testing print ------
    # print("Sentence Max : {0}\n Longest Setence: \n{1}\n".format(max(sentence_length), longest_sentence[max(sentence_length)]) )


    return max(sentence_length),longest_sentence



def text_Processing(passedString, numbers):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """

    symbols_to_remove = '''!()-[]{};,:'"\,<>.?@#$%^&*_~-'''

    sentence_sym_remove =  re.sub(r"[\-#!\.//():,]", " ",passedString)


    #letterText = re.sub("[^a-zA-Z0-9.:\]", " ", passedString )

    if numbers == False:
        letterText = re.sub("[^a-zA-Z]", " ", sentence_sym_remove )
    else:
        letterText = re.sub("[^a-zA-Z0-9]", " ", passedString )

    letter_split = letterText.lower().split(" ")

    words_list_symbol_replaced = [elem.replace(symbols_to_remove, "") for elem in letter_split]

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in words_list_symbol_replaced ]

    stopWords = set(stopwords.words('english'))

    cleanData_stopWords = [word for word in stemmed_words if not word in stopWords if word != '' and word != r'\s']

    cleanData_string = ", ".join(cleanData_stopWords)

    return cleanData_string





def text_Processing_GloVe(passedString):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """


    symbols_to_remove = '''!()-[]{};,:'"\,<>.?@#$%^&*_~-'''


    passedString =  passedString.replace("NULL", " ")

    # passedString_acyronm_replaced = replace_str(passedString)
    #

    passedString_acyronm_replaced = replace_str(passedString)

    sentence_sym_remove =  re.sub(r"[\-#!\.//():]", " ", passedString_acyronm_replaced)

    letterText = re.sub("[^a-zA-Z]", " ", sentence_sym_remove  )

    # letterText = re.sub("[^a-zA-Z0-9]", " ", passedString )

    letter_split = letterText.lower().split(" ")

    # Finding unique words
    letter_split_set = set(letter_split)

    letter_split_unique = list(letter_split)

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in letter_split_unique]

    stopWords = set(stopwords.words('english'))

    cleanData_stopWords = [ str(word)  for word in stemmed_words if not word in stopWords if word != '' and word != r'\s' ]

    cleanData_string = ", ".join(cleanData_stopWords)

    return cleanData_stopWords




def text_Create_Glove_Corpus(passedString):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """
    # creating SymSpell object to correct spelling errors
    sys_spell = create_SysSpell_Object()


    vocab_set = get_Vocab_Set(text = passedString,  recreate_Vocab_Set = True)

    symbols_to_remove = '''!()-[]{};,:'"\,<>./?@#$%^&*_~'''

    passedString =  passedString.replace("NULL", " ")

    passedString_acyronm_replaced = replace_str(passedString)

    print("\n Before Correction string: {} \n".format(passedString_acyronm_replaced))
    correct_spelling_result = sys_spell.word_segmentation(passedString_acyronm_replaced)
    words_correct_spelling = correct_spelling_result.corrected_string
    print("\n Corrected string:  {}\n".format(words_correct_spelling))


    letterText = re.sub("[^a-zA-Z]", " ", words_correct_spelling)

    wordsList = letterText.lower().split(" ")

    words_list_symbol_replaced = [elem.replace(symbols_to_remove, "") for elem in wordsList]

    # print("Words List: ")
    # print(wordsList)



    # print("\n Vocab List: {}\n".format(vocab_List) )

    words_list_split = split_words_within_words(wordsList, input_Vocab_Set =vocab_set)

    print("\nwords first split: {}\n".format(words_list_split))


    stemmer = SnowballStemmer('english')


    # stemmed_words = [stemmer.stem(word) for word in wordsList]


    stemmed_words = [
                        [

                            stemmer.stem(word)
                            for word in sentence
                        ]
                                for sentence in words_list_split
                    ]

    # stemmed_words = [
    #                     stemmer.stem(word)
    #                         for word in words_list_split
    #                 ]


    print("Stemmed Words: {0}".format(stemmed_words))
    # print(stemmed_words)

    stopWords = set(stopwords.words('english'))

    cleanData_stopWords = [     [
                                    word for word in elem
                                        if not word in stopWords
                                ]
                                    for elem in stemmed_words
                          ]

    print("stop words cleared {0}: \n".format(cleanData_stopWords))
    print(cleanData_stopWords)


    cleanData_string = ""
    for words in cleanData_stopWords:
        cleanData_string += ",".join(words)


    # print("Clean Data String (after join): {0} \n".format())

    return cleanData_string



def text_Processing_Old(passedString):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """



    passedString =  passedString.replace("NULL", " ")

    letterText = re.sub("[^a-zA-Z0-9]", " ", passedString)

    wordsList = letterText.lower().split(" ")

    # print("Words List: ")
    # print(wordsList)
    #
    # vocab_List = list(create_vocab_set(wordsList, input_Vocab_Set = vocab_Set))
    #
    # print(vocab_List)
    #
    # words_list_split = list(split_words_within_words(wordsList, input_Vocab_Set = {}))
    #
    # print("words first split")
    # print(words_list_split)

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in wordsList]

    print("Stemmed Words: ")
    print(stemmed_words)

    stopWords = set(stopwords.words('english'))

    cleanData_stopWords = [word for word in stemmed_words if not word in stopWords]

    print("stop words cleared: \n")
    print(cleanData_stopWords)


    cleanData_string = ",".join(cleanData_stopWords)

    return cleanData_string
