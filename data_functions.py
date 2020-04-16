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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from uszipcode import SearchEngine   # Used to get geolocation information from zipcode

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import re
from collections import OrderedDict



current_dir = os.path.abspath(os.path.curdir)
initial_data_dir = os.path.abspath(os.path.join(current_dir, "Initial_Datasets"))
feature_gen_data_dir = os.path.abspath(os.path.join(current_dir, "Feature_Generated_Datasets"))
train_test_dir = os.path.abspath(os.path.join(current_dir, "Train_Test_Data"))
prediction_dir = os.path.abspath(os.path.join(current_dir, "Predictions"))
word_vec_dir = os.path.abspath(os.path.join(current_dir, "Word_Vectors"))
vocab_dir = os.path.abspath(os.path.join(word_vec_dir, "Vocab"))
geo_data_dir = os.path.abspath(os.path.join(feature_gen_data_dir, "Geolocation_Data"))

text_feature_path = os.path.abspath( os.path.join(feature_gen_data_dir, "Text_Data") )
text_feature_file_name = "text_feature_fraud.pk"

initial_data_file = "MasterFraudData_more datav2.xlsx"
new_feature_gen_file = "Fraud_Data_Features_Generated.xlsx"
num_classes = 2 # Number of different classes that is used for label


replace_acronym_dict = {
                            'O/D': 'other driver',
                            'O/V': 'other vehicle',
                            'O/P': 'other person',
                            'T/P/O': '',
                            'TOTAL LOSS': 'vehicle completely totalled',
                            'pltf': 'plaintiff',
                            'plaintif': 'plaintiff',
                            'TPV': 'third party vehicle',
                            'TPV1': 'third party vehicle number 1',
                            'TPV2': 'third party vehicle number 2',
                            'CV' : 'claimant vehicle',
                            'CV1': 'first claimant vehicle',
                            'CV2': 'second claimant vehicle',
                            'CD' : 'claimant driver',
                            'CVD' : 'claimant vehicle driver',
                            'CUST': 'customer',
                            'IV' : 'insured vehicle',
                            'Insd' : 'insured',
                            'BREKE': 'brake',
                            'LITE' : 'light',
                            'OV' : 'other vehicle',
                            'OV1': 'first other vehicle',
                            'OV2': 'second other vehicle',
                            'OV3': 'third other vehicle',
                            'ID': 'insured driver',
                            'DIV': 'driver of insured vehicle',
                            'Dmgs':  'Damages',
                            'DUI': 'driving under the influence',
                            'TP' : 'thirdy party',
                            'BI' : 'back injury',
                            'IP' : 'insured person',
                            'IVD' : 'insured vehicle driver',
                            'EC' : '',
                            'HWY': 'highway',
                            'Loc:': 'location is ',
                            'ATV' : 'all terrain vehicle',
                            'APT': 'apartment',
                            'L': 'left',
                            'R': 'right',
                            'E': 'east',
                            'W': 'west',
                            'N.E.': 'north east',
                            'R/T': 'right turn',
                            'NYCC': 'New York Communities for Change',
                            'PVC': 'polyvinyl chloride',
                            'WB': 'west bound',
                            'EB': 'east bound',
                            'SB': 'south bound',
                            'EMP': 'employee',
                            'EE' : 'employed employee',
                            'CLMT': 'claimant',
                            'clmnt' : 'claimant',
                            'clt' : 'claimant',
                            'DIV' : 'driver',
                            'REPO''D': 'repossessed',
                            'D1' : 'driver number 1',
                            'VEH': 'vehicle',
                            'VEH1' : 'first vehicle',
                            'V1': 'first vehicle',
                            'V2': 'second vehicle',
                            'V3': 'third vehicle',
                            'SUV': 'sport utility vehicles',
                            'VEH2': 'vehicle 2',
                            'RE' : 'rear ended',
                            'Tyre': 'tire',
                            'mi': 'miles',
                            'TT' : '',
                            'TPD': 'third party driver',
                            'IC': '',
                            'FNOL': '',
                            'ADP' : '',
                            'ASAP': 'as soon as possible',
                            'PD': 'police department'


    }





def create_indicator_matrix(label_Vector):

    num_labels = len(label_Vector)
    target_matrix = np.zeros( (num_labels, num_classes) )

    # creating one hot encoded matrix
    for index in range(num_labels):
        target_matrix[index, label_Vector[index]] = 1

    return target_matrix



# Helper function which takes in zip codes
def get_Geolocation_Data(zip_Code_Data):

    geo_data_path = os.path.join(geo_data_dir, "current_geo_data.pk")
    geo_search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database

    df_complete_geo_data = pd.DataFrame(columns = [ 'Claim Number' , 'Longitude', 'Latitude'])
    geo_list_complete = []
    print("Type of zip code data: {}".format(type(zip_Code_Data)))


    if (os.path.exists(geo_data_path)):
        with open(geo_data_path, 'rb') as read:
            df_complete_geo_data = dill.load(read)

    else:


        for current_index, current_row in zip_Code_Data.iterrows():
            # print("Zip type: {} \n".format(type(zip)) )
            # print("Current zip {}\n".format(zip))
            # print("Index: {}\n".format(current_index))
            current_zip_code = current_row['LOSS_LOCATION_ZIP']
            current_claim_num = current_row['Claim Number']

            # if (type(current_zip_code) == 'str') and (current_zip_code != '99999'):
            if (isinstance(current_zip_code, str) == True ):

                # print("Current Zip Code {}".format(current_zip_code))
                if ("99999" not in current_zip_code) and (len(current_zip_code) == 5) and (current_zip_code.strip() != "99999") and (current_zip_code.isalpha() == False):

                    # print("Current Zip Code {}".format(current_zip_code))
                    # print("Current zip {}\n".format(current_zip_code))
                    # print("Index: {}\n".format(current_index))


                    geo_info = geo_search.by_zipcode(current_zip_code)
                    geo_loc_dict = geo_info.to_dict()


                    geo_loc_lng = geo_loc_dict['lng']
                    geo_loc_lat = geo_loc_dict['lat']


                    current_geo_data = [ current_claim_num,  geo_loc_lng, geo_loc_lat]

                    # if (geo_loc_lng is not None) and (geo_loc_lat is not None):
                    #     geo_list_complete.append(current_geo_data)
                    geo_list_complete.append(current_geo_data)


            if (current_index == (len(zip_Code_Data) - 1) ):
                print("Current Index: {0}, Len of Zip Code Col: {1}".format(current_index, len(zip_Code_Data)))
                df_complete_geo_data = pd.DataFrame(geo_list_complete, columns = [ 'Claim Number' , 'Longitude', 'Latitude'] )

                # remove junk rows
                # null_geoloc_rows = df_complete_geo_data[(df_complete_geo_data['Longitude'] == 'Nan') ].index
                # null_geoloc_rows.append( df_complete_geo_data[ df_complete_geo_data['Latitude'] == 'NaN'].index )
                # df_complete_geo_data.drop(null_geoloc_rows, inplace = True)


                with open(geo_data_path, 'wb') as write:
                    dill.dump(df_complete_geo_data, write)

    #print("Geo List Complete:  {0}\n".format(geo_list_complete) )

    return df_complete_geo_data, geo_list_complete











def get_Fraud_Dataset(**kwargs):

    recreate_features  = kwargs['recreate_features']

    init_data_path = os.path.abspath(os.path.join(current_dir, initial_data_dir))
    fraud_data_path = os.path.join(init_data_path, initial_data_file)

    feature_gen_data_path = os.path.join(feature_gen_data_dir, new_feature_gen_file)
    text_feature_fraud_path = os.path.join(text_feature_path, text_feature_file_name)

    print("Initial data path: {0} \n Local Fraud Data Path: {1} \n Feature Gen Path: {2}".format(init_data_path, fraud_data_path, feature_gen_data_path))




    if os.path.exists(feature_gen_data_path) and (recreate_features  == False):

        feature_data_df = pd.read_excel(feature_gen_data_path)

    # Otherwise use pandas to load in excel file and only keep the important columns
    else:

        # Checking if flag to recreate features is set to true and if so then delete feature files
        if (recreate_features == True) :

            if (os.path.exists(feature_gen_data_path)):
                os.remove(feature_gen_data_path)

            if (os.path.exists(text_feature_fraud_path)):
                os.remove(text_feature_fraud_path)

        # loop to generate list of indices which will be used to keep certain columsn
        col_keep_indices = []
        for col_index in range(18):
            col_keep_indices.append(col_index)

        # Grabbing the dataset from excel (only first 18 columns)
        initial_large_data_df = pd.read_excel(fraud_data_path, usecols = col_keep_indices,
                                 na_values=["NULL", '', "", ' ', "N/A", "null"],
                                 dtype = {
                                          'Fraud Acceptance': str,
                                          'Loss_Description': str,
                                          'Loss  City': str, 'LOSS_LOCATION_STATE': str,
                                          'LOSS_LOCATION_ZIP': str,
                                          'Loss Date': np.datetime64, 'Loss Time': np.datetime64,
                                          'Main Cause': str, 'Sub Cause': str,
                                          'Policy Eff Date': np.datetime64, 'Policy Exp Date': np.datetime64,
                                          'Claim Create Date': np.datetime64, 'Claim Receipt Date':  np.datetime64
                                        }
                                )

        # Dropping rows that are labelled 'not applicable', only focusing on 'accepted' and 'rejected' labels
        # not_applicable_rows = initial_large_data_df[initial_large_data_df['Fraud Acceptance'] == 'Not Applicable'].index
        # initial_large_data_df.drop(not_applicable_rows, inplace=True)

        initial_large_data_df = initial_large_data_df.fillna("NULL")

        # Grabbing zip code data from large dataframe
        zip_codes = pd.concat( [initial_large_data_df['Claim Number'] , initial_large_data_df['LOSS_LOCATION_ZIP']], axis = 1)

        # Getting geolocation dataframe with helper function
        new_geoloc_data, geo_list = get_Geolocation_Data(zip_codes)

        # Merging large dataframe with
        initial_large_data_df = pd.merge(initial_large_data_df, new_geoloc_data , on='Claim Number')

        # generating features with helper function
        feature_data_df = feature_Generation_Init_Data(initial_large_data_df)



        with pd.ExcelWriter(feature_gen_data_path) as xlsx_writer:
            feature_data_df.to_excel(xlsx_writer, 'Fraud_Data_Features_Generated', header=True, index_label= False)
            xlsx_writer.save()



        with open(text_feature_fraud_path, 'wb') as write:
            dill.dump(feature_data_df['Fraud_Text'], write)



    return feature_data_df


def feature_Generation_Init_Data(initial_Dateset):

    new_text_feature_df = pd.DataFrame( columns = ['Claim Number', 'New_Loss_Descrip'])
    new_text_list = []
    # iterating through initial dataframe
    for index, row in initial_Dateset.iterrows():


        # print("Current loss descrip: {}".format(row["Loss_Description"]))
        # adding injury damage and body part description to text feature if it is not null
        if row["Injury Damage"].lower().find("null") == -1:
            row["Loss_Description"] +=  " , injury damage is "   + row["Injury Damage"]



        # adding  body part description to text feature if it is not null
        if row["Body Part Description"].lower().find("null") == -1 :
            row["Loss_Description"] +=  " , the following body part is injured: "   + row["Body Part Description"]



        # adding main/sub cause to text feature if it is not already included and if it is not null
        if (row["Main Cause"] != 'nan') and (row["Loss_Description"].find(row["Main Cause"]) == -1):

            # This case will add both sub and main causes to text feature since neither is in the loss descrip
            if (row["Sub Cause"] != 'nan') and (row["Loss_Description"].find(row["Sub Cause"]) == -1 ) and ( row["Sub Cause"].lower().find("other") == -1):
                row["Loss_Description"] += row["Main Cause"] + " - " + row["Sub Cause"] + " - " + row["Loss_Description"]


            # This case will only need to add the main cause
            elif (row["Sub Cause"] != 'nan') and (row["Sub Cause"] in row["Loss_Description"]) :
                row["Loss_Description"] += row["Main Cause"] + " - " + row["Sub Cause"] + " - " + row["Loss_Description"]



        # Adding loss datetime data to text feature
        if(str(row["Loss Time"]).find("NULL") == -1):

            if (row["Loss Time"] != 'nan'):

                row["Loss_Description"] +=  ", the loss occurred at the following date and time: " + str(row["Loss Time"])


            elif (row["Loss Time"] == 'nan'):
                row["Loss_Description"]  +=  ", the loss occurred at the following date: " + str( row["Loss Date"] )


        # Adding geolocation data to text feature
        if (row["Longitude"] != 'NaN') or (row["Latitude"] != 'NaN'):
            row["Loss_Description"] +=  ", the loss occurred at the following longitude: " + str( row["Longitude"] ) + ", and latitude: " + str(row["Latitude"])


        current_text = [ row["Claim Number"], row["Loss_Description"] ]
        # print("Current Loss Descrip: {}\n".format(current_text))

        new_text_list.append(current_text)




    new_text_feature_df = pd.DataFrame(new_text_list, columns =  ['Claim Number', 'New_Loss_Descrip'])
    # print("\n New Text Feature: \n{0}".format(new_text_feature_df))

    #Getting rid of these columns since the data is added to text feature
    col_exclude_list = ['Main Cause', 'Sub Cause', 'Injury Damage', 'Body Part Description', 'LOSS_LOCATION_ZIP']
    col_keep_list = [col for col in initial_Dateset.columns if col not in col_exclude_list]


    filtered_dataset = initial_Dateset[col_keep_list]

    # new_feature_df = pd.concat([filtered_dataset, new_text_feature_df], axis=1)
    new_feature_df = pd.merge(filtered_dataset, new_text_feature_df, on='Claim Number')




    new_feature_df.rename( columns = {
                                        'Fraud Acceptance': 'Fraud_Label',
                                        'New_Loss_Descrip': 'Fraud_Text'
                                    },
                        inplace=True)




    return new_feature_df





# Helper function to replace acronyms with the words they represent
def replace_str(mainStr):


    for oldStr, newStr in replace_acronym_dict.items():

        if oldStr in mainStr:

            mainStr = re.sub(r'\b{0}\b'.format(oldStr), newStr, mainStr)


    return mainStr


# Helper function which looks for words within words and splits them by adding a space in between
# At the moment this function will split two words, but it can be called multiple times to split multiple words
def split_words_within_words_old(vocab_List, **kwargs):

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

    for key, elem in dsc_Orderd_Dict.items():

        for word in asc_Vocab_List:

            if len(word) > 1 and (word.isalpha() == True):
                wordIndex = elem.find(word)
                current_word_len = len(word)

                if wordIndex != -1:

                    # match is made for word on the left
                    if (wordIndex > 0) and  (elem != word):

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

                            dsc_Orderd_Dict = OrderedDict(sorted(dsc_Orderd_Dict.items(), key = lambda val : len(val[1]), reverse = True))

                            asc_Vocab_List.append(left_word)
                            asc_Vocab_List =  sorted(vocab_List, key=len)


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

                            dsc_Orderd_Dict = OrderedDict(sorted(dsc_Orderd_Dict.items(), key = lambda val : len(val[1]), reverse = True))

                            asc_Vocab_List.append(right_word)
                            asc_Vocab_List =  sorted(vocab_List, key=len)





    return_dict = set(  word for word in dsc_Orderd_Dict.values())

    return current_Vocab_Set, return_dict






def create_Vocab_Set(text):

    current_vocab_path = os.path.join(vocab_dir, "current_Vocab_Set.pk")
    current_vocab_xlsx_path =  os.path.join(vocab_dir, "current_Vocab_Set.xlsx")

    if (os.path.exists(current_vocab_path)):
        with open(current_vocab_path, 'rb') as read:
            vocab_set = dill.load(read)
    else:

        vocab_set = set()

        # loading feature text data
        text_serialized_path = os.path.join(text_feature_path, text_feature_file_name)


        if (os.path.exists(text_serialized_path)):
            # Loading serialized text feature
            with open(text_serialized_path, 'rb') as read:
                text = dill.load(read)
        else:
            fraud_data = get_Fraud_Dataset()
            text = fraud_data['Fraud_Text']

        symbols_to_remove = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        for sentence in list(text):
            words_replaced_acryonms = replace_str(sentence)
            words_list = words_replaced_acryonms.split(" ")
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





def text_Processing_Create_Glove_Corpus(passedString):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """

    vocab_set = create_Vocab_Set(passedString)

    symbols_to_remove = '''!()-[]{};,:'"\,<>./?@#$%^&*_~'''

    passedString =  passedString.replace("NULL", " ")

    passedString_acyronm_replaced = replace_str(passedString)


    letterText = re.sub("[^a-zA-Z]", " ", passedString_acyronm_replaced)

    wordsList = letterText.lower().split(" ")

    words_list_symbol_replaced = [elem.replace(symbols_to_remove, "") for elem in wordsList]

    # print("Words List: ")
    # print(wordsList)
    #
    vocab_List = create_Vocab_Set(words_list_symbol_replaced)

    # print("\n Vocab List: {}\n".format(vocab_List) )

    words_list_split = split_words_within_words(wordsList, input_Vocab_Set = vocab_List)

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


def text_Processing(passedString):

    """
    Takes in string and returns list
    1. Replaces the acronyms in text with the phrases they stand for
    2. Split any words that are lumped together (such as 'reportedthat' and 'hitsomething')
    3. Remove none letters and change leters to lowercase
    4. Remove stop words (such as 'the', 'and', 'a')
    5. Remove similar stemmed words (SnowballStemmer)
    6. Return list of cleaned text words
    """



    vocab_set = create_Vocab_Set(passedString)

    passedString =  passedString.replace("NULL", " ")

    passedString_acyronm_replaced = replace_str(passedString)

    letterText = re.sub("[^a-zA-Z0-9]", " ", passedString_acyronm_replaced)

    wordsList = letterText.lower().split(" ")


    # print("Words List: ")
    # print(wordsList)
    #
    # vocab_List = list(create_Vocab_Set(wordsList, input_Vocab_Set = vocab_Set))
    #
    # print(vocab_List)
    #
    # words_list_split = list(split_words_within_words(wordsList, input_Vocab_Set = {}))
    #
    # print("words first split")
    # print(words_list_split)

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in wordsList]

    # print("Stemmed Words: ")
    # print(stemmed_words)

    stopWords = set(stopwords.words('english'))

    cleanData_stopWords = [word for word in stemmed_words if not word in stopWords]

    # print("stop words cleared: \n")
    # print(cleanData_stopWords)


    cleanData_string = ",".join(cleanData_stopWords)

    return cleanData_string



def get_training_testing_data(features, labels, test_percentage, recreate_train_test_data):
    """
        Takes in text and label data and preforms a train_test_split with sklearn to
        divide dataset into training and testing data. Also outputs the training and testing series
        into excel files that can be loaded.
        Returns: the split training (X_train, y_train) and testing data (X_test, y_test) sets
    """

    train_size = 1.00 - test_percentage
    rand_state = 47

    train_file = 'training_{}_{}randState.xlsx'.format(train_size, rand_state)
    test_file = 'testing_{}_{}randState.xlsx'.format(test_percentage, rand_state)


    train_path = os.path.join(mydir, train_test_dir, train_file)
    test_path =  os.path.join(mydir, train_test_dir, test_file)

    train_test_folder = os.path.join(mydir, train_test_dir)
    # deleting old file and reloading new one
    if recreate_train_test_data == True and os.path.exists(train_test_folder):
        for root, dirs, files in os.walk(train_test_folder):
            for file in files:
                os.remove(os.path.join(root,file))

    # Column headers for train and test files
    train_columns = ['X_train', 'y_train']
    test_columns = ['X_test', 'y_test']

    if ( os.path.exists(train_path) or os.path.exists(test_path) ):
        X_train, y_train = load_data(train_path, train_columns[0], train_columns[1])
        X_test, y_test = load_data(test_path, test_columns[0], test_columns[1])


    else:

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_percentage, random_state= rand_state)
        X_train.rename(train_columns[0])
        X_test.rename(train_columns[1])


        train_df = pd.DataFrame({train_columns[0]: X_train, train_columns[1]: y_train}, columns = train_columns)


        test_df = pd.DataFrame({test_columns[0]: X_test, test_columns[1]: y_test}, columns = test_columns)


        with pd.ExcelWriter(train_path) as train_writer:
            train_df.to_excel(train_writer, 'Training_Bayes'  , header=True, index_label = False)
            train_writer.save()

        with pd.ExcelWriter(test_path) as test_writer:
            test_df.to_excel(test_writer, 'Test_Bayes' , header=True, index_label = False)
            test_writer.save()


    return X_train, X_test, y_train, y_test
