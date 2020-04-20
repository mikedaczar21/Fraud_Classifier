# Importing the libraries
import numpy as np
import pandas as pd
import string
import dill
import os
import nltk
import random
from joblib import dump, load

# Text cleaning imports
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from collections import Counter, defaultdict
from uszipcode import SearchEngine   # Used to get geolocation information from zipcode
from symspellpy.symspellpy import SymSpell, Verbosity  # Library to clean up misspelled words
import pkg_resources


# Datetime imports
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date


from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import re
from collections import OrderedDict
from text_processing_cleanup import replace_str, replace_str_general
import shap



current_dir = os.path.abspath(os.path.curdir)
initial_data_dir = os.path.abspath(os.path.join(current_dir, "Initial_Datasets"))
feature_gen_data_dir = os.path.abspath(os.path.join(current_dir, "Feature_Generated_Datasets"))
prediction_dir = os.path.abspath(os.path.join(current_dir, "Predictions"))


word_vec_dir = os.path.abspath(os.path.join(current_dir, "Word_Vectors"))
vocab_dir = os.path.abspath(os.path.join(word_vec_dir, "Vocab"))
geo_data_dir = os.path.abspath(os.path.join(feature_gen_data_dir, "Geolocation_Data"))


dictionary_path = os.path.join(vocab_dir, "frequency_dictionary_english.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
text_feature_path = os.path.abspath( os.path.join(feature_gen_data_dir, "Text_Data") )

# Very important parameter for text feature (Glove vs Regular Training)
text_feature_type = "GloVe"
text_feature_type = "NoLossDate"


num_classes = 2 # Number of different classes that is used for label


combined_feature_file_name = "CNN_prob_combined_features"
combined_feature_serial_name = "CNN_prob_combined_features"


# Combining directories into absolute paths which functions can use

init_data_path = os.path.abspath(os.path.join(current_dir, initial_data_dir))
fraud_serial_path = os.path.join(init_data_path, "FraudData_Init.pk")

gis_file_name = "GIS_Data.pk"
gis_data_path = os.path.join(init_data_path, gis_file_name)


combined_feature_xlsx_path = os.path.join(feature_gen_data_dir, combined_feature_file_name)
combined_feature_serialized_path = os.path.join(feature_gen_data_dir, combined_feature_serial_name)

train_test_dir =  os.path.join(feature_gen_data_dir, "Train_Test_Split")

# Text paths
text_feature_file_name = "{}_feature_fraud.pk".format(text_feature_type)
text_feature_glove = "glove_text_feature.pk"
text_xlsx_file_name = "{}_text_feature.xlsx".format(text_feature_type)
text_feature_fraud_path = os.path.join(text_feature_path, text_feature_file_name)
text_feature_glove_path = os.path.join(text_feature_path, text_feature_glove)
text_xlsx_path = os.path.join(text_feature_path, text_xlsx_file_name)


# Parameters for SysSpell
max_edit_distance_dict = 2
prefix_length = 7
dict_term_index = 0
dict_count_index = 1



def get_combined_feature( features, cnn_Prob, recreate_combined_features, **kwargs):

    feature_type = kwargs['feature_type']

    combined_feature_excel_full = combined_feature_file_name + "_" +  feature_type + ".xlsx"
    combined_feature_serial_full = combined_feature_serial_name + "_" +  feature_type + ".pk"

    combined_feature_xlsx_path = os.path.join(feature_gen_data_dir, combined_feature_excel_full)
    combined_feature_serialized_path = os.path.join(feature_gen_data_dir, combined_feature_serial_full)

    combined_xlsx_exists = os.path.exists(combined_feature_xlsx_path)
    combined_serial_exists = os.path.exists(combined_feature_serialized_path )

    if (combined_serial_exists == True) and (recreate_combined_features == True):
        os.remove(combined_feature_xlsx_path)
        os.remove(combined_feature_serialized_path )



    if (combined_serial_exists == True)  and (recreate_combined_features == False):

        with open(combined_feature_serialized_path, 'rb') as read:
            combine_cnn_feature = dill.load(read)


    else:


        cnn_pred_prob = pd.read_excel(cnn_Prob)
        cnn_pred_prob.drop(['Feature'], axis=1)
        #combined_features_df = pd.concat( [cleaned_Text, features, cnn_Prob], axis = 1)

        combine_cnn_feature = features.join(cnn_pred_prob, how='inner')

        with open(combined_feature_serialized_path, 'wb') as write:
            dill.dump(combine_cnn_feature, write)

        with pd.ExcelWriter(combined_feature_xlsx_path) as xlsx_writer:
            combine_cnn_feature.to_excel(xlsx_writer, "Combined_Features_CNN_Prob", header=True, index_label= False)
            xlsx_writer.save()

    return combine_cnn_feature



def get_training_testing_data(features, labels, test_percentage, data_type, rand_state, recreate_train_test_data):
    """
        Takes in text and label data and preforms a train_test_split with sklearn to
        divide dataset into training and testing data. Also outputs the training and testing series
        into excel files that can be loaded.
        Returns: the split training (X_train, y_train) and testing data (X_test, y_test) sets
    """

    train_size = 1.00 - test_percentage


    train_file = 'training_{}_{}rand_{}.xlsx'.format(train_size, rand_state, data_type)
    test_file = 'testing_{}_{}rand_{}.xlsx'.format(test_percentage, rand_state, data_type)


    train_path = os.path.join( train_test_dir, train_file)
    test_path =  os.path.join( train_test_dir, test_file)


    # deleting old file and reloading new one
    if recreate_train_test_data == True and os.path.exists(train_file):
        os.remove(train_file)
        os.remove(test_path )




    if ( os.path.exists(train_path) or os.path.exists(test_path) ):

        train_data = pd.read_excel(train_path)
        X_train = train_data.loc[:, [col for col in list(train_data.columns) if col.find("Label") == -1 ] ]
        y_train = train_data.loc[:, [col for col in list(train_data.columns) if col.find("Label") > -1 ] ]

        test_data  =  pd.read_excel(test_path)
        X_test = test_data.loc[:, [col for col in list(test_data.columns) if col.find("Label") == -1 ]  ]
        y_test = test_data.loc[:,'Fraud_Label']


    else:

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = test_percentage, random_state= rand_state)

        if (data_type == "CNN_Encoded"):
            X_train_df = pd.DataFrame(X_train, columns = ['Word_'+ str(num) for num in range(X_train.shape[1]) ])
            y_train_df = pd.DataFrame(y_train, columns = [ 'Label_Reject', 'Label_Accept'])
            train_df = pd.concat([X_train_df, y_train_df], axis=1)

        elif (data_type == "CNN_Text"):

            y_train_df = pd.DataFrame(y_train, columns = [ 'Label_Reject', 'Label_Accept'])
            y_train_df.set_index(X_train.index, inplace= True)
            train_df = pd.concat([X_train, y_train_df], axis=1)

        else:

            train_df = X_train.join(y_train, how="inner")
            test_df = X_test.join(y_test, how = "inner")


        with pd.ExcelWriter(train_path) as train_writer:
            train_df.to_excel(train_writer, 'Training_{}'.format(data_type)  , header=True, index_label = False)
            train_writer.save()

        with pd.ExcelWriter(test_path) as test_writer:
            test_df.to_excel(test_writer, 'Test_{}'.format(data_type) , header=True, index_label = False)
            test_writer.save()


    return X_train, X_test, y_train, y_test



def export_predictions( features, prob, pred , actual, recreate_ProbPreds, pred_path, file_name, **kwargs):

    model_type = kwargs['model_type']

    combined_ProbPred_excel_full = file_name  + ".xlsx"
    combined_ProbPred_serial_full = file_name  + ".pk"

    combined_ProbPred_xlsx_path = os.path.join(pred_path, combined_ProbPred_excel_full)
    combined_ProbPred_serialized_path = os.path.join(pred_path, combined_ProbPred_serial_full)

    combined_xlsx_exists = os.path.exists(combined_ProbPred_xlsx_path)
    combined_serial_exists = os.path.exists(combined_ProbPred_serialized_path )

    if (combined_serial_exists == True) and (combined_xlsx_exists == True ) and (recreate_ProbPreds == True):
        os.remove(combined_ProbPred_xlsx_path)
        os.remove(combined_ProbPred_serialized_path )



    if (combined_serial_exists == True)  and (recreate_ProbPreds == False):

        with open(combined_ProbPred_serialized_path, 'rb') as read:
            # output_pred = pd.read_excel(combined_ProbPred_xlsx_path)
            output_pred = dill.load(read)


    else:

        # output_labels = ['Accepted' if elem == 1 else 'Rejected' for elem in actual]
        # output_pred = ['Accepted' if elem == 1 else 'Rejected' for elem in pred]
        model_pred_header =  '{}_Predictions'.format(model_type)

        confidence_prob = []
        for index, elem in enumerate(pred):

            # Fraud prection (or 'accepted' )
            if elem == 1:
                confidence_prob.append( round(float(prob[index][1]), 4) )
            # Non-fraud prediction (or 'rejected')
            elif elem == 0:
                confidence_prob.append( round(float(prob[index][0]), 4) )


        combined_ProbPreds_df =  pd.DataFrame(  {'Actual Label': actual, '{}_Predictions'.format(model_type): pred,
                                                 '{}_Prob_Fraud'.format(model_type): prob[:, 1],  '{}_Prob_Non_Fraud'.format(model_type): prob[:, 0],
                                                 '{}_Confid_Prob'.format(model_type): confidence_prob} )
        output_pred = features.join([combined_ProbPreds_df], how='inner', lsuffix='_left', rsuffix='_right')

        # output_pred['Actual Label'] = output_pred['Actual Label'].replace(1, "Accepted")
        # output_pred['Actual Label'] = output_pred['Actual Label'].replace(0, "Rejected")
        #
        # output_pred[model_pred_header] = output_pred[  model_pred_header].replace(1, "Accepted")
        # output_pred[model_pred_header] = output_pred[  model_pred_header].replace(0, "Rejected")


        with open(combined_ProbPred_serialized_path, 'wb') as write:
            dill.dump(output_pred, write)

        with pd.ExcelWriter(combined_ProbPred_xlsx_path) as xlsx_writer:
            output_pred.to_excel(xlsx_writer, "Pred_Prob_{}".format(model_type), header=True, index_label= False)
            xlsx_writer.save()





    return output_pred



def create_indicator_matrix(label_Vector, **kwargs):

    check_index = kwargs['check_index']

    num_labels = len(label_Vector)
    target_matrix = np.zeros( (num_labels, num_classes) )


    # creating one hot encoded matrix
    for index in range(num_labels):

        if check_index:

            if (index in label_Vector.index):
                current_label = label_Vector[index]
                target_matrix[index, current_label] = 1

        else:
            current_label = label_Vector[index]
            target_matrix[index, current_label] = 1



    return target_matrix



def create_SysSpell_Object():

    # creating SymSpell object to correct spelling errors
    sys_spell = SymSpell(max_edit_distance_dict, prefix_length)

    # load dictionary
    if not sys_spell.load_dictionary(dictionary_path, dict_term_index, dict_count_index):
        print("Dictionary not found\n.")
        return

    sys_spell.load_dictionary(dictionary_path, dict_term_index, dict_count_index)
    sys_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


    return sys_spell


def correcting_Spelling(input_Sentence):


    bad_chars = [';', ':', '!','*', '-', '.', '#', '*', '[]', '//']


    stopWords = set(stopwords.words('english'))

    sys_spell = create_SysSpell_Object()

    sentence_acroynm_replace = replace_str(input_Sentence.upper())

    sentence_sym_remove =  re.sub(r"[\-#!//()]", " ", sentence_acroynm_replace)

    sentence_split = sentence_sym_remove.lower().split(" ")

    # sentence_stopWords = [word for word in sentence_split if not word in stopWords if word != '' and word != r'\s']


    # sentence_symbols_remove = [word.replace(symbols_to_remove, " ") for word in sentence_split]
    #
    # sentence_join = " ".join(sentence_symbols_remove)

    corrected_words_list = []

    # print("Letter split {}".format(sentence_split))

    ###### LOOKUP COMPOUND
    # correct_spell_suggestions = sys_spell.lookup_compound(sentence_sym_remove, max_edit_distance=2)
    #
    # correct_sentence = correct_spell_suggestions[0]
    #
    # print("\nCurrent sentence: {0}\n".format(sentence_acroynm_replace))
    #
    # for correct_sugg in correct_spell_suggestions:
    #      print(" Suggested Sentence: {}\n".format( correct_sugg))
    #
    # corrected_words_list = correct_sentence.term
    #######

    # correct_spell_suggestions = sys_spell.word_segmentation(sentence_acroynm_replace)
    #
    # print("\nCurrent sentence: {0}\n".format(sentence_acroynm_replace))
    #
    # print(" Suggested Sentence: {}\n".format( correct_spell_suggestions.corrected_string))
    #
    # corrected_words_list = correct_spell_suggestions.corrected_string

    ###### SPELL CHECKER
    # spell = SpellChecker()
    #
    # misspelled_words = spell.unknown(sentence_sym_remove)
    #
    # print("\nCurrent sentence: {0}\n".format(sentence_acroynm_replace))
    #
    # if len(misspelled_words) > 0:
    #
    #     for word in sentence_sym_remove.lower().split(" "):
    #
    #         if word in misspelled_words:
    #             corrected_words_list.append( spell.correction(word) )
    #         else:
    #             corrected_words_list.append( word )
    #
    #     for word in misspelled_words:
    #          print(" Suggested Word: {}\n".format( spell.correction(word)))
    #          # corrected_words_list.append( spell.correction(word))
    #
    #     corrected_words_list = " ".join(corrected_words_list)
    #
    # else:
    #
    #     corrected_words_list = sentence_sym_remove

    ######## LOOKUP (look at each word)
    for word in sentence_split:

        if (word.isalpha() == True) and (len(word) > 2) and (word not in stopWords) and (word != r'\s'):

            # correct_spell_suggestions = sys_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            correct_spell_suggestions = sys_spell.lookup_compound(word, max_edit_distance=2)

            corrected_word = correct_spell_suggestions[0]
            #
            # print("\nCurrent word: {0}".format(word))
            # for correct_sugg in correct_spell_suggestions:
            #     print(" and Suggested Word {1}\n".format(word, correct_sugg))

            corrected_words_list.append(corrected_word.term)

        else:

            if(word != r'\s'):
                corrected_words_list.append(word)


    corrected_words_list = " ".join(corrected_words_list)
    ########


    return corrected_words_list.lower()


def get_city_state(state_data):


    geo_search = SearchEngine() # set simple_zipcode=False to use rich info database

    state = geo_search.by_state(state_data)

    return state[0].state_long


# Helper function which takes in zip codes
def get_Geolocation_Data(zip_Code_Data, fraud_type):

    geo_data_path = os.path.join(geo_data_dir, "current_geo_{}.pk".format(fraud_type))
    geo_search = SearchEngine(simple_zipcode=True) # set simple_zipcode=False to use rich info database

    df_complete_geo_data= pd.DataFrame(columns = [ 'Claim Number' , 'Longitude', 'Latitude'])
    geo_list_complete = []
    print("Type of zip code data: {}".format(type(zip_Code_Data)))


    if (os.path.exists(geo_data_path)):
        print("\nLoading Lat/Long")
        with open(geo_data_path, 'rb') as read:
            df_complete_geo_data = dill.load(read)

    else:


        for current_index in zip_Code_Data.index:

            current_zip_code = zip_Code_Data.loc[current_index ,'LOSS_LOCATION_ZIP']
            current_claim_num = zip_Code_Data.loc[current_index ,'Claim Number']

            # if (type(current_zip_code) == 'str') and (current_zip_code != '99999'):
            if (isinstance(current_zip_code, str) == True ):

                # print("Current Zip Code {}".format(current_zip_code))
                if ("99999" not in current_zip_code)  and (current_zip_code.strip() != "99999") and (current_zip_code.isalpha() == False):




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


def get_city_Data(city_data):

    city_replace_dict = {
                'N': 'north',
                'S': 'south',
                'E': 'east',
                'W': 'west',
                'FT': 'fort',
                'ST': 'saint',
                'MT': 'mount',
                'BCH': 'beach',
                'SF': 'San francisco'
    }

    bad_city_data = { 'UNKNOWN', 'UKNOWN',  'UNK', 'UKN', 'UNKNWO', 'NAN', 'VAARIOUS' , 'VARIOUS', 'VAROIUS', 'VARIOUSDD', 'VAROUS', 'VARIOU', 'TBD', 'UNKNOWN EAST CHICAGO'}

    geo_data_path = os.path.join(geo_data_dir, "current_city_data.pk")
    geo_data_path_xlsx = os.path.join(geo_data_dir, "current_city_data.xlsx")
    geo_search = SearchEngine() # set simple_zipcode=False to use rich info database

    df_city_data = pd.DataFrame(columns = [ 'Claim Number' , 'Old_City', 'New_City'])
    city_list = []
    print("Type of city data: {}".format(type(city_data)))


    if (os.path.exists(geo_data_path)):
        with open(geo_data_path, 'rb') as read:
            df_city_data = dill.load(read)

    else:


        for current_index, current_row in city_data.iterrows():
            # print("Zip type: {} \n".format(type(zip)) )
            # print("Current zip {}\n".format(zip))
            # print("Index: {}\n".format(current_index))



            current_lat = current_row['Latitude']
            current_long = current_row['Longitude']
            current_city = current_row['Loss  City']
            current_claim_num = current_row['Claim Number']

            city_has_nums_chars = bool(re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', current_city))

            city_letters_only = bool(re.match(r'^[a-zA-Z ]+$', current_city))

            claimant_found = current_city.lower().find("claimant")

            # unknown_prefix = current_city.lower().find("unk")
            #
            # various_prefix = current_city.lower().find("vari")
            #
            # bad_city_flag = False
            # if (unknown_prefix > -1):
            #     bad_city_flag = True
            # elif(various_prefix > -1):
            #     bad_city_flag = True

            current_lat_str = str(current_lat)
            current_long_str = str(current_long)
            new_city = ""
            geo_city = ""
            # Case where city data is bad but lat/long is good so can use that to do lookup for city
            if (current_lat_str.lower() != 'nan')  and ((current_city.upper() in bad_city_data) or (city_has_nums_chars == True) ):

                geo_info = geo_search.by_coordinates(current_lat, current_long, radius=10)


                geo_city = geo_info[0].city
                # print("Found geolocation lookup: {}".format(geo_city))


            if  (current_city.upper() not in bad_city_data) and (city_letters_only == True)  and (len(current_city)>1) and (claimant_found == -1) :



                current_city_replace = replace_str_general(current_city.upper(), city_replace_dict)

                city_sym_remove =  re.sub(r"[\-\.]", " ", current_city_replace.lower())

                city_space_removed = " ".join(city_sym_remove.split())

                new_city = city_space_removed

                # print("New City Cleaned: {}, Geocity: {}".format(new_city, geo_city))

            else:

                if geo_city.strip() != "":
                    new_city = geo_city
                else:
                    new_city = ""

            current_city_data = [current_claim_num, current_city, new_city]
            city_list.append(current_city_data)



            if (current_index == (len(city_data) - 1) ):
                print("Current Index: {0}, Len of City Col: {1}".format(current_index, len(city_data)))
                df_city_data = pd.DataFrame(city_list, columns = [ 'Claim Number' , 'Old_City', 'New_City'] )



                with pd.ExcelWriter(geo_data_path_xlsx) as xlsx_writer:
                    df_city_data.to_excel(xlsx_writer, 'Fraud_City', header=True, index_label= False)
                    xlsx_writer.save()


                with open(geo_data_path, 'wb') as write:
                    dill.dump(df_city_data, write)

    #print("Geo List Complete:  {0}\n".format(geo_list_complete) )

    return   df_city_data, city_list



def create_DateDiff_Features(initial_df_row):

    current_claim_num = initial_df_row['Claim Number']
    policy_eff_date = initial_df_row[ 'Policy Eff Date']
    policy_exp_date = initial_df_row['Policy Exp Date']
    claim_date = initial_df_row['Claim Receipt Date']
    loss_date = initial_df_row['Loss Date']


    loss_policyEff_diff =  (loss_date - policy_eff_date) / np.timedelta64(1,'D')

    policyExp_loss_diff =  (policy_exp_date - loss_date) / np.timedelta64(1,'D')

    claim_PolicyEff_diff = (claim_date - policy_eff_date) / np.timedelta64(1, 'D')

    claim_loss_diff = (claim_date - loss_date) / np.timedelta64(1, 'D')


    date_diff_dict = {
                    'claim_num': current_claim_num,
                    'loss_policyEff_diff':loss_policyEff_diff,
                    'policyExp_loss_diff':policyExp_loss_diff,
                    'claim_PolicyEff_diff': claim_PolicyEff_diff,
                    'claim_loss_diff': claim_loss_diff
                    }

    #date_diff_features_df = pd.DataFrame( date_diff_list, columns = ['Loss_PolicyEff_Diff', 'Loss_PolicyExp_Diff'])


    return date_diff_dict



def create_Body_Parts_Feature(initial_df_row):

    current_claim_num = initial_df_row['Claim Number']
    body_parts = str(initial_df_row['Body Part Description'])

    multi_body_parts_injured = -1



    if body_parts.lower().find("multiple") != -1:

        multi_body_parts_injured = 1

    else:

        multi_body_parts_injured = 0

    body_parts_dict = {
                    'claim_num': current_claim_num,
                    'body_parts_inj':multi_body_parts_injured
                    }


    #multi_body_parts_feature = pd.DataFrame(multi_body_parts_injured, columns = ['Multi_Body_Parts_Injured'])

    return   body_parts_dict


def get_subline_business(fraud_data):


    with open(gis_data_path, 'rb') as read:
        gis_fraud_df = dill.load(read)



    fraud_claimNum = fraud_data['Claim Number']
    first_claim = list(fraud_claimNum + "-001")

    idx_match = []
    fraud_data['Subline_Business'] = ''

    reverse_lookup = {elem:idx for idx, elem in enumerate(first_claim)}
    for idx, row in gis_fraud_df.iterrows():

        if reverse_lookup.get(row["FEATURE"], -1) != -1:
            fraud_row = fraud_data[ fraud_data['Claim Number'] == row["CLAIM_NUM"] ].index
            fraud_data.loc[fraud_row, 'Subline_Business'] = row["SUBLINE_OF_BUSINESS_DS"]



    return fraud_data







def get_Fraud_Dataset(**kwargs):

    recreate_features  = kwargs['recreate_features']
    fraud_type = kwargs['fraud_type']
    correct_spelling = kwargs['correct_spelling']

    text_feature_type = "NoLossDate"
    # Feature paths
    new_feature_gen_file = "Fraud_Features_{}_{}.xlsx".format(text_feature_type, fraud_type)
    feature_gen_data_path = os.path.join(feature_gen_data_dir, new_feature_gen_file)

    new_feature_serial_name= "Fraud_Features_{}_{}.pk".format(text_feature_type, fraud_type)
    feature_serial_path = os.path.join(feature_gen_data_dir, new_feature_serial_name)



    text_feature_file_name = "{}_feature_fraud.pk".format(fraud_type)
    text_feature_glove = "glove_text_feature.pk"
    text_xlsx_file_name = "{}_text_feature.xlsx".format(fraud_type)
    text_feature_fraud_path = os.path.join(text_feature_path, text_feature_file_name)
    text_feature_glove_path = os.path.join(text_feature_path, text_feature_glove)
    text_xlsx_path = os.path.join(text_feature_path, text_xlsx_file_name)


    print("Initial data path: {0} \n Local Fraud Data Path: {1} \n Feature Gen Path: {2}".format(init_data_path, fraud_serial_path, feature_gen_data_path))


    if os.path.exists(feature_serial_path) and (recreate_features  == False):

        print("Loading features \n")
        # feature_data_df = pd.read_excel(feature_gen_data_path)

        with open(feature_serial_path, 'rb') as read:
            feature_data_df = dill.load(read)

        feature_data_df = feature_data_df.drop(['Unnamed: 0'], axis=1)




    # Otherwise use pandas to load in excel file and only keep the important columns
    else:

        # Checking if flag to recreate features is set to true and if so then delete feature files
        # if (recreate_features == True) :

            # if (os.path.exists(feature_gen_data_path)):
            #     os.remove(feature_gen_data_path)

            # if (os.path.exists(text_feature_fraud_path)):
            #     os.remove(text_feature_fraud_path)




        with open(fraud_serial_path, 'rb') as read:
            initial_large_data_df = dill.load(read)

        print("Initial Large dataframe size before drop {}\n".format(len(initial_large_data_df)))

        if fraud_type == 'acceptance':
            # Dropping rows that are labelled 'not applicable', only focusing on 'accepted' and 'rejected' labels
            not_applicable_rows = initial_large_data_df[initial_large_data_df['Fraud Referral'] == 'No'].index
            initial_large_data_df.drop(not_applicable_rows, inplace=True)

        elif fraud_type == 'refferal':

            initial_large_data_df['Fraud Acceptance'] = initial_large_data_df['Fraud Acceptance'].replace("Not Applicable", 0)

            # Dropping rows with empty main cause
            empty_main_cause = initial_large_data_df[initial_large_data_df['Main Cause'].isna()].index
            initial_large_data_df.drop(empty_main_cause, inplace=True)


        # Dropping rows which do not have policy effective date
        null_policyEff_rows = initial_large_data_df[ initial_large_data_df['Policy Eff Date'].isna() == True ].index
        initial_large_data_df.drop(null_policyEff_rows, inplace = True)

        print("Initial Large dataframe size after drop {}\n".format(len(initial_large_data_df)))

        # initial_large_data_df = initial_large_data_df.fillna("")


        # Grabbing zip code data from large dataframe
        # zip_codes = pd.concat( [initial_large_data_df['Claim Number'] , initial_large_data_df['LOSS_LOCATION_ZIP']], axis = 1)

        # Getting geolocation dataframe with helper function
        new_geoloc_data, geo_list = get_Geolocation_Data(initial_large_data_df.loc[:, ['Claim Number','LOSS_LOCATION_ZIP'] ], fraud_type )


        # Merging large dataframe with
        initial_large_data_df = pd.merge(initial_large_data_df, new_geoloc_data , how='left', on='Claim Number')


        # new_city_data, city_list = get_city_Data(initial_large_data_df.loc[:,['Claim Number','Loss  City', 'Longitude', 'Latitude']])
        #
        # initial_large_data_df = pd.merge(initial_large_data_df, new_city_data, how='left', on='Claim Number')

        initial_large_data_df['Fraud Acceptance'] = initial_large_data_df['Fraud Acceptance'].replace("Accepted", 1)
        initial_large_data_df['Fraud Acceptance'] = initial_large_data_df['Fraud Acceptance'].replace("Rejected", 0)


        initial_large_data_df = get_subline_business(initial_large_data_df)


        print("Initial Large dataframe size after merge {}\n".format(len(initial_large_data_df)))

        # generating features with helper function
        feature_data_df = feature_Generation_Init_Data(initial_Dateset = initial_large_data_df, correct_spell = correct_spelling, text_feat_path = text_feature_fraud_path)

        # feature_data_df = feature_data_df.drop(['Unnamed: 0'], axis=1)

        # with pd.ExcelWriter(feature_gen_data_path) as xlsx_writer:
        #     feature_data_df.to_excel(xlsx_writer, 'Fraud_{}'.format(text_feature_type), header=True, index_label= False)
        #     xlsx_writer.save()

        with open(feature_serial_path, 'wb') as write:
            dill.dump(feature_data_df, write)





    return feature_data_df


def feature_Generation_Init_Data(initial_Dateset, correct_spell, text_feat_path):


    new_text_feature_df = pd.DataFrame( columns = ['Claim Number', 'New_Loss_Descrip', 'Loss_Descrip_NoAddition'])
    new_text_list = []

    current_datediff_list = []
    current_body_part_list = []


    # iterating through initial dataframe
    # for index, row in initial_Dateset.iterrows():
    for index in initial_Dateset.index:

        # Only recreate text feature if we need to
        if  (correct_spell == True):


            main_in_loss = True
            sub_in_loss = True

            # adding main/sub cause to text feature if it is not already included and if it is not null
            # if (initial_Dateset.loc[index,"Main Cause"] != 'nan')  and (initial_Dateset.loc[index,"Loss_Description"].lower().find(initial_Dateset.loc[index ,"Main Cause"].lower()) == -1):
            print("Current Main/SubCause: {} - {} ".format( initial_Dateset.loc[index ,"Main Cause"], initial_Dateset.loc[index ,"Sub Cause"] ) )
            # if (isinstance(initial_Dateset.loc[index ,"Main Cause"], str) == True) and (isinstance(initial_Dateset.loc[index ,"Sub Cause"], str) == True):

            current_main = str(initial_Dateset.loc[index ,"Main Cause"])
            current_sub = str(initial_Dateset.loc[index ,"Sub Cause"])
            if (current_main.lower() != 'nan')  and (current_main not in initial_Dateset.loc[index ,"Loss_Description"]):

                main_in_loss = False

                # This case will add both sub and main causes to text feature since neither is in the loss descrip
                if (current_sub.lower() != 'nan') and (current_sub not in initial_Dateset.loc[index ,"Loss_Description"]) :
                    sub_in_loss = False



                # This case will only need to add the main cause
                elif (current_sub.lower() != 'nan') and (current_sub in initial_Dateset.loc[index ,"Loss_Description"]) :
                    sub_in_loss = True




            # Using text_processing+cleanup helper function to correct misspellings in loss description
            corrected_loss_descrip_list = correcting_Spelling(initial_Dateset.loc[index ,"Loss_Description"])
            print("Row: {0} out of {1} \n".format(index, len(initial_Dateset)))
            print("Loss Descrip Before Correct Spelling: {1}\n".format(index, initial_Dateset.loc[index ,"Loss_Description"]))


            initial_Dateset.loc[index ,"Loss_Description"] = corrected_loss_descrip_list


            print("Loss Descrip After Correct Spelling: {0}\n".format(initial_Dateset.loc[index ,"Loss_Description"]))



            if (current_main.lower() != 'nan')   and (main_in_loss == False):

                # This case will add both sub and main causes to text feature since neither is in the loss descrip
                if (current_sub.lower() != 'nan') and (sub_in_loss == False) :

                    initial_Dateset.loc[index ,"Loss_Description"] = current_main + " - " + current_sub + " - " + initial_Dateset.loc[index ,"Loss_Description"]

                elif  (current_sub.lower() != 'nan') and (sub_in_loss == True):
                    initial_Dateset.loc[index ,"Loss_Description"] = current_main + " - " + initial_Dateset.loc[index ,"Loss_Description"]




            # adding injury damage and body part description to text feature if it is not null
            current_injury = str(initial_Dateset.loc[index ,"Injury Damage"])

            if current_injury.lower() != 'nan':
                initial_Dateset.loc[index ,"Loss_Description"] += " injury damage is "  + initial_Dateset.loc[index ,"Injury Damage"]


            current_body_desc = str(initial_Dateset.loc[index ,"Body Part Description"])
            # adding  body part description to text feature if it is not null
            if  current_body_desc.lower() != 'nan':
                initial_Dateset.loc[index ,"Loss_Description"] +=  " the following body part is injured "   + initial_Dateset.loc[index ,"Body Part Description"]



            # Adding geolocation data to text feature
            current_long = str(initial_Dateset.loc[index ,"Longitude"])
            current_lat = str(initial_Dateset.loc[index ,"Latitude"])



            current_text = [ initial_Dateset.loc[index ,"Claim Number"], initial_Dateset.loc[index ,"Loss_Description"], corrected_loss_descrip_list ]
            # print("Current Loss Descrip: {}\n".format(current_text))

            print("Final loss descrip: {}\n".format(initial_Dateset.loc[index ,"Loss_Description"]))
            new_text_list.append(current_text)





        policy_eff_date = str(initial_Dateset.loc[index ,"Policy Eff Date"])
        policy_exp_date = str(initial_Dateset.loc[index ,"Policy Exp Date"])
        claim_date = str(initial_Dateset.loc[index ,"Claim Receipt Date"])

        if (policy_eff_date.lower() != 'nan'):
            current_datediff_dict = create_DateDiff_Features(initial_Dateset.loc[index, :])
            current_datediff = [current_datediff_dict['claim_num'], current_datediff_dict['loss_policyEff_diff'], current_datediff_dict['policyExp_loss_diff'], current_datediff_dict['claim_PolicyEff_diff'], current_datediff_dict['claim_loss_diff'] ]
            current_datediff_list.append(current_datediff)

        # print("Current Date Diff Features: Loss Policy Eff Diff: {0}, Loss Policy Exp Diff: {1}\n".format(current_datediff_dict['loss_policyEff_diff'],current_datediff_dict['policyExp_loss_diff'] ))



        # print("Current Body Part Feature {}\n".format(current_body_part_dict['body_parts_inj']))



    if (os.path.exists( text_feat_path )):

        with open(text_feat_path, 'rb') as read:
            new_text_feature_df = dill.load(read)

    else:
        new_text_feature_df = pd.DataFrame(new_text_list, columns =  ['Claim Number', 'New_Loss_Descrip', 'Loss_Descrip_NoAddition'])

        with open(text_feat_path, 'wb') as write:
            dill.dump(new_text_feature_df, write)

        # with pd.ExcelWriter(text_xlsx_path) as xlsx_writer:
        #     new_text_feature_df.to_excel(xlsx_writer, "{}_Text_Feature".format(text_feature_type), header=True, index_label= False)
        #     xlsx_writer.save()




    date_diff_features_df = pd.DataFrame( current_datediff_list, columns = ['Claim Number', 'Loss_PolicyEff', 'Loss_PolicyExp', 'Claim_PolicyEff', 'Claim_Loss'])

    new_text_feature_df = pd.merge(new_text_feature_df, date_diff_features_df, on='Claim Number')

    #Getting rid of these columns since the data is added to text feature
    col_exclude_list = [ 'Injury Damage', 'Body Part Description', 'LOSS_LOCATION_ZIP']
    col_keep_list = [col for col in initial_Dateset.columns if col not in col_exclude_list]


    filtered_dataset = initial_Dateset[col_keep_list]

    new_feature_df = pd.merge(filtered_dataset, new_text_feature_df, on='Claim Number')


    new_feature_df.rename( columns = {
                                        'Fraud Acceptance': 'Fraud_Label',
                                        'New_Loss_Descrip': 'Fraud_Text'
                                    },
                        inplace=True)




    return new_feature_df





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



def print_class_report_confusion_matrix(label, prediction, classifier, word_vector):
    """
        Prints out classification report based on the predictions of the classifier.
        Returns: an output dictionary from the classification report (which contains precision, f1 and recall scores)
    """
    print("\n Results for {} (using {}):".format(classifier, word_vector))
    print(classification_report(label, prediction))
    print(confusion_matrix(label, prediction))

    return classification_report(label, prediction, output_dict=True)
