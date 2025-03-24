""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
import argparse
from model import transformers
import numpy as np


def task():
    """ Runs the CNN model for bloodMNIST dataset


    """

    print("################ NN Transformers training is starting ################")
    print('\n')

    max_len = 20

    # Download the dataset
    df_train, df_test = utils.download_dataset()

    # Clean and prepare english and arabic text
    df_train['en'] = df_train['en'].apply(lambda row: utils.clean_english_text(row))
    df_train['ar'] = df_train['ar'].apply(lambda row: utils.clean_and_prepare_text(row))

    print(df_train)

    # Scan the phrases
    sequence_len = utils.scan_phrases(df_train['en'], df_train['ar'])

    # Tokenization
    inputs, outputs, arabic_vocab_size, english_vocab_size = utils.tokenization(df_train['en'], df_train['ar'], sequence_len)

    # Model Training
    transformers.transformer_model_training(inputs, outputs, arabic_vocab_size, english_vocab_size, sequence_len)
    # Preprocess
    #english_texts, arabic_texts = utils.preprocess_function(arabic_texts, english_texts)

    # Tokenization
   # encoder_input_data, decoder_input_data, decoder_target_data, arabic_vocab_size, english_vocab_size = utils.tokenization(english_texts, arabic_texts, max_len)
    
    # Model Training
    #transformers.transformer_model_training(encoder_input_data, decoder_input_data, arabic_vocab_size, english_vocab_size, max_len, decoder_target_data)

    # # Run the CNN model

    # if decision == 'train':
    #     CNN_B.CNN_model_training(train_dataset, validation_dataset, test_dataset)
        
    # elif decision == 'test':
    #     CNN_B.CNN_model_testing(test_dataset)

    # print('\n')
    # print("################ Task B via CNN has finished ################")


if __name__ == "__main__":
    # Create Datasets folder
    utils.create_directory("figures")

    task()


    






