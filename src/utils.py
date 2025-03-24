""" Provide python functions which can be used by multiple python files


"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import os
import arabic_reshaper
from unicodedata import normalize
from bidi.algorithm import get_display
# from datasets import load_dataset
# from datasets import Dataset
from transformers import T5Tokenizer
import tensorflow as tf
from tensorflow import keras
# import string
from string import digits
# from sklearn.model_selection import train_test_split



def create_directory(directory):
    """Create directory under the current path

    Args:
        directory: String formatted directory name
    
    """
    try:

        # Get the current path
        current_path = os.getcwd()

        # Merge the current path with the desired one
        path = os.path.join(current_path, directory)

        # If the directory exists, do nothing. Otherwise, create it
        if os.path.isdir(path):
            return
        else:
            os.mkdir(path)

    except Exception as e:
        print(f"Creating directory failed. Error: {e}")



def download_dataset():
    """Download dataset.

    This function downloads the BloodMNIST dataset from the medmnist library

    Args:
            dataset_name(str): The dataset name to be downloaded

    Returns:
            Training, validation and test datasets.

    """
    try:

        # dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng', trust_remote_code=True)

        # df_train = pd.DataFrame(dataset['validation'])
        # df_test = pd.DataFrame(dataset['test'])

        # arabic_texts = df_train['sourceString']
        # english_texts = df_train['targetString']

        # return arabic_texts, english_texts, df_test

        dataframe = pd.read_csv('ara_eng.txt', names=['en','ar'], usecols=['en', 'ar'], sep='\t')
        dataframe = dataframe.sample(frac=1, random_state=42)
        dataframe = dataframe.reset_index(drop=True)
        df_train = dataframe.iloc[:21000]
        df_test = dataframe.iloc[21000:]

        #print(dataframe)
        return df_train, df_test

    except Exception as e:
        print(f"Downloading dataset failed. Error: {e}")


def clean_english_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text

def remove_diacritics(text):
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(arabic_diacritics, '', text)
    return text

def clean_arabic_text(text):
    text = remove_diacritics(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'[،؛؟.!"#$%&\'()*+,-/:;<=>?@[\]^_`{|}~]', '', text)
    return text

def clean_and_prepare_text(text):
    text = '[Start] '+clean_arabic_text(text) + ' [end]'
    return text

def scan_phrases(arabic, english):
    en = english
    ar = arabic

    en_max_len = max(len(line.split()) for line in en)
    ar_max_len = max(len(line.split()) for line in ar)
    sequence_len = max(en_max_len, ar_max_len)

    print(f'Max phrase length (English): {en_max_len}')
    print(f'Max phrase length (Arabic): {ar_max_len}')
    print(f'Sequence length: {sequence_len}')

    return sequence_len


def tokenization(english_texts, arabic_texts, sequence_len):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        # Tokenizer for Arabic
        ar_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        #ar_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        # Tokenizer and pad Arabic sentences
        # arabic_sequences = arabic_tokenizer(arabic_texts.tolist(), padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')
        # input_ids = arabic_sequences['input_ids']

        #ar_tokenizer.fit_on_texts(arabic_texts)
        #ar_sequences = ar_tokenizer.texts_to_sequences(arabic_texts)
        ar_x = ar_tokenizer(arabic_texts.tolist(), max_length=sequence_len, padding='max_length', truncation=True, return_tensors='tf')
        ar_x = ar_x['input_ids']
        print(f"Arabic Tokenization Completed!!")
        # Tokenizer for English sentences
        english_tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # # Tokenize for pad English sentences
        en_y = english_tokenizer(english_texts.tolist(), padding='max_length', truncation=True, max_length=sequence_len+1, return_tensors='tf')
        en_y = en_y['input_ids']
        # en_tokenizer = tf.keras.layers.TextVectorization(output_sequence_length=sequence_len+1)
        # en_tokenizer.adapt(english_texts)
        # en_y = en_tokenizer(english_texts)
        # en_y = en_y.numpy()

        print(f"English Tokenization Completed!!")
        # Get Vocabulary Sizes
        arabic_vocab_size = len(ar_tokenizer.get_vocab())
        english_vocab_size = len(english_tokenizer.get_vocab())

        # Prepare input and output data for training
        # encoder_input_data = input_ids
        # decoder_input_data = output_ids[:, :-1]
        # decoder_target_data = output_ids[:, 1:]

        inputs = {'encoder_input': ar_x, 'decoder_input': en_y[:, :-1]}
        outputs= en_y[:, 1:]
        print(inputs)
        print(outputs)

        return inputs, outputs, arabic_vocab_size, english_vocab_size

    except Exception as e:
        print(f"tokenization failed. Error: {e}")


def save_model(model, model_name):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        # Convert the model structure into json
        model_structure = model.to_json()

        # Creates a json file and writes the json model structure
        file_path = Path(f"./model/{model_name}.json")
        file_path.write_text(model_structure)

        # Saves the weights as .h5 file
        model.save_weights(f"./model/{model_name}.weights.h5")

    except Exception as e:
        print(f"Saving the Transformer model failed. Error: {e}")



def load_model(task_name, model_name):
    """Save CNN model.

    This function loads the saved CNN model and weights to be used later on.

    Args:
            model_name(str)
            
    Returns:
            CNN model

    """

    try:
        
        # Locate the model structure file
        file_path = Path(f"{task_name}/model/{model_name}.json")

        # Read the json file and extract the CNN model
        model_structure = file_path.read_text()
        model = model_from_json(model_structure)

        # Load the CNN weights
        model.load_weights(f"{task_name}/model/{model_name}.weights.h5")

        return model
    
    except Exception as e:
        print(f"Loading the CNN model failed. Error: {e}")



def plot_accuray_loss(model_history):
    """Plot accuracy loss graphs for the CNN model.

    This function plots the CNN model's accuracy and loss against epoch into a fig file.

    Args:
            model history

    """

    try:

        # Create the subplots variables.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))

        # Plot the accuracy subplot
        accuracy = model_history.history['accuracy']
        validation_accuracy = model_history.history['val_accuracy']
        epochs = range(1, len(accuracy)+1)
        ax1.plot(epochs, accuracy, label="Training Accuracy")
        ax1.plot(epochs, validation_accuracy, label="Validation Accuracy")
        ax1.set_title('Training and validation accuracy')
        ax1.set_xlabel('Number of Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid()

        # Plot the loss subplot
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        ax2.plot(epochs, loss, label="Training loss")
        ax2.plot(epochs, val_loss, label="Validation loss")
        ax2.set_title('Training and validation loss')
        ax2.set_xlabel('Number of Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid()

        # Save the subplots file.
        fig.savefig(f'./figures/Transformers_accuracy_loss_1.png')
    
    except Exception as e:
        print(f"Plotting accuracy and loss has failed. Error: {e}")
