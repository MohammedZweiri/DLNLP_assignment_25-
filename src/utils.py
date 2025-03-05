""" Provide python functions which can be used by multiple python files


"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from datasets import Dataset
from transformers import T5Tokenizer
import arabic_reshaper
from bidi.algorithm import get_display
import tensorflow as tf
import string
from string import digits
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



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



def load_dataset(dataset_path):
    """Download dataset.

    This function downloads the BloodMNIST dataset from the medmnist library

    Args:
            dataset_name(str): The dataset name to be downloaded

    Returns:
            Training, validation and test datasets.

    """
    try:

        dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng')

        df_train = pd.DataFrame(dataset['validation'])
        df_test = pd.Dataframe(dataset['test'])

        arabic_texts = df_train['sourceString']
        english_texts = df_train['targetString']

        return arabic_texts, english_texts

    except Exception as e:
        print(f"Downloading dataset failed. Error: {e}")



def preprocess_function(arabic_text, english_text):
    
    arabic_texts = arabic_text.astype(str)
    english_texts = english_text.astype(str)

    # Lowercase all characters
    english_texts = english_texts.apply(lambda x: x.lower() if isinstance(x, str) else x)

    # Remove quotes
    english_texts = english_texts.apply(lambda x: re.sub("'", '', x) if isinstance(x, str) else x)
    arabic_texts = arabic_texts.apply(lambda x: re.sub("'", '', x) if isinstance(x, str) else x)

    # Remove Digits
    digits_removal = str.maketrans('', '', digits)
    english_texts = english_texts.apply(lambda x: x.translate(digits_removal) if isinstance(x, str) else x)

    # Remove extra spaces
    english_texts = english_texts.apply(lambda x: x.strip() if isinstance(x, str) else x)
    arabic_texts = arabic_texts.apply(lambda x: x.strip() if isinstance(x, str) else x)
    english_texts = english_texts.apply(lambda x: re.sub(" +", " ", x) if isinstance(x, str) else x)
    arabic_texts = arabic_texts.apply(lambda x: re.sub(" +", " ", x) if isinstance(x, str) else x)

    # Add start and end tokens to target sequences
    english_texts = english_texts.apply(lambda x : '<start> '+ x + ' <end> ' if isinstance(x, str) else x)
    

    return english_texts, arabic_texts

def tokenization(english_texts, arabic_texts):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        # Tokenizer for Arabic
        arabic_tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Tokenizer and pad Arabic sentences
        arabic_sequences = arabic_tokenizer(arabic_texts.tolist(), padding='max_length')
        input_ids = arabic_sequences['input_ids']

        # Tokenizer for English sentences
        english_tokenizer = T5Tokenizer.from_pretrained('t5_small')

        # Tokenize for pad English sentences
        english_sequences = english_tokenizer(english_texts.tolist(), padding='max_length')
        output_ids = english_sequences['input_ids']

        # Get Vocabulary Sizes
        arabic_vocab_size = len(arabic_tokenizer.get_vocab())
        english_vocab_size = len(english_tokenizer.get_vocab())

        # Prepare input and output data for training
        encoder_input_data = input_ids
        decoder_input_data = output_ids[:, :-1]
        decoder_target_data = output_ids[:, 1:]

        return encoder_input_data, decoder_input_data, decoder_target_data

    except Exception as e:
        print(f"tokenization failed. Error: {e}")


def make_batches(ds, Buffer_size, Batch_size):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        return(
            ds
            .shuffle(Buffer_size)
            .batch(Batch_size)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    except Exception as e:
        print(f"Make Batches failed. Error: {e}")


def save_model(task_name,model, model_name):
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
        file_path = Path(f"{task_name}/model/{model_name}.json")
        file_path.write_text(model_structure)

        # Saves the weights as .h5 file
        model.save_weights(f"{task_name}/model/{model_name}.weights.h5")

    except Exception as e:
        print(f"Saving the CNN model failed. Error: {e}")



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



def plot_accuray_loss(task_name, model_history):
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
        fig.savefig(f'{task_name}/figures/CNN_accuracy_loss.png')
    
    except Exception as e:
        print(f"Plotting accuracy and loss has failed. Error: {e}")
