""" Provide python functions which can be used by multiple python files


"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import os
import arabic_reshaper
from bidi.algorithm import get_display
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



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

        with open(dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        data = [line.strip().split("\t")[:2] for line in lines]

        df = pd.DataFrame(data, columns=["Eng", "Ar"])

        df['Ar'] = df['Ar'].apply(clean_arabic)
        
        #print(df["Ar"])
        print(df.info()) 
        #print(df.isnull().sum())

    except Exception as e:
        print(f"Downloading dataset failed. Error: {e}")



def clean_arabic(text):
    """Clean Arabic text.

    This function performs data transform via normalization.

    Args:
            text.

    Returns:
            normalized training, validation and test datasets.

    """

    try:
        
        text = re.sub(r"[\u064B-\u065F]", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = arabic_reshaper.reshape(text)
        text = get_display(text)
        return text

    
    except Exception as e:
        print(f"Cleaning Arabic text failed. Error: {e}")


def tokenization(df):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        arabic_tokenizer = Tokenizer(filters="", oov_token="<UNK>")
        english_tokenizer = Tokenizer(filters="", oov_token="<UNK>")

        print("Checkpoint 1")
        arabic_tokenizer.fit_on_texts(df["Ar"])
        english_tokenizer.fit_on_texts(df["Eng"])

        print("Checkpoint 2")
        X_sequence = arabic_tokenizer.text_to_sequences(df['Ar'])
        Y_sequence = english_tokenizer.text_to_sequences(df['Eng'])

        print("Checkpoint 3")
        max_len_ar = max(len(seq) for seq in X_sequence)
        max_len_en = max(len(seq) for seq in Y_sequence)

        print("Checkpoint 4")
        X_padded = pad_sequences(X_sequence, maxlen=max_len_ar, padding="post")
        Y_padded = pad_sequences(Y_sequence, maxlen=max_len_en, padding="post")

        print("Checkpoint 5")
        print("Arabic Vocabulary Size:", len(arabic_tokenizer.word_index))
        print("English Vocabulary Size:", len(english_tokenizer.word_index))

    except Exception as e:
        print(f"Text tokenization failed. Error: {e}")





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
