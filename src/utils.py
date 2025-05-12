""" Provide python functions which can be used by multiple python files


"""

import pandas as pd
import re
import os
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


def load_split_dataset(dataset):
    """
    This function loads the dataset text file, then converts into a pandas dataframe. Then it applies the cleaning data before performing a split

    Args:
        Dataset
    Outputs:
        training, validation and test sets.
    
    """

    try:

        # load the dataset
        data = pd.read_csv(dataset, sep='\t', names=['english', 'arabic'])

        # Drop any rows that contain NaN values
        data.dropna(inplace=True)

        # clean the data
        data['arabic'] = data['arabic'].apply(clean_arabic)

        # Split data
        train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        print("Training samples:", len(train_df))
        print("Validation samples:", len(val_df))
        print("Test samples:", len(test_df))

        return train_df, val_df, test_df
    
    except Exception as e:
        print(f"Loading and splitting the Arabic-English dataset failed. Error: {e}")



def clean_arabic(text):
    """
    This function  applies the cleaning arabic texts data.

    Args:
        Arabic text
    Outputs:
        cleaned arabic text.
    
    """

    try:

        # Remove diacritics
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

        # Remove tatweel (kashida)
        text = re.sub(r'\u0640', '', text)

        # Remove non-arabic characters
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

        # Normalize arabic letters
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ؤ', 'ء', text)
        text = re.sub(r'ئ', 'ء', text)
        text = re.sub(r'ة', 'ه', text)

        return text

    except Exception as e:
        print(f"Cleaning the Arabic texts failed. Error: {e}")