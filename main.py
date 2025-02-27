""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
import argparse
import numpy as np


def task(dataset_path):
    """ Runs the CNN model for bloodMNIST dataset


    """

    print("################ Task B via CNN is starting ################")
    print('\n')

    # Download the dataset
    df = utils.load_dataset(dataset_path)

    # Tokenization
    utils.tokenization(df)
    

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

    dataset_path = "ara.txt"
    task(dataset_path)


    






