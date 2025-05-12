""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
import argparse
from model import  marianMT
import numpy as np


def task(decision):
    """ Runs the model for Arabic-English Machine Translation task


    """

    print("⏳ Arabic-English Machine Translation has started ⏳")
    print('\n')

    # Load, Clean and split the dataset
    train_df, val_df, test_df = utils.load_split_dataset('dataset/ara_eng.txt')

    # Data preparation and tokenization
    train_dataset, val_dataset, test_dataset = marianMT.data_preparation(train_df, val_df, test_df)

    # Run the model
    if decision == 'train':
         marianMT.training_convergence(train_dataset, val_dataset, test_dataset)
        
    elif decision == 'test':
         marianMT.model_evaluation(test_dataset)
         marianMT.translate()

    print('\n')
    print("✅ Arabic-English Machine Translation has finished ✅")


if __name__ == "__main__":

    # Create Datasets folder
    utils.create_directory("figures")

    # Decision argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decision', default='test',
                        help ='select the task')
    
    args = parser.parse_args()
    decision = args.decision


    task(decision)


    






