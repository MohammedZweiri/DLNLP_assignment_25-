# ELEC0141 Deep Learning For Natural Language Processing assingment 2024/2025

There is one task for this assignment:
1. Arabic to English Machine Translation.
https://www.kaggle.com/competitions/UJ-Arabic-translation-competition

## What are the folders and files
1. Main folder contains main.py file, which runs other instances of python scripts.
2. `A` folder contains the following folders:
    - `model` folder which contains `marianMT.py` when running the scripts.
    - `pretrained_model` folder contains the trained marianMT model.
    - `src` folder contain `utils.py` script. This file is called by all ML model scripts in this assignment to perform a centralized tasks, such as: 
        - Loading Arabic-English dataset.
        - Cleaning the Arabic texts.
        - Split dataset into train, validation and test.
        - Creating directories.
3. `Datasets` folder contains the Arabic-English dataset `ara_eng.txt`.

## Important note before the procedure.
1. `main.py` has one arguments set.
   - `decision`, which the user can define how to run the models. You can either run on the training, validation and test dataset (training the model from scratch) using `-d train` or test the loaded model using test dataset by adding no input. The default is set to `test`.

  
## Procedures

1. You should be able to see a `requirements.txt` file, which contains all the libraries needed for this assignment. To install all the libraries from it, simply use `pip install -r requirements.txt` from the command line in the root folder.

2. Once installed, you can start running the tasks. There multiple ways to do it.
    - If you want to run the models for the tasks as running models on the test datasets only, then run `python main.py`
    - If you want to run the models for the tasks as performing the entire training and validation process , then run `python main.py -d train`