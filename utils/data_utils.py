"""
UTILS FOR DATA PREPARATION

This script contains function to prepare, split and load different kinds of datasets
"""

# --- DEPENDENCIES ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- FUNCTIONS ---
    
def prepare_data(embedding_type, n_splits):
    """
    Function to load labelled datasets for word2vec or fasttext, 
    and split the data in 2 (train, test) or 3 (train, val, test), using stratification 
    """
    # load labelled embeddings depending on type
    if embedding_type == "word2vec":
        X = np.load("../data/labelled_data/X_w2v_asent.npy")
        y = np.load("../data/labelled_data/y_w2v_asent.npy")
        
    elif embedding_type == "fasttext":
        X = np.load("../data/labelled_data/X_ft_asent.npy")
        y = np.load("../data/labelled_data/y_ft_asent.npy")
        
    # turn sentiment scores into positive/negative classes for stratification
    y_class = np.where(y>=0, 1, y) # make 0 to 1 (is only one 0)
    y_class = np.where(y_class<0, -1, y_class)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=3, shuffle=True, stratify=y_class)
    
    # if 2 splits, return train and test data
    if n_splits == 2:
        return X_train, X_test, y_train, y_test
    
    # if 3 splits, split again and return train, val and test data
    elif n_splits == 3:
        # turn sentiment scores into positive/negative classes for stratification
        y_class = np.where(y_train>=0, 1, y_train) # make 0 to 1 (is only one 0)
        y_class = np.where(y_class<0, -1, y_class)
        # split train into train and validation data
        X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                                            random_state=3, shuffle=True, stratify=y_class)
        # return all three datasets
        return X_train, y_train, X_test, y_test, X_val, y_val
    
    
def load_data_to_predict(embedding_type):
    """
    Function to load unlabelled data for lexicon expansion depending on word2vec or fasttext
    """
    if embedding_type == "word2vec":
        embeddings_to_predict = np.load("../data/unlabelled_data/w2v_embeds_to_predict.npy")
        words_to_predict = np.load("../data/unlabelled_data/w2v_words_to_predict.npy")
        words_to_predict = words_to_predict.tolist() # convert to list
        
    elif embedding_type == "fasttext":
        embeddings_to_predict = np.load("../data/unlabelled_data/ft_embeds_to_predict.npy")
        words_to_predict = np.load("../data/unlabelled_data/ft_words_to_predict.npy")
        words_to_predict = words_to_predict.tolist() # convert to list
        
    return embeddings_to_predict, words_to_predict

    

    
  
