import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

    
def prepare_data(embedding_type, n_splits):
    
    if embedding_type == "word2vec":

        X = np.load("../data/training_data/X_w2v_asent.npy")
        y = np.load("../data/training_data/y_w2v_asent.npy")
        
    elif embedding_type == "fasttext":

        X = np.load("../data/training_data/X_ft_asent.npy")
        y = np.load("../data/training_data/y_ft_asent.npy")
        
    # define stratify class
    y_class = np.where(y>=0, 1, y) # make 0 to 1 (is only one 0)
    y_class = np.where(y_class<0, -1, y_class)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=3, shuffle=True, stratify=y_class)
    
    if n_splits == 2:
        return X_train, X_test, y_train, y_test
    
    elif n_splits == 3:
        # define stratify class again
        y_class = np.where(y_train>=0, 1, y_train) # make 0 to 1 (is only one 0)
        y_class = np.where(y_class<0, -1, y_class)

        # split train val
        X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=0.2, 
                                                            random_state=3, shuffle=True, stratify=y_class)

        return X_train, y_train, X_test, y_test, X_val, y_val
    
    
def load_data_to_predict(embedding_type):
    
    if embedding_type == "word2vec":
        embeddings_to_predict = np.load("../data/prediction_data/w2v_embeds_to_predict.npy")
        words_to_predict = np.load("../data/prediction_data/w2v_words_to_predict.npy")
        words_to_predict = words_to_predict.tolist() # convert to list
        
    elif embedding_type == "fasttext":
        embeddings_to_predict = np.load("../data/prediction_data/ft_embeds_to_predict.npy")
        words_to_predict = np.load("../data/prediction_data/ft_words_to_predict.npy")
        words_to_predict = words_to_predict.tolist() # convert to list
        
    return embeddings_to_predict, words_to_predict

    

    
  
