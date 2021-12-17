import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split_data(X, y):
    """
    Split data 
    """

    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, 
                                                        random_state=3, shuffle=True, stratify=y)

    # split train val
    X_train, X_val, y_train, y_val =  train_test_split(X_train, y_train, test_size=0.2, train_size=0.8, 
                                                        random_state=3, shuffle=True, stratify=y_train)
    
    return X_train, y_train, X_test, y_test, X_val, y_val
    
def plot_loss(history, out_history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [SENTIMENT]')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_history)
    
    

    
  
