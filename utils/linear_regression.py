"""
SCRIPT CONTAINING LINEAR REGRESSION CLASS
"""

# --- DEPENDENCIES --

# basics
import os,sys
import numpy as np
import pandas as pd
import itertools

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

# import utils
sys.path.append(os.path.join(".."))
from utils.model_utils import plot_history

# --- LINEAR REGRESSION CLASS ---

class LinearRegression:
    def __init__(self, optimizer, learning_rate, batch_size, epochs):
        
        # define hyperparameters
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs
        
        # define optimizer based on input
        if optimizer == "adam":
            self.opt_lr = tf.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "sgd":
            self.opt_lr = tf.optimizers.SGD(learning_rate=learning_rate)
        
        # build and compile model
        self.model = tf.keras.Sequential([Dense(units=1)])
        self.model.compile(
            optimizer=self.opt_lr,
            loss='mean_absolute_error')
        
    def train(self, X_train, y_train, X_val, y_val, verbose=0, save_model=False, output_path=None):
        """
        Train model using hyperparameters, and validate on validation data
        """
        # train model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=verbose,
            validation_data=(X_val, y_val))
        
        # get train and val loss after last epoch
        self.train_loss = self.history.history["loss"][-1]
        self.val_loss = self.history.history["val_loss"][-1]
        
        # if save model true, save plot of model history
        if save_model == True:
            plot_history(self.history, f"{output_path}/model_history.png")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on a test data
        """
        # compute test loss
        self.test_loss = self.model.evaluate(
            X_test, y_test, verbose=0)
        
        print(f"[INFO] Test loss: {self.test_loss}")
    
    def save_model_details(self, embedding_type, output_path):
        """
        Save parameters and results of the model in a txt file
        """
        # define filepath
        file_path = os.path.join(output_path, "model_details.txt")
        # save parameters and results
        with open(file_path, 'w') as f:
            f.write(f"LINEAR REGRESSION DETAILS for {embedding_type}\n")
            f.write("----------------------\n\n")
            f.write(f"optimizer: {self.optimizer}\n")
            f.write(f"learning_rate: {self.learning_rate}\n")
            f.write(f"batch_size: {self.batch_size}\n")
            f.write(f"epochs: {self.epochs}\n")
            f.write(f"val_loss: {self.val_loss}\n")
            f.write(f"test_loss: {self.test_loss}\n")

        
        
        
            
    
        
        
        
        
    
        
        