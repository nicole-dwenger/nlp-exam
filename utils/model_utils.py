"""
SCRIPT CONTAINING UTILITY FUNCTIONS FOR LINEAR REGRESSION AND NEURAL NETWORK MODEL
"""

# --- DEPENDENCIES ---

import numpy as np
import os, itertools
import matplotlib.pyplot as plt
from datetime import datetime
    
# --- FUNCTIONS ---

def unique_path(filepath):
    """
    Function to create unique filename by enumerating if path exists already 
    """ 
    # if the path does not exist
    if not os.path.exists(filepath):
        #keep the original filepath
        return filepath
    
    # if path exists:
    else:
        i = 1
        # split the path and append a number
        path, ext = os.path.splitext(filepath)
        # add extension
        new_path = "{}_{}{}".format(path, i, ext)
        
        # if the extension exists, enumerate one more
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path
    
def get_hpcombinations(model_type):
    """
    Function to define and retrieve the possible hyperparameter options
    for the linear regression and neural network model
    """
    # if model is linear regression
    if model_type == "linear_regression":
        # dictinary of hyperparameter options
        hpdict = {"optimizer": ['adam', 'sgd'],
                  "learning_rate": [0.001, 0.01],
                  "batch_size": [10, 50, 100],
                  "epochs": [50, 100, 150, 200]}

    # if model is neural network
    elif model_type == "neural_network":
        # dictionary of hyperparameter options
        hpdict = {"optimizer": ['adam', 'sgd'],
                  "learning_rate": [0.001, 0.01],
                  "batch_size": [10, 50, 100],
                  "epochs": [50, 100, 150, 200],
                  "hidden_layers": [[500,10], [500,150,10], [500,150,50], [750,150,50]],
                  "activation": ["relu", "softmax"]}
        
    # get possible parameter combinations 
    hp_combinations = itertools.product(*(hpdict[key] for key in hpdict.keys()))
    hp_combinations = list(hp_combinations)
    
    return hp_combinations

def write_model_results(model, model_type, optimizer, lr, batch_size, epochs, hidden_layers, activation, out_path):
    """
    Function to write the parameters and results of the linear regresion or neural netowkr model to a txt file 
    """
    # if model is linear regression
    if model_type == "linear_regression":
        # open output file and save information
        with open(out_path, 'a') as file:
            file.write(f"MODEL USED FOR PREDICTIONS: {model_type}\n")
            file.write(f"{datetime.now()}\n")
            file.writelines([f"optimizer: {optimizer}\n",
                             f"lr: {lr}\n", 
                             f"batch_size: {batch_size}\n",
                             f"epochs: {epochs}\n",
                             f"train_loss: {model.train_loss}\n",
                             f"val_loss: {model.val_loss}\n", 
                             f"test_loss: {model.test_loss}\n\n\n"])
    
    # if model is linear regression
    elif model_type == "neural_network":
        # open output file and save information
        with open(out_path, 'a') as file:
            file.write(f"MODEL USED FOR PREDICTIONS: {model_type}\n")
            file.write(f"{datetime.now()}\n")
            file.writelines([f"optimizer: {optimizer}\n", 
                             f"lr: {lr}\n", 
                             f"batch_size: {batch_size}\n",
                             f"epochs: {epochs}\n",
                             f"hidden_layers: {hidden_layers}\n", 
                             f"activation: {activation}\n",
                             f"train_loss: {model.train_loss}\n",
                             f"val_loss: {model.val_loss}\n", 
                             f"test_loss: {model.test_loss}\n\n\n"])
    
def plot_history(history, out_history):
    """
    Function to plot model training and validation loss over epochs
    """
    # plot line for train loss
    plt.plot(history.history['loss'], label='Train Loss')
    # plot line for validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # limit y axis scale
    plt.ylim([0, 2])
    # labels of axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss [MAE]')
    # title
    plt.title('Training and Validation Loss over Epochs')
    # create a legend
    plt.legend()
    plt.grid(True)
    # save the figure
    plt.savefig(out_history)    
    
def plot_error_distribution(y_test, y_preds, out_path):
    """
    Function to plot the distribution of prediction errors on test data
    """
    # clear plot
    plt.clf()
    # calculate prediction errors
    error = y_preds - y_test
    # plot histogram with bins
    plt.hist(error, bins=[-10,-9.5,-9,-8.5,-8,-7.5,-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,
                          0.5,1, 1.5, 2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10], ec="black")
    # define titles and axes labels
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Absolute Prediction Error')
    _ = plt.ylabel('Count')
    # save the figure
    plt.savefig(out_path)
    
def plot_regression_error(y_test, y_preds, out_path):
    """
    Function to plot a subset of the true and predicted values of the test data
    """
    # clear the previous plot
    plt.clf()
    # define axes
    _, ax = plt.subplots()
    # create scatter plot for true and predicted values
    ax.scatter(x = range(0, y_test.size), y=y_test, c = 'blue', label = 'True Sentiment Score', alpha = 0.3)
    ax.scatter(x = range(0, y_preds.size), y=y_preds, c = 'red', label = 'Predicted Sentiment Score', alpha = 0.3)
    # define title and axes desriptions
    plt.title('True and Predicted Values for a Subset of 500 Observations')
    plt.xlabel('Index of Observation')
    plt.ylabel('Sentiment Score')
    # limit x axis values
    plt.xlim([0,500])
    # add legend
    plt.legend()
    plt.show()
    # save the figure
    plt.savefig(out_path)
    
def plot_regression_error_diagonal(y_test, y_preds, out_path):
    """
    Function to plot the true and predicted values diagnoally
    """
    # clear the previous plot
    plt.clf()
    # define axes
    a = plt.axes(aspect='equal')
    # scatter plot of true and predicted values as dots
    plt.scatter(y_test, y_preds,s=5, alpha=0.7)
    # define labels and title
    plt.xlabel('True Sentiment Score')
    plt.ylabel('Predicted Sentiment Score')
    plt.title('True vs Predicted Sentiment Scores')
    # define x and y axis labels
    lims = [-5, 5]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims, color="black")
    # save the figure
    plt.savefig(out_path)
              
def plot_predicted_distribution(predictions, out_path):
    """
    Function to save a plot of the distribution of the values predicted for the unlabelled data
    """
    # clear the previous plot
    plt.clf()
    # create histogram of predictions
    plt.hist(predictions, bins=[-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5, -1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5], ec="black")
    # limit x axis
    plt.xlim([-5,5])
    # define labels and title
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Sentiment Scores')
    # save the figure
    plt.savefig(out_path)
    

    
    
    