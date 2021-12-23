import numpy as np
import os, itertools
import matplotlib.pyplot as plt
from datetime import datetime

def plot_history(history, out_history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [SENTIMENT]')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_history)
    
def get_hpcombinations(model_type):
    
    if model_type == "linear_regression":
        
        hpdict = {"optimizer": ['adam', 'sgd'],
                  "learning_rate": [0.001, 0.01],
                  "batch_size": [10, 50, 100],
                  "epochs": [50, 100, 150, 200]}

    elif model_type == "neural_network":
        
        hpdict = {"optimizer": ['adam', 'sgd'],
                  "learning_rate": [0.001, 0.01],
                  "batch_size": [10, 50, 100],
                  "epochs": [50, 100, 150, 200],
                  "hidden_layers": [[500,10], [500,150,10], [500,150,50], [750,150,50]],
                  "activation": ["relu", "softmax"]}
        
    # get possible parameter options
    hp_combinations = itertools.product(*(hpdict[key] for key in hpdict.keys()))
    hp_combinations = list(hp_combinations)
    
    return hp_combinations

def write_model_results(model, model_type, optimizer, lr, batch_size, epochs, 
                        hidden_layers, activation, output_path):
    
    if model_type == "linear_regression":
    
        with open(f'{output_path}/model_results.txt', 'a') as file:
            file.write(f"MODEL USED FOR PREDICTIONS: {model_type}\n")
            file.write(f"{datetime.now()}\n")
            file.writelines([f"optimizer: {optimizer}\n",
                             f"lr: {lr}\n", 
                             f"batch_size: {batch_size}\n",
                             f"epochs: {epochs}\n",
                             f"train_loss: {model.train_loss}\n",
                             f"val_loss: {model.val_loss}\n", 
                             f"test_loss: {model.test_loss}\n\n\n"])
        
    elif model_type == "neural_network":
        
        with open(f'{output_path}/model_results.txt', 'a') as file:
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

    
def unique_path(filepath):
    """
    Create unique filename by enumerating if path exists already 
    Input:
      - filepath: desired fielpath
    Returns:
      - new_path: enumerated if it exists already
    """ 
    # If the path does not exist
    if not os.path.exists(filepath):
        # Keep the original filepath
        return filepath
    
    # If path exists:
    else:
        i = 1
        # Split the path and append a number
        path, ext = os.path.splitext(filepath)
        # Add extension
        new_path = "{}_{}{}".format(path, i, ext)
        
        # If the extension exists, enumerate one more
        while os.path.exists(new_path):
            i += 1
            new_path = "{}_{}{}".format(path, i, ext)
            
        return new_path
    
    
def plot_regression_error(y_test, y_preds, out_path):

    _, ax = plt.subplots()

    ax.scatter(x = range(0, y_test.size), y=y_test, c = 'blue', label = 'Actual', alpha = 0.3)
    ax.scatter(x = range(0, y_preds.size), y=y_preds, c = 'red', label = 'Predicted', alpha = 0.3)

    plt.title('Actual and predicted values')
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.xlim([0,1000])
    plt.legend()
    plt.show()
    plt.savefig(out_path)
        
        
def plot_error_distribution(y_test, y_preds, out_path):
    
    plt.clf()
    error = y_preds - y_test
    plt.hist(error, bins=100)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')
    plt.savefig(out_path)
    
def plot_regression_error_diagonal(y_test, y_preds, out_path):
    
    plt.clf()
    a = plt.axes(aspect='equal')
    plt.scatter(y_test, y_preds)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [-5, 5]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig(out_path)
    

    
    
    