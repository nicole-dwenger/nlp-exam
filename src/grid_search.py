"""
SCRIPT TO PERFORM GRID SEARCH FOR LINEAR REGRESSION OR NEURAL NETWORK 

This script uses grid search of a combinaton of parameters defined in ../utils/model_utils.py to find
the optimal combination of parameters for a linear regression or neural network model, 
trained to predict sentiment scores based on word embeddings.
  
Input:
  - --model_type: str, required, "linear_regression" or "neural_network", defines which model to perform grid search on
  - --embedding_type, str, required, "word2vec" or "fasttext", defines embeddings to use for training
  - --output_path, str, optional, default: "../output/", defines path for output
  
Output saved in {output_path}/grid_search/{model_type}/{embedding_type}/
  - gridsearch_results.txt: results of grid search
"""

# --- DEPENDENCIES ---

# basics
import sys, os
import argparse
import itertools
from tqdm import tqdm
from datetime import datetime

# utils
sys.path.append(os.path.join(".."))
from utils.data_utils import prepare_data
from utils.linear_regression import LinearRegression
from utils.neural_network import NeuralNetwork
from utils.model_utils import get_hpcombinations


# -- MAIN FUNCTION --

def main(model_type, embedding_type, output_path):
    
    # --- PREPARATIONS ---
    
    # define output path
    output_path = os.path.join(output_path, "grid_search", model_type, embedding_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # write info to file
    output_file = os.path.join(output_path, "gridsearch_results.txt")
    with open(output_file, 'a') as file:
        file.write(f"\nRun from {datetime.now()}\n")
        file.write(f"Results of Grid Search for {model_type} and {embedding_type}:\n")
        file.write(f"Order of Params: optimizer, learning_rate, batch_size, epochs, hidden_layers, activation\n\n")
    
    # prepare data
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(embedding_type, n_splits=3)
    
    # prepare list to save mae values
    val_losses = []
    
    # get hyperparameter combinations
    hpcombiantions = get_hpcombinations(model_type)
    
    # --- TRAIN MODELS ---
    
    # loop through the possible hyperparameter combinatons:
    for combination in tqdm(hpcombiantions):
        
        # define linear regression
        if model_type == "linear_regression":
            optimizer, learning_rate, batch_size, epochs = combination[:4]
            model = LinearRegression(optimizer, learning_rate, batch_size, epochs)
            
        # define neural network
        elif model_type == "neural_network":
            optimizer, learning_rate, batch_size, epochs, hidden_layers, activation = combination[:6]
            model = NeuralNetwork(optimizer, learning_rate, batch_size, epochs, hidden_layers, activation)
            
        # train the specified model
        model.train(X_train, y_train, X_val, y_val, verbose=0, save_model=False)
        # append values to validation losses to find the best one
        val_losses.append(model.val_loss)
        
        # save results to file
        with open(output_file, 'a') as file:
            file.writelines([f"{combination}\n", 
                             f"train_loss: {model.train_loss}, val_loss: {model.val_loss}\n\n"])
            
    # --- FIND BEST MODEL ---
        
    # get the best lowest val loss
    min_value = min(val_losses)
    # find the corresponding combinaton of parameters
    best_idx = val_losses.index(min_value)
    best_combination = hpcombiantions[best_idx]
    
    # get the parameters of the best model for linear regression
    if model_type == "linear_regression":
        optimizer, learning_rate, batch_size, epochs = best_combination[:4]
        
    # get the parameters of the best model for neural network
    elif model_type == "neural_network":
        optimizer, learning_rate, batch_size, epochs, hidden_layers, activation = best_combination[:6]
        
    
    # append best model to file
    with open(output_file, 'a') as file:
            file.write("\n---------\n")
            file.write("BEST MODEL\n")
            file.writelines([f"{best_combination}\n", f"val_loss: {min_value}"])
            file.write("\n---------\n")

    
if __name__=="__main__":

    # -- ARGUMENT PASER ---
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, required=True,
                        help='linear_regression or neural_network')
    
    parser.add_argument('--embedding_type', type=str, required=True,
                        help='word2vec or fasttext')
    
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to output directory', default="../output")
    
    args = parser.parse_args()
    
    # -- RUN MAIN FUNCTION ---

    main(model_type=args.model_type,
         embedding_type=args.embedding_type, 
         output_path=args.output_path)

