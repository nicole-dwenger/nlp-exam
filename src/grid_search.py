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


def main(model_type, embedding_type, output_path):
    
    # PREPARATIONS #
    
    # define output path
    output_path = os.path.join(output_path, "grid_search", model_type, embedding_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # write intro to file
    output_file = os.path.join(output_path, "gridsearch_results.txt")
    with open(output_file, 'a') as file:
        file.write(f"\nRun from {datetime.now()}\n")
        file.write(f"Results of Grid Search for {model_type} and {embedding_type}:\n")
        file.write(f"Order of Params: optimizer, learning_rate, batch_size, epochs, hidden_layers, activation\n\n")
    
    # prepare data
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(embedding_type, n_splits=3)
    
    # prepare list to save mae values
    val_losses = []
    
    # get hyperparameters
    hpcombiantions = get_hpcombinations(model_type)
    
    # TRAIN MODELS #
    
    # loop through the possibel hyper options:
    for combination in tqdm(hpcombiantions):
        
        if model_type == "linear_regression":
            optimizer, learning_rate, batch_size, epochs = combination[:4]
            model = LinearRegression(optimizer, learning_rate, batch_size, epochs)
            
        elif model_type == "neural_network":
            optimizer, learning_rate, batch_size, epochs, hidden_layers, activation = combination[:6]
            model = NeuralNetwork(optimizer, learning_rate, batch_size, epochs, hidden_layers, activation)
            
        # train the model
        model.train(X_train, y_train, X_val, y_val, verbose=0, save_model=False)
        # append values
        val_losses.append(model.val_loss)
        
        # save results to file
        with open(output_file, 'a') as file:
            file.writelines([f"{combination}\n", 
                             f"train_loss: {model.train_loss}, val_loss: {model.val_loss}\n\n"])
            
    # FIND BEST MODEL #
        
    # get the best combination
    min_value = min(val_losses)
    best_idx = val_losses.index(min_value)
    best_combination = hpcombiantions[best_idx]
    
    if model_type == "linear_regression":
        optimizer, learning_rate, batch_size, epochs = best_combination[:4]
        
    elif model_type == "neural_network":
        optimizer, learning_rate, batch_size, epochs, hidden_layers, activation = best_combination[:6]
        
    
    # save best model to file
    with open(output_file, 'a') as file:
            file.write("\n---------\n")
            file.write("BEST MODEL\n")
            file.writelines([f"{best_combination}\n", f"val_loss: {min_value}"])
            file.write("\n---------\n")

    
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, required=True,
                        help='String that specifies the model type')
    
    parser.add_argument('--embedding_type', type=str, required=True,
                        help='String that specifies the wordembeddings')
    
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to output directory', default="../output")
    
    args = parser.parse_args()

    main(model_type=args.model_type,
         embedding_type=args.embedding_type, 
         output_path=args.output_path)

