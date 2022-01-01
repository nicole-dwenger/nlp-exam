"""
SCRIPT FOR TRAINING LINEAR REGRESSION MODEL OR NEURAL NETWORK 
AND PERFORM SENTIMENT PREDICTION ON UNLABELLED DATA

This script trains a linear regression or neural network model to predict sentiment scores based on 
word2vec or fasttext word embeddings. It will evaluate the model on test data, and then generate
predictions for unlabelled word embeddings. Further, several plots of the predictions for the train, 
test and unlabelled data will be generated.
  
Input:
    - --embedding_type, str, required, "word2vec" or "fasttext", defines embeddings to use for training
    - --model_type: str, required, "linear_regression" or "neural_network", defines which model to perform grid search on
    - --optimizer: str, required, "sgd" or "adam", defines the optimiser used for training
    - --lr: float, required, defines the learning rate used for training
    - --batch_size: int, required, defines the batch size used in training
    - --epochs: int, required, defines number of epochs for training
    - --hidden_layers: list of int, required if neural network, example: 150 50, defines the number of and the nodes in the hidden layers
    - --activation: str, required if neural network, "relu" or "softmax", defines actiation function of nodes
    - --output_path, str, optional, default: "../output/", defines path for output
  
Output saved in {output_path}/sentiment_prediction/{model_type}/{embedding_type}/
    - model_history.png: plots history of model training
    - error_distibuton.png: plots distribution of error for predictions on labelled test data
    - regression_error_d_test.png: plots diagonally true and predicted sentiment scores on labelled test data
    - regression_error_d_train.png: plots diagonally true and predicted sentiment scores on labelled train data
    - regression_error_test.png: plots subset of true and predicted sentiment scores on labelled test data
    - regression_error_train.png: plots subset of true and predicted sentiment scores on labelled train data
    - sentiment_predictions.csv: csv file containing predictions of unlabelled data
    - distribution_predictions.png: plots distribution of sentiment predictions on unlabelled data
    - results.txt: writes model info, train and test results, and information of predictions on unlabelled data
"""

# --- DEPENDENCIES ---

# basics
import numpy as np
import pandas as pd
import itertools
import argparse
import sys, os
from statistics import mean, median

# tensorflow
# hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(12) # set a seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

# utils
sys.path.append(os.path.join(".."))
from utils.data_utils import prepare_data, load_data_to_predict
from utils.linear_regression import LinearRegression
from utils.neural_network import NeuralNetwork
from utils.model_utils import (write_model_results, unique_path, plot_regression_error, 
                               plot_error_distribution,plot_regression_error_diagonal,
                               plot_predicted_distribution)


# -- MAIN FUNCTION --

def main(embedding_type, model_type, optimizer, lr, batch_size, epochs, hidden_layers, activation, output_path):
    
     # --- PREPARATIONS ---
    
    # define output path
    output_path = unique_path(os.path.join("..", "output", "sentiment_prediction", model_type, embedding_type))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # load training and test data for model
    X_train, X_test, y_train, y_test = prepare_data(embedding_type, n_splits=2)
    
    # load data to predict
    embeddings_to_predict, words_to_predict = load_data_to_predict(embedding_type)
    
     # --- MODEL TRAINING AND EVALUATION ---
    
    print("[INFO] Initialising model training...")
    
    if model_type == "linear_regression":
        model = LinearRegression(optimizer, lr, batch_size, epochs)
        
    elif model_type == "neural_network":
        model = NeuralNetwork(optimizer, lr, batch_size, epochs, hidden_layers, activation)
        
    # train model 
    model.train(X_train, y_train, X_test, y_test, verbose=0, save_model=True, output_path=output_path)
    # evaluate on test
    model.evaluate(X_test, y_test)
    
    # --- SAVE MODEL METRICS AND EVALUATION ---
    
    # save model results
    write_model_results(model, model_type, optimizer, lr, batch_size, epochs, 
                            hidden_layers, activation, f"{output_path}/results.txt")
    
    # plot results of predictions on training data
    y_preds = model.model.predict(X_train)
    plot_regression_error(y_train, y_preds, f"{output_path}/regression_error_train.png")
    plot_regression_error_diagonal(y_train, y_preds, f"{output_path}/regression_error_d_train.png")
    
    # plot results of predictions on test data
    y_preds = (model.model.predict(X_test)).flatten()
    plot_regression_error(y_test, y_preds, f"{output_path}/regression_error_test.png")
    plot_regression_error_diagonal(y_test, y_preds, f"{output_path}/regression_error_d_test.png")
    
    # plot distribution of erros
    plot_error_distribution(y_test, y_preds, f"{output_path}/error_distribution.png")
    
    print("[INFO] Saved model results!")
        
    # --- SENTIMENT PREDCICTIONS ---
    
    print("[INFO] Predicting new sentiments...")
    
    # predict on new embeddings to be added to lexicon
    predictions = model.model.predict(embeddings_to_predict)
    predictions = list(itertools.chain(*predictions))
    
    # restrict predictions to be between -5 and 5
    restricted_preds = [5 if sent>5 else sent for sent in predictions]
    restricted_preds = [-5 if sent<-5 else sent for sent in restricted_preds]
          
    # match predictions with orginal words/lemmas
    predictions_df = pd.DataFrame(list(zip(words_to_predict, predictions, restricted_preds)), 
                                  columns = ["word", "predicted_sentiment", "restricted_sentiments"])
    
    # save predictions
    predictions_df.to_csv(f"{output_path}/sentiment_predictions.csv", index=False)
    
    # plot distribution of predicted values
    plot_predicted_distribution(restricted_preds, f"{output_path}/distribution_predictions.png")
    
    # save mean and median of predicted values
    with open(f"{output_path}/results.txt", 'a') as file:
            file.write(f"METRICS OF PREDICTIONS FOR UNLABELLED WORD EMBEDDINGS:\n")
            file.write(f"Mean: {mean(restricted_preds)}\n")
            file.write(f"Median: {median(restricted_preds)}\n")
            file.close()
    
    print(f"[INFO] Done. Predicted sentiments saved in {output_path}")

    
if __name__=="__main__":
    
    # -- ARGUMENT PASER ---

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embedding_type', type=str, required=True,
                        help="word2vec or fasttext")
   
    parser.add_argument('--model_type', type=str, required=True,
                       help="linear_regression or neural_network") 
    
    parser.add_argument('--optimizer', type=str, required=True,
                       help="adam or sgd")
    
    parser.add_argument('--lr', type=float, required=True,
                       help="float defining learning rate of model")
    
    parser.add_argument('--batch_size', type=int, required=True,
                       help="int defining batch size used in model training")
    
    parser.add_argument('--epochs', type=int, required=True,
                       help="int defining number of epochs for training")
    
    parser.add_argument('--hidden_layers', nargs="*",type=int, required=False,
                       help="Number of int defining hidden layers, e.g. 150 50")
    
    parser.add_argument('--activation', type=str, required=False,
                       help="relu or softmax")    
    
    parser.add_argument('--output_path', type=str, default="../output/", 
                        help='Path to output directory')
    
    args = parser.parse_args()
                        
    # -- RUN MAIN FUNCTION ---

    main(embedding_type = args.embedding_type, 
         model_type = args.model_type,
         optimizer=args.optimizer,
         lr=args.lr,
         batch_size=args.batch_size,
         epochs=args.epochs,
         hidden_layers=args.hidden_layers,
         activation=args.activation,
         output_path=args.output_path)
    