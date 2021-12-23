import numpy as np
import pandas as pd
import itertools
import argparse
import sys, os

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.random.set_seed(12)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# utils
sys.path.append(os.path.join(".."))
from utils.data_utils import prepare_data, load_data_to_predict
from utils.linear_regression import LinearRegression
from utils.neural_network import NeuralNetwork
from utils.model_utils import (write_model_results, unique_path, plot_regression_error, 
                               plot_error_distribution,plot_regression_error_diagonal)

def main(embedding_type, model_type, optimizer, lr, batch_size, epochs, hidden_layers, activation, output_path):
    
    # define output path
    output_path = unique_path(os.path.join("..", "output", "sentiment_prediction", model_type, embedding_type))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # load training and test data for model
    X_train, X_test, y_train, y_test = prepare_data(embedding_type, n_splits=2)
    
    # load data to predict
    embeddings_to_predict, words_to_predict = load_data_to_predict(embedding_type)
    
    ### TRAIN MODEL ###
    
    print("[INFO] Initialising model training...")
    
    if model_type == "linear_regression":
        model = LinearRegression(optimizer, lr, batch_size, epochs)
        
    elif model_type == "neural_network":
        model = NeuralNetwork(optimizer, lr, batch_size, epochs, hidden_layers, activation)
        
    # train model 
    model.train(X_train, y_train, X_test, y_test, verbose=0, save_model=True, output_path=output_path)
    # evaluate on test
    model.evaluate(X_test, y_test)
    # save model results
    write_model_results(model, model_type, optimizer, lr, batch_size, epochs, 
                            hidden_layers, activation, output_path)
    
    y_preds = model.model.predict(X_train)
    plot_regression_error(y_train, y_preds, f"{output_path}/regression_error_train.png")
    plot_regression_error_diagonal(y_train, y_preds, f"{output_path}/regression_error_d_train.png")
    
    y_preds = model.model.predict(X_test)
    plot_regression_error(y_test, y_preds, f"{output_path}/regression_error_test.png")
    plot_regression_error_diagonal(y_test, y_preds, f"{output_path}/regression_error_d_test.png")
    
    plot_error_distribution(y_test, y_preds, f"{output_path}/error_distribution.png")
    
    print("[INFO] Saved model results!")
        

    ### MAKE PREDICTIONS ###
    
    print("[INFO] Predicting new sentiments...")
    
    # predict on new embeddings to be added to lexicon
    predictions = model.model.predict(embeddings_to_predict)
    predictions = list(itertools.chain(*predictions))
    
    # Restrict predictions to be between -5 and 5
    restricted_preds = [5 if sent>5 else sent for sent in predictions]
    restricted_preds = [-5 if sent<-5 else sent for sent in restricted_preds]
          
    # match predictions with words
    predictions_df = pd.DataFrame(list(zip(words_to_predict, predictions, restricted_preds)), 
                                  columns = ["word", "predicted_sentiment", "restricted_sentiments"])
    
    # save predictions to /output
    predictions_df.to_csv(f"{output_path}/sentiment_predictions.csv", index=False)
    
    print(f"[INFO] Done. Predicted sentiments saved in {output_path}")

    
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--embedding_type', type=str, help='String that specifies the wordembeddings', required=True)
    parser.add_argument('--model_type', type=str, required=True) 
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--hidden_layers', nargs="*",type=int, required=False)
    parser.add_argument('--activation', type=str, required=False)    
    parser.add_argument('--output_path', type=str, default="../output/", help='Path to output directory')
    
    args = parser.parse_args()

    main(embedding_type = args.embedding_type, 
         model_type = args.model_type,
         optimizer=args.optimizer,
         lr=args.lr,
         batch_size=args.batch_size,
         epochs=args.epochs,
         hidden_layers=args.hidden_layers,
         activation=args.activation,
         output_path=args.output_path)
    