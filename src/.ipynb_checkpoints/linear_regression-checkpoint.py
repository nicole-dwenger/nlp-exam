import numpy as np
import pandas as pd
import itertools

# sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import plot_model
import keras_tuner as kt

# utils
import sys, os
sys.path.append(os.path.join(".."))
from utils.data_utils import split_data, plot_loss 

def main():
    
    # load data
    X = np.load("../output/X_array.npy")
    y = np.load("../output/y_array.npy")
    
    # load embeddings and words to predict
    embeddings_to_predict = np.load("../output/X_array_embeds_to_predict.npy")
    words_to_predict = np.load("../output/X_array_words_to_predict.npy")
    words_to_predict = words_to_predict.tolist() # convert to list
    
    # make binary for stratification (making sure there are an equal number of sentiments in train/test)
    y_binary = np.where(y>0, 1, -1)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        train_size=0.8, 
                                                        random_state=10, 
                                                        shuffle=True, 
                                                        stratify=y_binary)
    # define normalizer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    
    # define linear model
    linear_model = tf.keras.Sequential([
        #normalizer,
        Dense(units=1)])
    
    # compile model
    linear_model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=0.1),
        loss='mean_absolute_error')
    
    # train model
    history = linear_model.fit(
        X_train,
        y_train,
        batch_size=30,
        epochs=5,
        verbose=1,
        validation_split = 0.2)
    
    # evaluate
    results = linear_model.evaluate(
        X_test, y_test, verbose=1)
    
    # save loss plot
    plot_loss(history, "../output/linear_history_plot.png")
    
    # predict on new embeddings to be added to lexicon
    predictions = linear_model.predict(embeddings_to_predict)
    predictions = list(itertools.chain(*predictions))
    
    # Restrict predictions to be between -5 and 5
    restricted_preds = [5 if sent>5 else sent for sent in predictions]
    restricted_preds = [-5 if sent<-5 else sent for sent in restricted_preds]
    
    print(max(restricted_preds), min(restricted_preds))
          
    # match predictions with words
    predictions_df = pd.DataFrame(list(zip(words_to_predict, predictions, restricted_preds)), columns = ["word", "predicted_sentiment", "restricted_sentiments"])
    
    # save predictions to /output
    predictions_df.to_csv("../output/linear_sentiment_predictions.csv")
    
    
if __name__=="__main__":
    main()
    