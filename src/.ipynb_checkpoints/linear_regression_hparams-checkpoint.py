# sklearn
import numpy as np
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
from tensorboard.plugins.hparams import api as hp
import keras_tuner as kt

# utils
import sys, os
sys.path.append(os.path.join(".."))
from utils.data_utils import split_data, plot_loss

def model_builder(hp):
    linear_model = tf.keras.Sequential([
        #normalizer,
        Dense(units=1)])
    
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.02])
    hp_batch_size = hp.Choice('batch_size', values = [10,20,50])

    linear_model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                loss="mean_absolute_error")

    return linear_model


def main():
    
    # load data
    X = np.load("../output/X_array.npy")
    y = np.load("../output/y_array.npy")
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, 
                                                        random_state=5, shuffle=True)
    
    # define normalizer
    # normalizer = tf.keras.layers.Normalization(axis=-1)
    

    tuner = kt.Hyperband(model_builder,
                         objective='val_loss',
                         max_epochs=10,
                         factor=3,
                            )
    
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    
    
    # train model
    #history = linear_model.fit(
    #    X_train,
    #    y_train,
    #    batch_size=30,
    #    epochs=5,
    #    verbose=1,
    #    validation_split = 0.2)
    
    #results = linear_model.evaluate(
    #    X_test, y_test, verbose=0)
    
    #print(results)
    
    #plot_loss(history, "../output/history_plot.png")
    
if __name__=="__main__":
    main()
    