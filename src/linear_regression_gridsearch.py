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

# utils
import sys, os
sys.path.append(os.path.join(".."))
from utils.data_utils import split_data, plot_loss

def build_model(learn_rate):
    # create model
    linear_model = tf.keras.Sequential([
        #normalizer,
        Dense(units=1)])
    
    # Compile model
    optimizer = tf.optimizers.SGD(learn_rate)
    linear_model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error')
    
    return linear_model


def main():
    
    # load data
    X = np.load("../output/X_array.npy")
    y = np.load("../output/y_array.npy")
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, 
                                                        random_state=5, shuffle=True)
    
    # define normalizer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    
    model = KerasClassifier(build_fn=build_model, verbose=0)
    
    # define possible paramteres
    learn_rate = [0.01,0.02,0.1,0.2]
    batch_size = [10,20,30,60]
    
    # define grid
    param_grid = dict(learn_rate=learn_rate, batch_size=batch_size)
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5, scoring="neg_mean_absolute_error")
    grid_result = grid.fit(X_train, y_train)
    
    # summarize results
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # compile model
    #linear_model.compile(
    #    optimizer=tf.optimizers.SGD(learning_rate=learning_ra),
    #    loss='mean_absolute_error')
    
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
    