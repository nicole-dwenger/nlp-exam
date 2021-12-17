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
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import plot_model

# utils
import sys, os
sys.path.append(os.path.join(".."))
from utils.data_utils import split_data, plot_loss
from contextlib import redirect_stdout


def main():
    
    # load data
    X = np.load("../output/X_array.npy")
    y = np.load("../output/y_array.npy")
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, 
                                                        random_state=5, shuffle=True)
    
    # define neural network
    model = Sequential([
      #norm,
      InputLayer(input_shape=(300,)),
      Dense(50, activation='relu'),
      Dense(100, activation='relu'),
      Dense(1)])
    
    # compile model
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.01))
    
    # train model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        verbose=1, epochs=100)
    
    # results
    results = model.evaluate(
        X_test, y_test, verbose=0)
    
    print(results)
    
    # plot
    plot_loss(history, "../output/dnn_history_plot.png")
    
    # summary
    with open("../output/model_summary.txt", "w") as file:
        with redirect_stdout(file):
            model.summary()
            
    plot_model(model, to_file = "../output/model_plot.png", show_shapes = True, show_layer_names = True)
    
if __name__=="__main__":
    main()
    
    

    
    
    