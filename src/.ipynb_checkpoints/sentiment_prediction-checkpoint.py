import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import plot_model

def neural_network(optimizer=SGD(0.01), loss="binary_crossentropy", metrics="accuracy"):
    
    # create a sequential model
    model = Sequential()
    
    model.add(Dense(500, input_shape=(300,)))
    model.add(Dense(100, activation="linear"))
    model.add(Dense(10, activation="linear"))
    model.add(Dense(1, activation="sigmoid"))

    # categorical cross-entropy, optimizer defined in function call
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=metrics)

    # return the compiled model
    return model

    

def main():
    
    # load data
    X = np.load("../output/X_array.npy")
    y = np.load("../output/y_tarray.npy")
    
    y = np.where(y<=0, -1, y)
    y = np.where(y>0, 1, y)
    y = y.astype(int)
    
    
    
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    
    print("loaded data!")
    
    # split data
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
    
    print(y_val)
    
    
    # turn data into tensors
    #X_train, y_train = np_to_transformer(X_train, y_train)
    #X_test, y_test = np_to_transformer(X_test, y_test)
    #X_val, y_val = np_to_transformer(X_val, y_val)
    
    # if linear regression
    #if model == "linear_regression":
        
        
    #elif model == "neural_network":
    
    nn_model = neural_network()

    history = nn_model.fit(X_train, y_train, validation_data = (X_val, y_val), 
                           batch_size = 100, epochs = 20, verbose = 1)

    predictions = nn_model.predict(X_val, 50)
    
    print(predictions[:20])
    print(predictions.argmax(axis=1)[:20])
    
    predictions = np.where(predictions<=0.5, 0, 1) 
    
    print(classification_report(y_val,
                                predictions))

    
    print(report)
    print(set(y_val) - set(predictions))
    
    
    
if __name__=="__main__":
    main()

