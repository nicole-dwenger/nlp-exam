import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def nn_model(optimizer="adam", loss="mse", metrics="accuracy"):
    
    # create a sequential model
    model = Sequential()
    
    model.add(Dense(100, input_shape=(300,), activation="relu"), )
    model.add(Dense(1))

    # categorical cross-entropy, optimizer defined in function call
    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=metrics)

    # return the compiled model
    return model
    
    
    
    
    

    
    
    
    
    
    
    


