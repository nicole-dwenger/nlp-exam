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
import keras_tuner as kt
from tensorboard.plugins.hparams import api as hp

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.plugins.hparams import plugin_data_pb2

# utils
import sys, os
sys.path.append(os.path.join(".."))
#from utils.data_utils import split_data, plot_loss 

#HP_NUM_UNITS = hp.HParam('learning', hp.Discrete([16, 32]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001, 0.01, 0.1]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([10, 30, 50, 100]))

#METRIC_ACCURACY = "accuracy"

# Clear logs from previous rounds
!rm -rf ./logs/

# for logging
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_OPTIMIZER, HP_LEARNING_RATE, HP_BATCH_SIZE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy')],
  )

    
def train_test_model(hparams):
    linear_model = tf.keras.models.Sequential([
        #normalizer
        #1 layer
        Dense(units=1)])
      
    # setting the optimizer and learning rate
    optimizer = hparams[HP_OPTIMIZER]
    learning_rate = hparams[HP_LEARNING_RATE]
    #batch_size = hparams[HP_BATCH_SIZE]
    
    if optimizer == "adam":
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
    
# Compile the model with the optimizer and learning rate specified in hparams
    linear_model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error') #,metrics=['accuracy']
    
    linear_model.fit(X_train, y_train, batch_size=batch_size, epochs=40)
    #_, accuracy = linear_model.evaluate(
    #    X_test, y_test, verbose=1)
    #return accuracy
    #OR
    accuracy = linear_model.evaluate(
        X_test, y_test, verbose=0)
    return accuracy

# logging hparams summary for each run
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    
    
 
#def main():
# load data
X = np.load("../output/X_array.npy")
y = np.load("../output/y_array.npy")

y_binary = np.where(y>0, 1, -1)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, random_state=10, shuffle=True, stratify=y_binary)

# GRID SEARCH
session_num = 0

for optimizer in HP_OPTIMIZER.domain.values:
    for learning_rate in HP_LEARNING_RATE.domain.values:
          for batch_size in HP_BATCH_SIZE.domain.values:
              hparams = {
                  #HP_NUM_UNITS: num_units,
                  #HP_DROPOUT: dropout_rate,
                  HP_OPTIMIZER: optimizer,
                  HP_LEARNING_RATE: learning_rate,
                  HP_BATCH_SIZE: batch_size
              }
              run_name = "run-%d" % session_num
              print('--- Starting trial: %s' % run_name)
              print({h.name: hparams[h] for h in hparams})
              run('logs/hparam_tuning/' + run_name, hparams)
              session_num += 1
    
    
#if __name__=="__main__":
#    main()
