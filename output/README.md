# Output Directories

### Grid Search
This directory contains results of the `grid_search.py` script. The folder structure corresponds to the different kinds of models and word embeddings used. 

### Sentiment Prediction
This directory contains results of the `sentiment_prediction.py`script. Thus, it contains visualisations and results for the labelled test data and predictions for the unlabelled data. The folder structure corresponds to the different kinds of models (trained on optimal hyperparameters as obtained by grid search) and word embeddings used.  

### Lexicon Evaluation
This directory contains results of the lexicon evaluation on three sentiment classification tasks, defined in `lexicon_evaluation.py` The folder structure corresponds to the different kinds of models and word embeddings used, sentiment expansions are stored in corresponding directories in `Sentiment Prediction/`. Note, that the evaluation were run on the 02/01/2022. Thus, if the evaluation is replicated at another point, results may differ due to the changing nature of the Twitter Dataset`