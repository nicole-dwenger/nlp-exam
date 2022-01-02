# Source Code Directory

### notebooks/
This directory contains the notebooks `prepare_labelled_data.ipynb` and `prepare_unlabelled_data.ipynb` to prepare the labelled, and unlabelled data, of which the output is stored in corresponding directories in `data/`. The notebook `expanded_lexicon_contribution.ipynb` calculates the contribution of the lexicon expansion for the three different datasets used in the `lexcion_evaluation.py`

### grid_search.py
This script performs grid search on a linear regression or neural network model. Functions and classes for this script are stored in the `utils/` directory. The output of this script is stored in the corresponding directories in `output/grid_search/`. 

### sentiment_prediction.py
This script trains a linear regression or neural network model on the combined labelled train and validation data, using defined parameters. Then the model is evaluated on unseen test data and used for predictions on the unlabelled data. The output is stored in the `output/sentiment_prediction/` directory. 

### lexicon_evaluation.py
This script performs evaluation of the base [sentida2](https://github.com/Guscode/Sentida2) lexicon and expanded lexicon on three benchmark sentiment classification datasets. For this, the rule-based model [Asent](https://github.com/KennethEnevoldsen/asent) is used. The benchmark evaluation is based on similar benchmarks performed by [danlp](https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks). The output is stored in `output/lexicon_evaluation/`. 