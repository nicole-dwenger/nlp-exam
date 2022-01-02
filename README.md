# Danish Lexicon Expansion



[Description](#description) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Contributors](#contributors)

## Description
> This project is an exam project for the course NLP of the Master's in Cognitive Science at Aarhus University.

This repository contains scripts and resources for a project, which aimed to expand the Danish sentiment lexicon [sentida2](https://github.com/Guscode/Sentida2). Specifically, the project used word2vec and fasttext word embeddings, trainined of the [Danish Gigaword corpus](https://gigaword.dk) to predict the sentiment of new words and thereby expand the sentiment lexicon. Linear regression and neural network models were trained on the existing sentiment lexicon to predict the sentiment scores of the 10.000 to 20.000 most frequent words in the Danish language. Subsequently, the sentiment model [Asent](https://github.com/KennethEnevoldsen/asent) was used to compare the base lexicon [sentida2](https://github.com/Guscode/Sentida2) to the extended lexica. 
 

## Repository Structure

```
|-- data/                       # Directory containing prepared training and prediction data
    |-- unlabelled_data/        # Directory for prepared data for prediction of sentiments, i.e. lexicon expansion
    |-- labelled_data/          # Directory for prepared data to develop and evaluate model for sentiment prediction
    
|-- embeddings/                 # Directory containing embeddings (not on github)
|-- lemmas/                     # Directory contining list of the most frequenly used lemmas (not on github)
|-- lexicons/                   # Directory containing base lexicon sentida2 (not on github)

|-- output/                     # Directory for outputs 
    |-- grid_search             # Directory containing output of grid search of linear regression and neural network models
    |-- sentiment_prediction    # Directory containing output of final linear regression and neural network model, and sentiment predictions  
    |-- lexicon_evaluation      # Directory containing output of lexicon evaluation of sentiment task
 
|-- src/                        # Directory for python scripts
    |-- notebooks               # Directory for notebooks used for data preprocessing
    |-- grid_search.py          # Script for grid search of linear regression and neural network parameters
    |-- sentiment_prediction.py # Script for model evaluation on test data and sentiment prediction 
    |-- lexicon_evaluation.py.  # Script for lexicon evaluation of sentiment classification task
    
|-- utils/                      # Directory for utility scripts
    |-- data_utils.py           # Utility functions for data processing
    |-- model_utils.py          # Utility functions for model training and evaluation
    |-- linear_regression.py    # Linear regression class
    |-- neural_network.py       # Neural network class
    |-- classification_utils.py # Utility functions for sentiment classification
    |-- twittertokens.py        # Defines tokens for Twitter API (not on github)

|-- README.md
|-- requirements.txt            # Dependencies to run scripts and notebooks
|-- install_requirements.txt     # Bash script to install dependencies and language models
```

The files described in the repository structure, which are not on GitHub can be provided on request. 

## Usage

**!** The scripts have only been tested on Linux, using Python 3.9.1.  

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. Dependencies are listed in the `requirements.txt` file. To install all necessary requirements, including the Danish spaCy language model, the bash script `install_requirements.sh` shoud be used. 

## Contributors
This project was developed in a joined effort by Sofie Ditmer, Louise Nyholm Jensen and Nicole Dwenger.