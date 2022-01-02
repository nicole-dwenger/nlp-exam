"""
SCRIPT TO EVALUATE LEXICONS ON BENCHMARK SENTIMENT CLASSIFICATION TASKS

This script uses the sentiment model Asent and the base sentida2 or extended lexicon for
benchmark sentiment classification tasks on three datasets:
- EuropalSentiment1
- LccSentiment 
- TwitterSent

The code and process is adopted from the Alexandra Institute (Danlp):
- https://github.com/alexandrainst/danlp/tree/master/examples/benchmarks
  
Input:
  - --lexicon: str, required, "sentida2" or "sentida2_expanded", defines lexicon to use
  - --model_type: str, required, "linear_regression" or "neural_network", if expanded lexicon is used, defines which expansion
  - --embedding_type: str, required, "word2vec" or "fasttext", if expanded lexicon is used, defines which expansion
  - --pos_threshold: float, required, defines above which compound score a document is classified as positive
  - --neg_threshold: float, required, defines below which compound score a document is classified as negative
  - --output_path: str, default: "../output/", defines path for output
  
Output saved by in {output_path}/lexicon_evaluation/{model_type}/{embedding_type}/
  - f1_report_{lexicon}_{pos_threshold}_{neg_threshold}.txt: f1 report for all three datasets
"""

# --- DEPENDENCIES ---

# basics
import sys, os
import argparse
import pandas as pd

# nlp
import spacy
import asent
from asent.component import Asent
from asent.lang.da import LEXICON, NEGATIONS, INTENSIFIERS, CONTRASTIVE_CONJ
from danlp.datasets import EuroparlSentiment1, LccSentiment, TwitterSent

# utils
sys.path.append(os.path.join(".."))
from utils.twittertokens import setup_twitter
from utils.classification_utils import (sentiment_score_to_label, 
                                        sentiment_score_to_label_asent, 
                                        f1_class, f1_report)


# --- MAIN FUNCTION ---

def main(lexicon, model_type, embedding_type, pos_threshold, neg_threshold, output_path):
    
    # --- PREPARATIONS ---
    
    # define output path
    output_path = os.path.join(output_path, "lexicon_evaluation", model_type, embedding_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # define name of output file
    output_file = f"{output_path}/f1_report_{lexicon}_{pos_threshold}_{neg_threshold}.txt"
        
    # create output txt file and save informaton
    with open(output_file, 'a') as f:
        f.write(f"LEXICON EVAL RESULTS\n")
        f.write(f"Model: {model_type}, Embeddings: {embedding_type}, Lexicon: {lexicon}\n\n")
        f.close()
        
    # --- LEXICON ---
        
    # load base sentida2 lexicon
    LEXICON = pd.read_csv("../lexicons/sentida2_lexicon.csv")
    
    # prepare lexicon 
    if lexicon == "sentida2":
        LEXICON = LEXICON

    # if extended lexicon should be used, expand sentida2
    elif lexicon == "sentida2_expanded":
        print("[INFO] Will expand base sentida2!")
        # load defined expansion and append
        EXPANSION = pd.read_csv(f"../output/sentiment_prediction/{model_type}/{embedding_type}/sentiment_predictions.csv")
        EXPANSION = EXPANSION.rename(columns={"word":"word", "restricted_sentiments":"score"})
        EXPANSION = EXPANSION.drop(columns =["predicted_sentiment"])
        LEXICON = pd.concat([LEXICON, EXPANSION])
        
    # put lexicon into dictionary
    LEXICON = dict(zip(LEXICON.word, LEXICON.score))
    print(f"[INFO] {lexicon} prepared!")
        
    # --- SENTIMENT MODEL ---
    
    # load spacy nlp pipeline
    nlp = spacy.load("da_core_news_lg")

    # add Asent to nlp pipeline
    Asent(nlp, name = "classification", 
          lexicon=LEXICON, 
          negations=NEGATIONS, 
          intensifiers=INTENSIFIERS, 
          contrastive_conjugations=CONTRASTIVE_CONJ,
          lowercase=True,
          lemmatize=True, 
          force=True)
    
    # define function to retrieve document compound score
    def asent_score(sent):
        doc = nlp(sent)
        return doc._.polarity.compound
    
    # --- SENTIMENT CLASSIFICATION ---

    # loop through each of the datasets and load them
    for dataset in ['euparlsent','lccsent','twitter']:
        
        # if it is europalsent or lccsent data
        if dataset in ["euparlsent", "lccsent"]:
            if dataset == "euparlsent":
                data = EuroparlSentiment1()
                df = data.load_with_pandas()
            elif dataset == "lccsent":
                data = LccSentiment()
                df = data.load_with_pandas()
            
            print(f"[INFO] Data of {dataset} loaded...")
            
            # get the compound score for each document
            df['pred'] = df.text.map(asent_score)
            
            # turn predictions into classes
            prediction_classes=[]
            for index, row in df.iterrows():
                prediction_classes.append(sentiment_score_to_label_asent(row["pred"], pos_threshold, neg_threshold))
            df["pred"] = prediction_classes
            
            # turn true values into classes
            df['valence'] = df['valence'].map(sentiment_score_to_label)
            
            # generate f1 report for evaluation
            f1_report(df['valence'], df['pred'], 'Asent', dataset, output_file=output_file)

        # if it is twitter dataset
        elif dataset == "twitter":
            # use access tokens from developer account
            setup_twitter()
            # load data, concatenating val and train data
            twitSent = TwitterSent()
            df_val, _ = twitSent.load_with_pandas()
            # df = pd.concat([df_val, df_train])
            
            # evaluate on the validation data, in line with danlp
            df = df_val
            print(len(df))
            
            # get the compound score for each document
            print(f"[INFO] Data of {dataset} loaded...")
            df['pred'] = df.text.map(asent_score)
            
            # turn predictions into classes
            prediction_classes=[]
            for index, row in df.iterrows():
                prediction_classes.append(sentiment_score_to_label_asent(row["pred"], pos_threshold, neg_threshold))
            df["pred"] = prediction_classes
            
            # generate f1 report for evaluation
            f1_report(df['polarity'], df['pred'], 'Asent', "twitter_sentiment(val)", output_file=output_file)
            

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    # -- ARGUMENT PASER ---
    
    parser.add_argument('--lexicon', type=str, required=True,
                        help='String that specifies the lexicon type')
    
    parser.add_argument('--model_type', type=str, required=True,
                        help='String that specifies the model type')
    
    parser.add_argument('--embedding_type', type=str, required=True,
                        help='String that specifies the wordembeddings')
    
    parser.add_argument('--pos_threshold', type=float, required=True,
                        help='Threshold to classify document as positive, i.e. doc is positive if score > pos_threshold')
    
    parser.add_argument('--neg_threshold', type=float, required=True,
                        help='Threshold to classify document as negative, i.e. doc is negative if score < neg_threshold')
    
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to output directory', default="../output")
    
    args = parser.parse_args()
    
    # -- RUN MAIN FUNCTION ---

    main(lexicon=args.lexicon,
         model_type=args.model_type,
         embedding_type=args.embedding_type, 
         pos_threshold=args.pos_threshold,
         neg_threshold=args.neg_threshold,
         output_path=args.output_path)