import sys, os
import argparse
import pandas as pd

import spacy
import asent
from asent.component import Asent
from asent.lang.da import LEXICON, NEGATIONS, INTENSIFIERS, CONTRASTIVE_CONJ
from danlp.datasets import EuroparlSentiment1, LccSentiment, TwitterSent

sys.path.append(os.path.join(".."))
from utils.twittertokens import setup_twitter
from utils.classification_utils import (sentiment_score_to_label, 
                                        sentiment_score_to_label_asent, 
                                        f1_class, f1_report)


def main(lexicon, model_type, embedding_type, output_path):
    
    # Prepare output directory
    output_path = os.path.join(output_path, "lexicon_evaluation", model_type, embedding_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create txt file
    with open(f'{output_path}/f1_report.txt', 'a') as f:
        f.write(f"LEXICON EVAL RESULTS\n")
        
    LEXICON = pd.read_csv("../lexicons/sentida2_lexicon.csv")
    
    # Prepare lexicon
    if lexicon == "asent_base":
        LEXICON = LEXICON

    elif lexicon == "asent_expanded":
        print("will expand!")
        EXPANSION = pd.read_csv(f"../output/sentiment_prediction/{model_type}/{embedding_type}_4/sentiment_predictions.csv")
        EXPANSION = EXPANSION.rename(columns={"word":"word", "restricted_sentiments":"score"})
        EXPANSION = EXPANSION.drop(columns =["predicted_sentiment"])
        LEXICON = pd.concat([LEXICON, EXPANSION])
        
    print("loaded concat")
        
    LEXICON = dict(zip(LEXICON.word, LEXICON.score))
    
    print("zipped")
        
    nlp = spacy.load("da_core_news_lg")

    # Add Asent to NLP pipeline
    Asent(nlp, name = "classification", 
          lexicon=LEXICON, 
          negations=NEGATIONS, 
          intensifiers=INTENSIFIERS, 
          contrastive_conjugations=CONTRASTIVE_CONJ,
          lowercase=True,
          lemmatize=True, 
          force=True)
    
    def asent_score(sent):
        doc = nlp(sent)
        return doc._.polarity.compound

    # load dataset
    for dataset in ['euparlsent','lccsent','twitter']:
        if dataset in ["euparlsent", "lccsent"]:
            if dataset == "euparlsent":
                data = EuroparlSentiment1()
                df = data.load_with_pandas()
            elif dataset == "lccsent":
                data = LccSentiment()
                df = data.load_with_pandas()
            
            print(f"[INFO] Data of {dataset} loaded...")
            df['pred'] = df.text.map(asent_score)
            print("MIN: ", min(df['pred']), "MAX: ", max(df["pred"]))
            df['pred'] = df.pred.map(sentiment_score_to_label_asent)
            df['valence'] = df['valence'].map(sentiment_score_to_label)
            f1_report(df['valence'], df['pred'], 'Asent', dataset)

        elif dataset == "twitter":
            setup_twitter()
            twitSent = TwitterSent()
            df_val, df_train = twitSent.load_with_pandas()
            df = pd.concat([df_val, df_train])
            
            print(f"[INFO] Data of {dataset} loaded...")
            df['asent'] = df.text.map(asent_score).map(sentiment_score_to_label_asent)
            f1_report(df['polarity'], df['asent'], 'Asent', "twitter_sentiment(val)")
            

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lexicon', type=str, required=True,
                        help='String that specifies the lexicon type')
    
    parser.add_argument('--model_type', type=str, required=True,
                        help='String that specifies the model type')
    
    parser.add_argument('--embedding_type', type=str, required=True,
                        help='String that specifies the wordembeddings')
    
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to output directory', default="../output")
    
    args = parser.parse_args()

    main(lexicon=args.lexicon,
         model_type=args.model_type,
         embedding_type=args.embedding_type, 
         output_path=args.output_path)