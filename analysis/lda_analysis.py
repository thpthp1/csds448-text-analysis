import nltk
from numpy.lib.function_base import vectorize
from texthero.visualization import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import texthero as hero
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import logging
from texthero import preprocessing
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_stopwords,
                   preprocessing.remove_digits,
                   preprocessing.remove_whitespace]
import sys


def main(arg1, arg2):
    GAME = arg1

    if arg2 == 'all':
        RATING = 'all'
    else:
        RATING = max(int(arg2), 1)

    df = pd.read_csv(f'./clean/clean-{GAME}.csv')
    df['clean_text'] = hero.clean(df['content'], custom_pipeline)

    def display_topics(model, feature_names, no_top_words):
        f = open("%s_%s.txt"%(GAME,RATING), "w")
        print("File created! Analyzing...")
        for topic_idx, topic in enumerate(model.components_):
            try:
                print(f"Topic {topic_idx}:", file=f)
                print(" ".join([feature_names[i]
                                for i in topic.argsort()[:-no_top_words - 1:-1]]), file=f)
            except Exception:
                logging.error("%s %s Topic: %s Did not work" %(GAME,RATING,topic_idx))
        f.close()

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_df = df[df.score == RATING] if RATING != 'all' else df
    text_count = vectorizer.fit_transform(text_df['clean_text'])
    lda = LatentDirichletAllocation(learning_method='online', learning_offset=50.,random_state=0).fit(text_count)
    print("Creating File %s_%s.txt...."%(GAME,RATING))
    display_topics(lda, vectorizer.get_feature_names(), 10)

if __name__ == '__main__':
    #try:
    sys.exit(main(sys.argv[1], sys.argv[2]))
    #except Exception:
     #   logging.error(sys.argv[1] + sys.argv[2] + " did not work")