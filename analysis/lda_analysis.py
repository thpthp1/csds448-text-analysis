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

from texthero import preprocessing

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_stopwords,
                   preprocessing.remove_digits,
                   preprocessing.remove_whitespace]
import sys

GAME = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2] == 'all':
        RATING = 'all'
    else: RATING = max(int(sys.argv[2]), 1) 
else: 
    RATING = 1

df = pd.read_csv(f'./clean/clean-{GAME}.csv')
df['clean_text'] = hero.clean(df['content'], custom_pipeline)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
text_df = df[df.score == RATING] if RATING != 'all' else df
text_count = vectorizer.fit_transform(text_df['clean_text'])
lda = LatentDirichletAllocation(learning_method='online', learning_offset=50.,random_state=0).fit(text_count)

display_topics(lda, vectorizer.get_feature_names(), 10)