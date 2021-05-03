import nltk
from numpy.lib.function_base import vectorize
from texthero.visualization import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import texthero as hero
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from texthero import preprocessing

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_stopwords,
                   preprocessing.remove_digits,
                   preprocessing.remove_whitespace]
import sys

GAME = sys.argv[1]
RATING = max(int(sys.argv[2]), 1) if len(sys.argv) > 2 else 1
df = pd.read_csv(f'./clean/clean-{GAME}.csv')
df['clean_text'] = hero.clean(df['content'], custom_pipeline)

vectorizer = CountVectorizer()
text_count = vectorizer.fit_transform(df[df.score == RATING]['clean_text'])

#print(np.asarray(text_count.sum(axis=0)))

freqs = dict(zip(vectorizer.get_feature_names(), np.asarray(text_count.sum(axis=0))[0]))

#print(freqs)

wc = WordCloud(width = 800, height = 800, stopwords=STOPWORDS).generate_from_frequencies(freqs)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig(f'./images/{GAME}-{RATING}wc.png')

print(f"Processing done for {GAME} with rating {RATING}")