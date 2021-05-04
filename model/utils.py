from sklearn.feature_extraction.text import TfidfVectorizer
import texthero as hero
from texthero import preprocessing
import texthero
import pandas as pd
import numpy as np


TEXTHERO_FILTER = [preprocessing.fillna,
                   preprocessing.remove_urls,
                   preprocessing.remove_html_tags,
                   preprocessing.lowercase,
                   preprocessing.remove_stopwords,
                   preprocessing.remove_whitespace]


def vectorize(df: pd.DataFrame, text_col: str, kwargs: dict):
    data = texthero.clean(df[text_col], TEXTHERO_FILTER)
    vectorizer = TfidfVectorizer(**kwargs)
    data_mat = vectorizer.fit_transform(data)
    return data_mat, vectorizer