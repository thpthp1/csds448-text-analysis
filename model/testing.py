from model.utils import vectorize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from utils import vectorize, TEXTHERO_FILTER
import numpy as np
import pickle
from texthero import preprocessing
import texthero
from text_matching_clf import TextMatchingClassifier

reviews = pd.read_csv("./clean/clean-geomdash.csv")

reviews["clean"] = texthero.clean(reviews["content"], pipeline=TEXTHERO_FILTER)

# SAMPLE_PER_CLASS = 250
# sampled_reviews = reviews.groupby('score').apply(lambda x: x.sample(SAMPLE_PER_CLASS))
sampled_reviews = reviews

with open('./model/vectorizer-reviews.pkl', 'rb') as vf:
    vectorizer = pickle.load(vf)

with open('./model/svc-reviews.pkl', 'rb') as svc_f:
    model = pickle.load(svc_f)

with open('./model/text-match-reviews.pkl', 'rb') as text_f:
    text_match_model = pickle.load(text_f)


X = vectorizer.transform(sampled_reviews["clean"])
output = pd.DataFrame({'content': sampled_reviews["content"],
                        "score" : sampled_reviews["score"],
                        "is_aggresive_IAP" : model.predict(X),
                        "is_agressive_IAP_text_match": text_match_model.predict(sampled_reviews["clean"])})


output.to_csv("./model/sample_pred_geomdash.csv")
