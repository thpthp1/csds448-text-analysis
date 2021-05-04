from operator import imatmul
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
from model.utils import vectorize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
from utils import vectorize, TEXTHERO_FILTER
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
from texthero import preprocessing
import texthero

REVIEW_COL = 'review'

class TextMatchingClassifier:

    def fit(self, text, Y, n_features=10):
        vectorizer = CountVectorizer(min_df=2, stop_words='english')
        X = vectorizer.fit_transform(text)
        self.vectorizer = vectorizer
        selector = SelectKBest(k=n_features, score_func=f_classif)
        selector.fit(X, Y)
        self.selector = selector
        return selector.get_support(indices=True)

    def predict(self, text, thresh=0):
        X = self.vectorizer.transform(text)
        X = X[:, self.selector.get_support()]
        #print((np.asarray(X.sum(axis=1)) > thresh).squeeze())
        return (np.asarray(X.sum(axis=1)) > thresh).squeeze()


if __name__ == "__main__":
    positive = pd.read_csv("./reviews/TaggedData - Sheet1.csv", index_col=None)
    dragalia_neg = pd.read_csv("./reviews/Dragalia Lost 2 non-p2w Reviews - clean-dragalia2.csv")
    dungeon_neg = pd.read_csv("./reviews/Non-p2w reviews dungeon keeper - clean-dungeonkeeper.csv")
    color_neg = pd.read_csv("./reviews/non p2w color stack reviews - clean-stackcolors.csv")

    nan_value = float("NaN")


    dragalia_neg.replace("", nan_value, inplace=True)
    dragalia_neg.dropna(subset=[REVIEW_COL], inplace=True)
    dungeon_neg.replace("", nan_value, inplace=True)
    dungeon_neg.dropna(subset=[REVIEW_COL], inplace=True)
    color_neg.replace("", nan_value, inplace=True)
    color_neg.dropna(subset=[REVIEW_COL], inplace=True)

    data = pd.DataFrame(
        { REVIEW_COL : positive[REVIEW_COL].append(dragalia_neg[REVIEW_COL], ignore_index=True)
                        .append(dungeon_neg[REVIEW_COL], ignore_index=True)
                        .append(color_neg[REVIEW_COL]),
            'tag': [True] * len(positive) + [False] * (len(dragalia_neg) + len(dungeon_neg) + len(color_neg))}
    )

    data["clean"] = texthero.clean(data[REVIEW_COL], TEXTHERO_FILTER)

    X, vectorizer = vectorize(data, REVIEW_COL, dict(stop_words='english', min_df=2))
    Y = data['tag'].to_numpy()

    kfold = StratifiedKFold(shuffle=True)

    fold = 0
    metrics = {
        'acc': [],
        'recall': [],
        'pre': [],
        'f1': [],
        'conf_mat': []
    }


    for train, test in kfold.split(X, Y):
        x_train, y_train = data["clean"].iloc[train], Y[train]
        x_test, y_test = data["clean"].iloc[test], Y[test]
        model = TextMatchingClassifier()
        features = model.fit(x_train, y_train)
        #print(features)
        #print(y_test)
        y_pred = model.predict(x_test) 
        metrics['acc'].append(accuracy_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['pre'].append(precision_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))
        metrics['conf_mat'].append(confusion_matrix(y_test, y_pred))
        print('Fold {}: acc: {}, pre: {}, recall: {}, f1: {}'
                        .format(fold+1, metrics['acc'][fold], metrics['pre'][fold], 
                                    metrics['recall'][fold], metrics['f1'][fold]))
        print('Confusion matrix')
        print(metrics['conf_mat'][fold])
        print(f'top features: {np.asarray(model.vectorizer.get_feature_names())[features]}')
        print('-'*40)
        fold += 1

    with open('./model/text-match-reviews.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    for key in metrics.keys():
        if key != 'conf_mat':
            print(f'{key}: {np.mean(metrics[key])} +- {np.std(metrics[key])}')