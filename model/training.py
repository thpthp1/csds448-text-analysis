from model.utils import vectorize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from utils import vectorize
import numpy as np
import pickle

REVIEW_COL = 'review'
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
    x_train, y_train = X[train, :], Y[train]
    x_test, y_test = X[test, :], Y[test]
    model = SVC(class_weight='balanced')
    model.fit(x_train, y_train)
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
    print('-'*40)
    fold += 1
with open('./model/vectorizer-reviews.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    
with open('./model/svc-reviews.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

for key in metrics.keys():
    if key != 'conf_mat':
        print(f'{key}: {np.mean(metrics[key])} +- {np.std(metrics[key])}')

    

