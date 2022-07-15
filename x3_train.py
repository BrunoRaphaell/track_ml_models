import pandas as pd
import numpy as np
import json

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils.auxiliary_functions import *

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("data_processed.csv")

y = df.pop("cons_general").to_numpy()
y[y< 4] = 0
y[y>= 4] = 1

X = df.to_numpy()
X = preprocessing.StandardScaler().fit_transform(X)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTreeClassifier(max_depth=20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

labels = ["True Neg","False Pos","False Neg","True Pos"]
categories = ["0", "1"]
plot_matriz_confusao(y_test,
                      y_pred,
                      group_names=labels,
                      categories=categories,
                      figsize=(8, 6),
                      cbar=False,
                      save_fig = True,
                      title="Matriz de confusão para o classificador Random Forest")

acc, precision, recall, f1_score_metric = return_metrics(y_test, y_pred)

with open("metrics.json", 'w') as outfile:
        json.dump({ "Accuracy": acc, "Precision": precision, "Recall" : recall, "F1-score": f1_score_metric}, outfile)

# Adicionar posteriormente
# pd.DataFrame({'actual': y_test, 'predicted': y_pred}).to_csv('classes.csv', index=False)
