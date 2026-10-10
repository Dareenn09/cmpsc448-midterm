"""Reference baseline: TF-IDF + Logistic Regression for input / output / both."""
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("splits/train.csv")
test = pd.read_csv("splits/test.csv")

texts = {
    "input": lambda d: d["LLM_Input"].astype(str),
    "output": lambda d: d["LLM_output"].astype(str),
    "both": lambda d: d["LLM_Input"].astype(str) + " [SEP] " + d["LLM_output"].astype(str),
}

os.makedirs("results", exist_ok=True)
results = {}
for mode, get in texts.items():
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, lowercase=False)
    X_train, X_test = vec.fit_transform(get(train)), vec.transform(get(test))
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(X_train, train["LLM_family"])
    pred = clf.predict(X_test)
    results[mode] = {"test_accuracy": accuracy_score(test["LLM_family"], pred),
                     "test_macro_f1": f1_score(test["LLM_family"], pred, average="macro")}
    print(f"TF-IDF + LR ({mode:6s})  acc {results[mode]['test_accuracy']:.4f}  "
          f"macro-F1 {results[mode]['test_macro_f1']:.4f}")

json.dump(results, open("results/baseline_tfidf_lr.json", "w"), indent=2)
