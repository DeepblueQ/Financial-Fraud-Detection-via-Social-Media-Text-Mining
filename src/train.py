import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def train_eval_xgb(X: np.ndarray, y: np.ndarray, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / max(1, pos)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    print(classification_report(y_te, pred, digits=4))
    print("ROC-AUC:", float(roc_auc_score(y_te, prob)))
    print("PR-AUC:", float(average_precision_score(y_te, prob)))
    return clf
