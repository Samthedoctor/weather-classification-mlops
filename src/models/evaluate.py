from sklearn.metrics import roc_auc_score

def compute_roc_auc(model, X, y):
    """Compute ROC-AUC score for a model."""
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.decision_function(X)
    return roc_auc_score(y, y_pred_proba)