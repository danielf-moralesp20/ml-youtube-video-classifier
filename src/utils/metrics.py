import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def metrics_summary(y_test, y_hat, y_hat_proba, output_path):
    metrics = {
        'f1': f1_score(y_test, y_hat, average='macro'),
        'recall': recall_score(y_test, y_hat, average='macro'),
        'precision': precision_score(y_test, y_hat, average='macro'),
        'auc': roc_auc_score(y_test, y_hat_proba, multi_class='ovr', average='macro')
    }

    json.dump(
        obj=metrics,
        fp=open(output_path, 'w')
    )
