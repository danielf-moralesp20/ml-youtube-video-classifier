import click
import joblib
import logging
import pandas as pd
import src.utils.files as files
import src.utils.paths as path
from sklearn.metrics import classification_report
from src.utils.metrics import metrics_summary


@click.command()
@click.option('-c', '--config', help="Configuration file path", required=True, type=click.Path(exists=True))
def main(config):
    logger = logging.getLogger(__name__)
    config_content = files.read_yaml(config)

    logger.info("Loading test dataset")
    df = pd.read_csv(path.data_dir("processed/test.csv"))

    X_test = df.drop(columns=['category'])
    y_test = df.category

    logger.info("Loading estimator")
    estimator = joblib.load(path.models_dir("model.joblib"))

    logger.info("Predicting")
    y_hat = estimator.predict(X_test)
    y_hat_proba = estimator.predict_proba(X_test)

    logger.info("Generating metrics")
    metrics_summary(
        y_test, y_hat, y_hat_proba,
        path.reports_dir('summary.json')
    )

    logger.info("Generating confusion matrix data")
    df_confusion_matrix = pd.concat([pd.Series(y_hat), y_test], axis=1)
    df_confusion_matrix.columns = ['prediction', 'real']
    df_confusion_matrix.to_csv(
        path.reports_dir("confusion_matrix_data.csv"), index=False
    )

    logger.info("Generating classification report data")
    clf_report = classification_report(y_test, y_hat, output_dict=True)
    df_clf_report = pd.DataFrame(clf_report).T
    df_clf_report.drop(
        index=['accuracy', 'macro avg', 'weighted avg'], inplace=True
    )
    df_clf_report.index.name = 'class'
    df_clf_report.to_csv('reports/classification_report_data.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
