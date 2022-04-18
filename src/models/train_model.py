import click
import joblib
import logging
import pandas as pd
import src.utils.files as files
import src.utils.paths as path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


@click.command()
@click.option('-c', '--config', help="Configuration file path", required=True, type=click.Path(exists=True))
def main(config):
    logger = logging.getLogger(__name__)
    config_content = files.read_yaml(config)

    logger.info("Loading train dataset")
    df = pd.read_csv(path.data_dir('processed/train.csv'))

    X_train = df.drop(columns=['category'])
    y_train = df.category

    logger.info("Searching best estimator")
    rand_search = GridSearchCV(
        GradientBoostingClassifier(
            random_state=config_content['base']['random_state']
        ),
        config_content['train_model']['cv_distributions'],
        cv=config_content['train_model']['cv'],
        scoring="f1_macro"
    )
    rand_search.fit(X_train, y_train)

    logger.info(f'Best score: {rand_search.best_score_}')

    logger.info("Saving estimator")
    estimator = rand_search.best_estimator_
    joblib.dump(estimator, path.models_dir("model.joblib"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
