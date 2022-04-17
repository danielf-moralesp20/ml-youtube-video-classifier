import click
import logging
import pandas as pd
import src.utils.files as files
import src.utils.paths as path
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


@click.command()
@click.option('-c', '--config', help="Configuration file path", required=True, type=click.Path(exists=True))
def main(config):
    logger = logging.getLogger(__name__)
    config_content = files.read_yaml(config)

    logger.info("Loading processed dataset")
    df = pd.read_csv(path.data_dir('processed/youtube.csv'))

    logger.info("Random undersampling")
    ru = RandomUnderSampler(
        random_state=config_content['base']['random_state'])
    df_balanced, y = ru.fit_resample(
        df.drop(columns=['category']), df.category)

    logger.info("TF-IDF vectorizing")
    tfidf_vectorizer = TfidfVectorizer(
        min_df=config_content['build_features']['min_df'])
    df_tfidf = pd.DataFrame(tfidf_vectorizer.fit_transform(
        df_balanced.document).toarray())

    logger.info("Splitting dataset between train and test")
    X_train, X_test, y_train, y_test = train_test_split(df_tfidf, y, stratify=y,
                                                        test_size=config_content['build_features']['test_size'],
                                                        random_state=config_content['base']['random_state'])

    logger.info('Saving train dataset')
    pd.concat([X_train, y_train], axis=1).to_csv(
        path.data_dir('processed/train.csv'), index=False)

    logger.info('Saving test dataset')
    pd.concat([X_test, y_test], axis=1).to_csv(
        path.data_dir('processed/test.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
