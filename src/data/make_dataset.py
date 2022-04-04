import click
import dvc.api
import logging
import pandas as pd
import src.utils.files as files
from io import BytesIO


@click.command()
@click.option('-c', '--config', help="Configuration file path", required=True, type=click.Path(exists=True))
def main(config):
    """ Runs data processing scripts to turn multiple raw datasets into just one raw full dataset. """
    logger = logging.getLogger(__name__)
    config_content = files.read_yaml(config)
    df_full = pd.DataFrame()

    for dataset in config_content['data_load']['datasets']:
        dataset_dir = dataset['dir']
        target_name = dataset.get('target', 'category')
        feature_title_name = dataset.get(
            'feature_names', {}).get('title', 'title')
        feature_desc_name = dataset.get('feature_names', {}).get(
            'description', 'description')

        logger.info(f'Retrieving  {dataset_dir}')

        data = dvc.api.read(dataset_dir, mode="rb", remote="storage")
        df = pd.read_csv(BytesIO(data))
        df = df[[feature_title_name, feature_desc_name, target_name]]
        df.rename(columns={feature_title_name: 'title',
                  feature_desc_name: 'description', target_name: 'category'}, inplace=True)

        logger.info(f'Concating {dataset_dir}')
        df_full = pd.concat([df_full, df])

    logger.info('Saving full dataset')
    df_full.to_csv('data/interim/youtube.csv', index=False)
    logger.info('Make Dataset process finished successfully')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
