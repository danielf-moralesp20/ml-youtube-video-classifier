import click
import logging
import pandas as pd
import src.utils.files as files
import src.utils.text as text
import src.utils.paths as path


@click.command()
@click.option('-c', '--config', help="Configuration file path", required=True, type=click.Path(exists=True))
def main(config):
    """ Runs feature processing scripts to turn raw data into cleaned data."""
    logger = logging.getLogger(__name__)
    config_content = files.read_yaml(config)

    logger.info("Loading interim dataset")
    df = pd.read_csv(path.data_dir('interim/youtube.csv'))

    logger.info("Cleaning data")
    df.category = df.category.str.lower()
    df.category.replace({'travel blog': 'travel_blog', 'art&music': 'art_music', 'science&technology': 'science_technology',
                        'science': 'science_technology', 'music': 'art_music', 'travel': 'travel_blog'}, inplace=True)
    df.drop_duplicates(subset=['title', 'description'], inplace=True)
    df.dropna(inplace=True)

    df_final = df.copy()
    df_final.description = df.title + ' ' + df.description
    df_final.drop(columns=['title'], inplace=True)

    logger.info('Removing non-english rows')
    df_is_english = df_final.description.apply(text.is_english, args=[
        config_content['clean_dataset']['english_threshold_confidence']
    ])
    df_final.drop(df_final[~df_is_english].index, inplace=True)

    logger.info('Formatting text')
    df_final['clean_description'] = df_final.description.apply(
        text.clean_text, args=[config_content['clean_dataset']['min_token_length']])
    df_final.drop(df_final[df_final.clean_description.str.len(
    ) < config_content['clean_dataset']['min_tokens_count']].index, inplace=True)

    logger.info('Saving full clean dataset')
    df_final.to_csv(path.data_dir('processed/youtube.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
