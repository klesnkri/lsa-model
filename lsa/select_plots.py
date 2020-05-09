import os.path
import pandas as pd


possible_genres = {'unknown', 'drama', 'comedy', 'horror', 'action', 'thriller', 'romance',
                   'western', 'crime', 'adventure', 'musical', 'crime drama',
                   'romantic comedy', 'science fiction', 'film noir'}


def select_plots(output_file, min_year=1990, max_rows=1000, seed=42, randomize=True,
                 countries=('American', 'British'),
                 genres=('comedy', 'romance', 'action', 'animation',
                         'crime drama', 'fantasy', 'science fiction')
                 ):
    folder = 'raw_data'
    file = 'wiki_movie_plots_deduped.csv'
    csv_path = os.path.join(folder, file)

    df = pd.read_csv(csv_path)

    df = df[df['Release Year'] > min_year]
    df = df[df['Director'] != 'Unknown']

    df = df[df['Genre'].isin(genres)]
    df = df[df['Origin/Ethnicity'].isin(countries)]

    # count director -> film count
    film_count = df.groupby(['Director']).count().Title.to_dict()
    # map director -> film count
    df['movies'] = df['Director'].map(film_count)

    print('> total plots after filtering:', len(df))
    if randomize:
        df = df.sample(frac=1, random_state=seed)[:max_rows]
    else:
        df = df.sort_values('movies', ascending=False)[:max_rows]

    print('> saving', len(df), 'plots')

    # for compatibility with current code
    df = df.rename(columns={
        'Title': 'title',
        'Director': 'author',
        'Wiki Page': 'url',
        'Plot': 'content'
    })

    df.to_csv(output_file, header=True, index=False)
    print('> plots saved to', output_file)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Select movie plots.')
    parser.add_argument('max_rows', type=int, help='maximum number of items to be selected')
    args = parser.parse_args()
    select_plots(max_rows=args.max_rows, output_file='data/plots.csv')