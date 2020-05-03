import os.path
import pandas as pd


def select_plots(output_file, min_year=1990, max_rows=1000):
    folder = 'raw_data'
    file = 'wiki_movie_plots_deduped.csv'
    csv_path = os.path.join(folder, file)

    df = pd.read_csv(csv_path)

    df = df[df['Release Year'] > min_year]
    df = df[df['Director'] != 'Unknown']

    countries = ['American', 'British', 'Japanese', 'Canadian']
    genres = ['drama', 'comedy', 'romance', 'action', 'romantic comedy', 'comedy-drama', 'crime drama', 'thriller',
              'science fiction']

    df = df[df['Genre'].isin(genres)]
    df = df[df['Origin/Ethnicity'].isin(countries)]

    # count director -> film count
    film_count = df.groupby(['Director']).count().Title.to_dict()
    # map director -> film count
    df['movies'] = df['Director'].map(film_count)

    df = df.sort_values('movies', ascending=False)[:max_rows]

    # for compatibility with current code
    df = df.rename(columns={
        'Title': 'title',
        'Director': 'author',
        'Wiki Page': 'link',
        'Plot': 'content'
    })

    df.to_csv(output_file, header=True, index=False)
    print('plots saved to', output_file)
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Select movie plots.')
    parser.add_argument('max_rows', type=int, help='maximum number of items to be selected')
    args = parser.parse_args()
    select_plots(max_rows=args.max_rows, output_file='data/plots.csv')