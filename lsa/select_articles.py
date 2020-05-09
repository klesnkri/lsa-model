import os.path
import pandas as pd
import csv
import json

def select_articles(count, output_file, seed=42):
    folder = 'raw_data'
    files = ['articles1.csv', 'articles2.csv', 'articles3.csv']
    files = [os.path.join(folder, f) for f in files]

    df = pd.concat((pd.read_csv(f, header=0) for f in files), ignore_index=True)
    df = df.sample(n=count, replace=False, random_state=seed)
    df = df[['title', 'author', 'url', 'content', 'id', 'publication']]
    
    df.to_csv(output_file, header=True, index=False)

    # add articles containing homonyms and synonyms
    csvfile = open(output_file, 'a')
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    file = open('homonyms_synonyms/articles.json')

    data = json.load(file) 

    for article in data['articles']: 
        writer.writerow([article['title'], article['author'], article['url'], article['content'], article['id'], article['publication']])

    # Closing file 
    csvfile.close() 
    
    print('articles saved to', output_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Select articles.')
    parser.add_argument('count', type=int, help='number of articles to be selected')
    args = parser.parse_args()
    select_articles(args.count, 'data/articles.csv')