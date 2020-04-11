import pandas as pd
import numpy as np
import string
import nltk
import re
import math
import os
import json
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

def parse_recipes(limit=100, output='recipes.csv'):
    articles = []
    with open('recipes.json') as file:
        for i, line in enumerate(file):
            if i == limit:
                break
                
            obj = json.loads(line)

            text = '{}\n'.format(obj['Description'])
            for ing in obj['Ingredients']:
                text += ing + '\n'
            for met in obj['Method']:
                text += met + '\n'
            articles.append({
                'title': obj['Name'],
                'author': obj['Author'],
                'link': obj['url'],
                'text': text,
            })

    df = pd.DataFrame(articles)
    if output:
        df.to_csv(output)
    return df

def load_data(files):
    return pd.concat((pd.read_csv(f, header=0) for f in files), ignore_index=True)

def get_lemmatization_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dictionary = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dictionary.get(tag, wordnet.NOUN)


def preprocess_docs(docs, use_lemmatizer=True, remove_numbers=True):
    '''Tokenize and preprocess documents.

    Does lemmatization/stemming. Removes stopwords. 
    
    Parameters
    ----------
    docs : pd.DataFrame
    use_lemmatizer : bool
                     Uses lemmazizer if True, othrerwise uses stemmer.
    remove_numbers : bool
    '''
    preproccessed_docs = []
    
    # English stop words list
    en_stop = set(stopwords.words('english'))
    
    # Word tokenizer that removes punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    
    # lemmatizer / Stemmer
    if use_lemmatizer:
        lemmatizer = WordNetLemmatizer()
    else:
        stemmer = SnowballStemmer("english")
    
    # tqdm displays progress bar
    for row in tqdm(docs.itertuples(index=True, name='Doc'), total=len(docs)):
        text = row.content
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        text_words = tokenizer.tokenize(text)
        
        if use_lemmatizer:
            text_words = [lemmatizer.lemmatize(word.lower(), get_lemmatization_pos(word.lower())) for word in text_words
                          if word not in string.punctuation and word.lower() not in en_stop]
        else:
            text_words = [stemmer.stem(word.lower()) for word in text_words
                         if word not in string.punctuation and word.lower() not in en_stop]
        
        preproccessed_docs.append({'words': text_words})
    
    print() # fix missing newline from tqdm
    
    
    pdocs = pd.DataFrame(preproccessed_docs)
    return pdocs


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for index, row in preprocessed_docs.iterrows():
        doc_id = index
        doc_words = row['words']
        
        document_by_term[doc_id] = {}
        
        for word in set(row['words']):
            document_by_term[doc_id][word] = doc_words.count(word)

    df = pd.DataFrame(document_by_term)
    
    return df


def reduce_terms(df_frequency, max_df=1, min_df=0, max_terms=None):
    '''Remove unimportant terms from term-by-document matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
    max_df : float , between [0, 1]
             Terms that appear in more % of documents will be ignored
    min_df : float , between [0, 1]
             Terms that appear in less % of documents will be ignored
    max_terms : int , None
                If not None, only top `max_terms` terms will be returned.
    '''
    df = df_frequency.copy()
    corpus_size = df.shape[1]

    if 'doc_frequency' not in df:
        df['doc_frequency'] = df_frequency.fillna(0).astype(bool).sum(axis=1) / corpus_size
            
    df = df[df.doc_frequency <= max_df]
    df = df[df.doc_frequency >= min_df]
    
    if max_terms is not None and max_terms < df.shape[0]:
        df['term_count'] = df_frequency.fillna(0).sum(axis=1)
        df = df.sort_values('term_count', ascending=False)
        df = df.head(max_terms)
        df.drop('term_count',axis=1, inplace=True)
    
    return df


def get_tf_idf(df_frequency):
    df = df_frequency.copy()
    # tf := word frequency / total frequency
    df.loc['total_words'] = df.sum()
        
    df = df.drop('total_words')[:] / df.loc['total_words']
    
    # idf := log ( len(all_documents) / len(documents_containing_word) )
    
    corpus_size = df.shape[1]

    # number of non-zero cols
    if 'doc_frequency' not in df_frequency:
        df['doc_frequency'] = df.fillna(0).astype(bool).sum(axis=1)
        
    df['idf'] = np.log( corpus_size / df['doc_frequency'] )
    
    # tf-idf := tf * idf
    _cols = df.columns.difference(['idf', 'doc_frequency'])
    df[_cols] = df[_cols].multiply(df["idf"], axis="index")
    
    df.drop(columns=['doc_frequency', 'idf'], inplace=True)
    
    return df



def custom_svd(A, full_matrices=True):
    eig_vals, eig_vecs = np.linalg.eig(A @ A.transpose())
    U = eig_vecs.real
    
    eig_vals, eig_vecs = np.linalg.eig(A.transpose() @ A)
    V = eig_vecs.transpose().real
    
    s_eigen = [math.sqrt(abs(x.real)) for x in eig_vals]
    
    if full_matrices == False:
        k = min(A.shape[0], A.shape[1])
        U = U[:, :k]
        V = V[:k, :]
    
    return U, np.array(s_eigen), V


def transform_to_concept_space(df_tf_idf, k=20, customSVD = False):
    '''Transform data to concept space.
    k : number of concepts
    '''
    if k > df_tf_idf.shape[1]:
        k = df_tf_idf.shape[1]
    
    values = df_tf_idf.fillna(0).to_numpy()
    
    if customSVD:
        U, s_eigen, V = custom_svd(values, False)
    else:
        U, s_eigen, V = np.linalg.svd(values, full_matrices=False)
    
    # Get only first k concepts
    S = np.diag(s_eigen[:k])
    
    concept_by_document = S @ (V[:,:k]).T
    query_projection = (U[:,:k] ).T
    return pd.DataFrame(concept_by_document), pd.DataFrame(query_projection)


def cosine_similarity(x, y):
    '''Returns cosine similarity of two vectors.'''
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_n_nearest(df_tf_idf, df_concept_by_doc, df_query_projection, i, n=None, min_sim=0):
    '''Returns most similar (column) vectors to `i`-th vector in `arr`.
    
    Parameters
    ----------
    df_concept_by_doc : pd.DataFrame
    df_query_projection : pd.DataFrame
    i : index of vector to be compared to
    n : return at most `n` vectors
    '''
    
    # If n isn't set set it to number of docs - 1
    if n == None:
        n = df_concept_by_doc.shape[1] - 1
    
    src_vector = df_query_projection.fillna(0).to_numpy() @ (df_tf_idf.fillna(0).to_numpy())[:,i]
    
    df = df_concept_by_doc.apply(func=cosine_similarity, axis=0, args=(src_vector, ))
    
    return df.sort_values(ascending=False)[1:n + 1]

class LSA:
    '''Wrapper for LSA methods and data.'''
    def __init__(self, data_files=['all_the_news_1000_articles.csv']):
        self.df_data = load_data(data_files)

    def preprocess(self, file='tf_idf.csv', read_cache=True, max_df=0.75, min_df=0.05, max_terms=1200):
        if read_cache:
            if not os.path.isfile(file):
                raise ValueError("Can't read file {}".format(file))
            self.df_tf_idf = pd.read_csv(file)
        else:
            df_words = preprocess_docs(self.df_data)
            df_frequency = get_term_by_document_frequency(df_words)
            df_reduced = reduce_terms(df_frequency, max_df=max_df, min_df=min_df, max_terms=max_terms)
            self.df_tf_idf = get_tf_idf(df_reduced)
            if file:
                self.df_tf_idf.to_csv(file)

    def compute(self, file1='concept_by_doc.csv', file2='query_projection.csv', read_cache=True):
        if read_cache:
            if not os.path.isfile(file1):
                raise ValueError("Can't read file {}".format(file1))
            if not os.path.isfile(file2):
                raise ValueError("Can't read file {}".format(file2))
            
            self.df_concept_by_doc = pd.read_csv(file1, index_col=0)
            # self.df_concept = pd.read_csv(cache_file, index_col=0, usecols=int) .. better?
            self.df_concept_by_doc.columns = self.df_concept_by_doc.columns.astype(int)
            
            self.df_query_projection = pd.read_csv(file2, index_col=0)
            self.df_query_projection.columns = self.df_query_projection.columns.astype(int)
            
            # Number of documents not ok
            if len(self.df_data) != self.df_concept_by_doc.shape[1]:
                raise ValueError('Unexpected query concept by document matrix size! Possibly caused by outdated cache.')
                
            # Number of terms not ok
            if self.df_tf_idf.shape[0] != self.df_query_projection.shape[1]:
                raise ValueError('Unexpected query projection matrix size! Possibly caused by outdated cache.')
        else:
            self.df_concept_by_doc, self.df_query_projection = transform_to_concept_space(self.df_tf_idf)
            if file1:
                self.df_concept_by_doc.to_csv(file1, header=True, index=True)
            if file2:
                self.df_query_projection.to_csv(file2, header=True, index=True)

    def load(self, read_cache=True):
        self.preprocess(read_cache=read_cache)
        self.compute(read_cache=read_cache)       
        
    def get_n_nearest(self, doc_index, n):
        best_match = get_n_nearest(self.df_tf_idf, self.df_concept_by_doc, self.df_query_projection, doc_index, n)

        df = self.df_data.iloc[best_match.index].copy()
        df['similarity'] = best_match
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LSA demo.')
    parser.add_argument('--no-cache', dest='cache', action='store_false', help='disable cache')
    parser.add_argument('--cache', dest='cache', action='store_true', help='enable cache')

    args = parser.parse_args()
    
    lsa = LSA()
    lsa.load(read_cache=args.cache)

    print('Loaded {} documents'.format(len(lsa.df_data)))
    print('Concept matrix shape: {}'.format(lsa.df_concept.shape))

    i = 2
    n = 5
    print('Example for i={} and n={}'.format(i, n))
    df = lsa.get_n_nearest(doc_index=i, n=n)

    print(df)

