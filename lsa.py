import pandas as pd
import numpy as np
import string
import nltk
import re
import math
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

def load_data(file_path):
    return pd.read_csv(file_path, header=0)

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
        text = row.text
        
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


def get_concept_by_document(df_tf_idf, customSVD = False):
    '''Transform data to concept space.
    '''
    values = df_tf_idf.fillna(0).to_numpy()
    
    if customSVD:
        U, s_eigen, V = custom_svd(values, False)
    else:
        U, s_eigen, V = np.linalg.svd(values, full_matrices=False)
    
    S = np.diag(s_eigen)
    
    concept_by_document = S @ V.T
    return pd.DataFrame(concept_by_document)


def cosine_similarity(x, y):
    '''Returns cosine similarity of two vectors.'''
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_n_nearest(df_concept, i, n=None, min_sim=0):
    '''Returns most similar (column) vectors to `i`-th vector in `arr`.
    
    Parameters
    ----------
    df_concept : pd.DataFrame
    i : index of vector to be compared to
    n : return at most `n` vectors
    '''
    
    src_vector = df_concept[i].copy()
    df = df_concept.apply(func=cosine_similarity, axis=0, args=(src_vector, ))
    
    if n:
        # skip first value - the src_vector itself
        return df.sort_values(ascending=False)[1:n + 1]
    else:
        return df.sort_values(ascending=False)

class LSA:
    '''Wrapper for LSA methods and data.'''
    def __init__(self):
        pass

    def load(self, data_file='articles.csv', use_cache=True, cache_file='concept.csv'):
        print('Loading data...', flush=True)
        self.df_data = load_data('articles.csv')
        
        if use_cache and os.path.isfile(cache_file):
            print('Loading concept matrix from "{}"...'.format(cache_file))
            self.df_concept = pd.read_csv(cache_file, index_col=0)
            self.df_concept.columns = self.df_concept.columns.astype(int)
        else:
            print('Preprocessing...', flush=True)
            preproccessed_docs = preprocess_docs(self.df_data)
            
            df_frequency = get_term_by_document_frequency(preproccessed_docs)

            df_reduced = reduce_terms(df_frequency, 0.8, 0.1)

            df_tf_idf = get_tf_idf(df_reduced)

            print('Creating concept matrix...', flush=True)
            self.df_concept = get_concept_by_document(df_tf_idf)

            if cache_file:
                self.df_concept.to_csv(cache_file, header=True, index=True)
                print('Concept matrix saved to {}'.format(cache_file))
        
    def get_n_nearest(self, doc_index, n):
        best_match = get_n_nearest(self.df_concept, doc_index, n)

        df = self.df_data.iloc[best_match.index].copy()
        df['similarity'] = best_match
        return df


if __name__ == "__main__":
    lsa = LSA()
    lsa.load()

    df = lsa.get_n_nearest(doc_index=2, n=5)

    print(df)