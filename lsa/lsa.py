import pandas as pd
import numpy as np
import string
import nltk
import re
import math
import os
import os.path
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

TF_IDF_FILE = 'tf_idf.csv'
CONCEPT_FILE = 'concept_by_doc.csv'
PROJECTION_FILE = 'query_projection.csv'
DATA_FILES = ('articles.csv',)


def load_data(files):
    return pd.concat((pd.read_csv(f, header=0) for f in files), ignore_index=True)


def get_lemmatization_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dictionary = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dictionary.get(tag, wordnet.NOUN)


def preprocess_docs(docs, use_lemmatizer=True, remove_numbers=True):
    """Tokenize and preprocess documents.

    Does lemmatization/stemming. Removes stopwords.

    Parameters
    ----------
    docs : pd.DataFrame
    use_lemmatizer : bool
                     Uses lemmatizer if True, otherwise uses stemmer.
    remove_numbers : bool
    """
    preprocessed_docs = []
    
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
        
        preprocessed_docs.append({'words': text_words})
    
    print()  # fix missing newline from tqdm

    return pd.DataFrame(preprocessed_docs)


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for index, row in preprocessed_docs.iterrows():
        doc_id = index
        doc_words = row['words']
        
        document_by_term[doc_id] = {}
        
        for word in set(row['words']):
            document_by_term[doc_id][word] = doc_words.count(word)

    return pd.DataFrame(document_by_term)


def reduce_terms(df_frequency, max_df=1, min_df=0, max_terms=None):
    """Remove unimportant terms from term-by-document matrix.

    Parameters
    ----------
    df_frequency : pd.DataFrame
    max_df : float , between [0, 1]
             Terms that appear in more % of documents will be ignored
    min_df : float , between [0, 1]
             Terms that appear in less % of documents will be ignored
    max_terms : int , None
                If not None, only top `max_terms` terms will be returned.
    """
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

    corpus_size = df.shape[1]

    # number of non-zero cols
    if 'doc_frequency' not in df_frequency:
        df['doc_frequency'] = df.fillna(0).astype(bool).sum(axis=1)
        
    # idf := log ( len(all_documents) / len(documents_containing_word) )
    df['idf'] = np.log(corpus_size / df['doc_frequency'])
    
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
    
    if not full_matrices:
        k = min(A.shape[0], A.shape[1])
        U = U[:, :k]
        V = V[:k, :]
    
    return U, np.array(s_eigen), V


def transform_to_concept_space(df_tf_idf, k, customSVD=False):
    """Transform data to concept space.
    k : number of concepts
    """
    if k > df_tf_idf.shape[1]:
        k = df_tf_idf.shape[1]
    
    values = df_tf_idf.fillna(0).to_numpy()
    
    if customSVD:
        U, s_eigen, V = custom_svd(values, False)
    else:
        U, s_eigen, V = np.linalg.svd(values, full_matrices=False)
    
    # Get only first k concepts
    S = np.diag(s_eigen[:k])
    
    concept_by_document = S @ (V[:, :k]).T
    query_projection = (U[:, :k]).T
    return pd.DataFrame(concept_by_document), pd.DataFrame(query_projection)


def cosine_similarity(x, y):
    """Returns cosine similarity of two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_n_nearest(df_tf_idf, df_concept_by_doc, df_query_projection, i, n=None, min_sim=0):
    """Returns most similar (column) vectors to `i`-th vector in `arr`.

    Parameters
    ----------
    df_tf_idf : pd.DataFrame
    df_concept_by_doc : pd.DataFrame
    df_query_projection : pd.DataFrame
    i : index of vector to be compared to
    n : return at most `n` vectors
    min_sim : return only vectors at least this similar - @TODO - add functionality or remove parameter
    """
    
    src_vector = df_query_projection.fillna(0).to_numpy() @ (df_tf_idf.fillna(0).to_numpy())[:, i]
    
    df = df_concept_by_doc.apply(func=cosine_similarity, axis=0, args=(src_vector, ))
    
    # Drop column corresponding to the same index
    df = df.drop(df.index[i])
    
    # Set n
    if n is None or n > df.shape[0]:
        n = df.shape[0]
    
    return df.sort_values(ascending=False)[:n]


def preprocess(data_dir='data', cache_dir='cache', max_df=0.75, min_df=0.05, max_terms=2000):
    """Calculate tf-idf for `data_files` and reduce word count based on arguments. Saves output inside `cache_dir`.

    Parameters
    ----------
    data_files : file paths
    cache_dir : str
    max_df : float
    min_df : float
    max_terms : int

    Returns
    -------
    pd.DataFrame with calculated tf-idf
    """
    data_files = [os.path.join(data_dir, f) for f in DATA_FILES]
    df_data = load_data(data_files)
    df_words = preprocess_docs(df_data)
    df_frequency = get_term_by_document_frequency(df_words)
    df_reduced = reduce_terms(df_frequency, max_df=max_df, min_df=min_df, max_terms=max_terms)
    df_tf_idf = get_tf_idf(df_reduced)

    out_file = os.path.join(cache_dir, TF_IDF_FILE)
    df_tf_idf.to_csv(out_file)
    print('Saved tf_idf dataframe to', out_file)

    return df_tf_idf


def compute(df_tf_idf, k=20, cache_dir='cache'):
    """Transform documents to concept space. Saves output inside `cache_dir`.

    Parameters
    ----------
    df_tf_idf : pd.DataFrame
    k : int
        Number of concepts.
    cache_dir : str

    Returns
    -------
    (pd.DataFrame, pd.DataFrame) with (concept_by_doct and query_projection)
    """
    df_concept_by_doc, df_query_projection = transform_to_concept_space(df_tf_idf, k)

    out_file_concept = os.path.join(cache_dir, CONCEPT_FILE)
    out_file_projection = os.path.join(cache_dir, PROJECTION_FILE)
    df_concept_by_doc.to_csv(out_file_concept, header=True, index=True)
    df_query_projection.to_csv(out_file_projection, header=True, index=True)
    print('Saved concept_by_doc dataframe to', out_file_concept)
    print('Saved query_projection dataframe to', out_file_projection)

    return df_concept_by_doc, df_query_projection


class LSA:
    """Wrapper for LSA methods and data."""
    df_data: pd.DataFrame
    df_tf_idf: pd.DataFrame
    df_concept_by_doc: pd.DataFrame
    df_query_projection: pd.DataFrame

    def __init__(self, data_dir='data', cache_dir='cache'):
        data_files = [os.path.join(data_dir, f) for f in DATA_FILES]
        tf_idf_file = os.path.join(cache_dir, TF_IDF_FILE)
        concept_file = os.path.join(cache_dir, CONCEPT_FILE)
        projection_file = os.path.join(cache_dir, PROJECTION_FILE)
        files_to_check = data_files + [tf_idf_file, concept_file, projection_file]
        for f in files_to_check:
            if not os.path.isfile(f):
                raise ValueError("Couldn't read file '{}'.".format(os.path.abspath(f)))

        self.df_data = load_data(data_files)
        self.df_tf_idf = pd.read_csv(tf_idf_file, index_col=0)

        self.df_concept_by_doc = pd.read_csv(concept_file, index_col=0)
        self.df_concept_by_doc.columns = self.df_concept_by_doc.columns.astype(int)

        self.df_query_projection = pd.read_csv(projection_file, index_col=0)
        self.df_query_projection.columns = self.df_query_projection.columns.astype(int)

        # Number of documents not ok
        if len(self.df_data) != self.df_concept_by_doc.shape[1]:
            raise ValueError('Unexpected query concept by document matrix size! Possibly caused by outdated cache.')

        # Number of terms not ok
        if self.df_tf_idf.shape[0] != self.df_query_projection.shape[1]:
            raise ValueError('Unexpected query projection matrix size! Possibly caused by outdated cache.')
        
    def get_n_nearest(self, doc_index, n):
        best_match = get_n_nearest(self.df_tf_idf, self.df_concept_by_doc, self.df_query_projection, doc_index, n)

        df = self.df_data.iloc[best_match.index].copy()
        df['similarity'] = best_match
        return df


def main():
    """Initialize and/or test LSA.

    Main is inside function to provide scope to variables.
    """
    import argparse

    cache_dir = 'cache'
    required_files = [os.path.join(cache_dir, CONCEPT_FILE),
                      os.path.join(cache_dir, PROJECTION_FILE)]
    files_found = all([os.path.isfile(f) for f in required_files])
    needs_init = False if files_found else True

    parser = argparse.ArgumentParser(description='LSA demo.')
    parser.add_argument('--init', action='store_true', dest='init', help='initialize LSA')
    parser.add_argument('--test', action='store_true', dest='test', help='test LSA')

    args = parser.parse_args()

    if args.init or needs_init:
        df_tf_idf = preprocess()
        compute(df_tf_idf)
        print('Initialization complete')

    if args.test:
        lsa = LSA()
        print('Loaded {} documents'.format(len(lsa.df_data)))

        i_test = 2
        n_test = 5
        print('Example for i={} and n={}'.format(i_test, n_test))

        df_test = lsa.get_n_nearest(doc_index=i_test, n=n_test)
        print(df_test)


if __name__ == "__main__":
    main()
