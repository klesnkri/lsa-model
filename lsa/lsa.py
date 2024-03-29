import pandas as pd
import numpy as np
import string
import nltk
import re
import math
import os
import os.path
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


def preprocess_docs(df, use_lemmatizer=True, remove_numbers=True):
    """Tokenize and preprocess documents.

    Does lemmatization/stemming. Removes stopwords.

    Parameters
    ----------
    df : pd.DataFrame
    use_lemmatizer : bool
                     Uses lemmatizer if True, otherwise uses stemmer.
    remove_numbers : bool
    """
    # English stop words list
    stops = set(string.punctuation).union(set(stopwords.words('english')))

    # Word tokenizer that removes punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    tag_dictionary = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

    if remove_numbers:
        df['content'] = df['content'].apply(lambda text: re.sub(r'\d+', '', text))

    # https://stackoverflow.com/questions/41674573/how-to-apply-pos-tag-sents-to-pandas-dataframe-efficiently
    texts = df['content'].tolist()
    # lemmatizer / Stemmer
    if use_lemmatizer:
        lemmatizer = WordNetLemmatizer()

        tagged_texts = nltk.pos_tag_sents(map(tokenizer.tokenize, texts))
        # for each row
        #     get word and it's tag
        #     lookup wordnet tag first letter in dictionary to get word type
        #     lemmatize
        tokens = [[lemmatizer.lemmatize(word, tag_dictionary.get(tag[0], wordnet.NOUN)) for word, tag in row] for
                  row in tagged_texts]
    else:
        stemmer = SnowballStemmer("english")

        tokens = map(tokenizer.tokenize, texts)
        tokens = [[stemmer.stem(word) for word in row] for row in tokens]

    df['tokens'] = tokens
    # to lowercase
    df['tokens'] = df['tokens'].map(lambda row: list(map(str.lower, row)))
    # remove stopwords
    df['tokens'] = df['tokens'].apply(lambda r: list(filter(lambda t: t not in stops, r)))

    return df


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for index, row in preprocessed_docs.iterrows():
        doc_id = index
        doc_words = row['tokens']
        
        document_by_term[doc_id] = {}
        
        for word in set(row['tokens']):
            document_by_term[doc_id][word] = doc_words.count(word)

    return pd.DataFrame(document_by_term)


def reduce_terms(df_frequency, max_df=1.0, min_df=1, max_terms=None, keep_less_freq=False):
    """Remove unimportant terms from term-by-document matrix.

    Parameters
    ----------
    df_frequency : pd.DataFrame
    max_df : float , between [0, 1]
             Terms that appear in more % of documents will be ignored
    min_df : int
             Terms that appear in <= number of documents will be ignored
    max_terms : int , None
                If not None or 0, only top `max_terms` terms will be returned.
    keep_less_freq : bool
                Decides wherever to keep most frequent or least frequent words when `max_terms` < len.
    """
    df = df_frequency.copy()
    corpus_size = df.shape[1]

    df['doc_apperance'] = df.fillna(0).astype(bool).sum(axis=1)
    df['doc_frequency'] = df['doc_apperance'] / corpus_size
            
    df = df[df.doc_frequency <= max_df]
    df = df[df.doc_apperance > min_df]
    
    if max_terms is not None and max_terms != 0 and max_terms < df.shape[0]:
        df = df.sort_values('doc_frequency', ascending=keep_less_freq)
        df = df.head(max_terms)

    return df.drop('doc_apperance', axis=1)


def get_tf_idf(df_reduced,first_normalization=True):
    df = df_reduced.copy()

    df = df.drop('doc_frequency', axis=1)
    
    # tf := word frequency in doc / total words in doc
    if first_normalization:
        df.loc['total_words'] = df.sum()
        df = df / df.loc['total_words']
        df = df.drop('total_words')
    # tf := word frequency in doc / max word frequency in all docs
    else:
        df['max_freq'] = df.max(axis=1)
        df = df.iloc[:, :-1].div(df['max_freq'], axis=0)

    
    # idf := log ( len(all_documents) / len(documents_containing_word) )
    # doc frequency was already computed in previous step - reuse
    df['idf'] = np.log(1 / df_reduced['doc_frequency'])
    
    # tf-idf := tf * idf
    _cols = df.columns.difference(['idf', 'doc_frequency'])
    df[_cols] = df[_cols].multiply(df["idf"], axis="index")
    
    df.drop(columns=['idf'], inplace=True)
    
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
    
    # if k == 0 -> k = number of terms, go through documents sequentially
    if k > df_tf_idf.shape[1] or k == 0:
        k = df_tf_idf.shape[1]
    
    values = df_tf_idf.fillna(0).to_numpy()
    
    if customSVD:
        U, s_eigen, V = custom_svd(values, False)
    else:
        U, s_eigen, V = np.linalg.svd(values, full_matrices=False)
    
    # Get only first k concepts
    S = np.diag(s_eigen[:k])

    # concept_by_document = S @ V[:, :k]
    concept_by_document = S @ V[:k, :]
    query_projection = (U[:, :k]).T
    return pd.DataFrame(concept_by_document), pd.DataFrame(query_projection)


def cosine_similarity(x, y):
    """Returns cosine similarity of two vectors."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_n_nearest(df_tf_idf, df_concept_by_doc, df_query_projection, i, n=None):
    """Returns most similar (column) vectors to `i`-th vector in `arr`.

    Parameters
    ----------
    df_tf_idf : pd.DataFrame
    df_concept_by_doc : pd.DataFrame
    df_query_projection : pd.DataFrame
    i : index of vector to be compared to
    n : return at most `n` vectors
    """
    
    src_vector = df_query_projection.fillna(0).to_numpy() @ (df_tf_idf.fillna(0).to_numpy())[:, i]
    
    df = df_concept_by_doc.apply(func=cosine_similarity, axis=0, args=(src_vector, ))
    
    # Drop column corresponding to the same index
    df = df.drop(df.index[i])
    
    # Set n
    if n is None or n > df.shape[0]:
        n = df.shape[0]
    
    return df.sort_values(ascending=False)[:n]


def preprocess(data_dir='data', cache_dir='cache', max_df=0.75, min_df=1, max_terms=0,
               use_lemmatizer=False, remove_numbers=True, keep_less_freq=False, first_normalization=True):
    """Calculate tf-idf for `data_files` and reduce word count based on arguments. Saves output inside `cache_dir`.

    Parameters
    ----------
    data_dir : str
            data files dir path
    cache_dir : str
    max_df : float
    min_df : int
    max_terms : int
    use_lemmatizer : bool
    remove_numbers : bool
    keep_less_freq : bool
                Decides wherever to keep most frequent or least frequent words when `max_terms` > len.

    Returns
    -------
    pd.DataFrame with calculated tf-idf
    """
    data_files = [os.path.join(data_dir, f) for f in DATA_FILES]
    df_data = load_data(data_files)
    df_words = preprocess_docs(df_data, use_lemmatizer=use_lemmatizer, remove_numbers=remove_numbers)
    df_frequency = get_term_by_document_frequency(df_words)
    df_reduced = reduce_terms(df_frequency, max_df=max_df, min_df=min_df, max_terms=max_terms,
                              keep_less_freq=keep_less_freq)
    df_tf_idf = get_tf_idf(df_reduced, first_normalization=first_normalization)

    out_file = os.path.join(cache_dir, TF_IDF_FILE)
    df_tf_idf.to_csv(out_file)
    print('Saved tf_idf dataframe to', out_file)

    return df_tf_idf


def compute(df_tf_idf, k=20, cache_dir='cache', customSVD=False):
    """Transform documents to concept space. Saves output inside `cache_dir`.

    Parameters
    ----------
    df_tf_idf : pd.DataFrame
    k : int
        Number of concepts.
    cache_dir : str
    customSVD : bool

    Returns
    -------
    (pd.DataFrame, pd.DataFrame) with (concept_by_doct and query_projection)
    """
    df_concept_by_doc, df_query_projection = transform_to_concept_space(df_tf_idf, k=k, customSVD=customSVD)

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

    def __init__(self, data_dir='data', cache_dir='cache', data_files=DATA_FILES):
        data_files = [os.path.join(data_dir, f) for f in data_files]
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
