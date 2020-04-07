#!/usr/bin/env python
# coding: utf-8

# # TODO
# 
# ## Major
# - [x] create term-by-document matrix (calculate words frequncies for each term-document pair)
#  - [ ] check that it's actually correct - especially if we don't map terms to wrong documents
# - [x] convert term-by-document frequencies to tf-idf (calcualte tf-idf for each term-document pair)
#  - [ ] check
# - [x] we may need actual (numpy?) matrix?
# - [x] LSI magic
# - [ ] Put it together
# - [ ] GUI
# 
# ### Minor
# - [x] remove numbers from terms - done but not sure if it's good thing to do, maybe it's also important for relevancy of docs,
# like for example when there is year written?

from lsa import *


np.random.seed(42)


df_data = load_data('articles.csv')


df_data.head(1)


preproccessed_docs = preprocess_docs(df_data)
preproccessed_docs


df_frequency = get_term_by_document_frequency(preproccessed_docs)


df_frequency


df_frequency.shape


reduce_terms(df_frequency, 0.8, 0.1,1000).sort_values('doc_frequency', ascending=False)


df_reduced = reduce_terms(df_frequency, 0.8, 0.1)


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


df_tf_idf = get_tf_idf(df_reduced)
display(df_tf_idf)


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


get_concept_by_document(df_tf_idf, True)


get_concept_by_document(df_tf_idf)


df_concept = get_concept_by_document(df_tf_idf)
df_concept


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


best_match = get_n_nearest(df_concept, 2, 3)
best_match


lsa = LSA()


lsa.load()

