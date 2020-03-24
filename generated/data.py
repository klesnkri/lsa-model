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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import umap


np.random.seed(42)


bp_data = pd.read_csv("articles.csv", header=0)


bp_data.head(1)


def get_lemmatization_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dictionary = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dictionary.get(tag, wordnet.NOUN)


def preprocess_docs(docs, use_lemmatizer = True):
    '''Tokenize and preprocess documents
    
    Parameters
    ----------
    use_lemmatizer : bool
                     Uses lemmazizer if True, othrerwise uses stemmer.
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
    
    for row in docs.itertuples(index=True, name='Doc'):
        text = row.text
        
        # remove numbers
        text = re.sub(r'\d+', '', text)
        
        text_words = tokenizer.tokenize(text)
        
        if use_lemmatizer:
            text_words = [lemmatizer.lemmatize(word.lower(), get_lemmatization_pos(word.lower())) for word in text_words
                          if word not in string.punctuation and word.lower() not in en_stop]
        else:
            text_words = [stemmer.stem(word.lower()) for word in text_words
                         if word not in string.punctuation and word.lower() not in en_stop]
        
        preproccessed_docs.append({'words': text_words})
    
    pdocs = pd.DataFrame(preproccessed_docs)
    return pdocs


preproccessed_docs = preprocess_docs(bp_data)
preproccessed_docs


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for index, row in preprocessed_docs.iterrows():
        doc_id = index
        doc_words = row['words']
        
        # computed later, @TODO: move computation here and fix below or remove
#         document_by_term[doc_id] = {
#             'total_words': len(doc_words)
#         }
        document_by_term[doc_id] = {}
        
        for word in set(row['words']):
            document_by_term[doc_id][word] = doc_words.count(word)

    df = pd.DataFrame(document_by_term)
    
    return df


df_frequency = get_term_by_document_frequency(preproccessed_docs)


df_frequency


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


reduce_terms(df_frequency).sort_values('doc_frequency', ascending=False).shape


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


def get_concept_by_document(df_tf_idf):
    '''Transform data to concept space.
    '''
    values = df_tf_idf.fillna(0).to_numpy()
    U, s_eigen, V = np.linalg.svd(values, full_matrices=False)
    
    S = np.diag(s_eigen)
    
    concept_by_document = S @ V.T
    return pd.DataFrame(concept_by_document)


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
    return df
    # skip first value - the src_vector itself
    if n:
        return df.sort_values(ascending=False)[1:n + 1]
    else:
        return df.sort_values(ascending=False)


get_n_nearest(df_concept, 1, 3)

