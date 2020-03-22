#!/usr/bin/env python
# coding: utf-8

# # TODO
# 
# ## Major
# - [x] create term-by-document matrix (calculate words frequncies for each term-document pair)
#  - [ ] check that it's actually correct - especially if we don't map terms to wrong documents
# - [x] convert term-by-document frequencies to tf-idf (calcualte tf-idf for each term-document pair)
#  - [ ] check
# - [ ] we may need actual (numpy?) matrix?
# - [ ] LSI magic
# 
# ### Minor
# - remove numbers from terms?

# In[224]:


import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpTokenizer


# In[225]:


np.random.seed(42)


# In[226]:


bp_data = pd.read_csv("articles.csv", header=0)


# In[227]:


bp_data.head(1)


# In[228]:


def preprocess_docs(docs, use_lemmatizer=True):
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
    
    for row in bp_data.itertuples(index=True, name='Doc'):
        text = row.text
        text_words = tokenizer.tokenize(text)
        
        if use_lemmatizer:
            text_words = [lemmatizer.lemmatize(word, pos="v").lower() for word in text_words if word.lower() not in en_stop]
        else:
            text_words = [stemmer.stem(word).lower() for word in text_words if word.lower() not in en_stop]
        
        preproccessed_docs.append({'id': row.Index, 'words': text_words})
    
    return preproccessed_docs


# In[229]:


preproccessed_docs = preprocess_docs(bp_data)


# In[230]:


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for doc_data in preprocessed_docs:
        doc_id = doc_data['id']
        doc_words = doc_data['words']
        
        document_by_term[doc_id] = {
            'total_words': len(doc_words)
        }
        
        
        for word in set(doc_data['words']):
            document_by_term[doc_id][word] = doc_words.count(word)

    df = pd.DataFrame(document_by_term)
    
    return df


# In[231]:


df_frequency = get_document_by_term_frequency(preproccessed_docs)


# In[232]:


df_frequency


# In[233]:


def get_tf_idf(df_frequency):
    df = df_frequency.copy()
    # tf := word frequency / total frequency
    df.drop('total_words', inplace=False)[:] /= df.loc['total_words']
    # idf := log ( len(all_documents) / len(documents_containing_word) )

    corpus_size = df_tf.shape[1]
    # number of non-zero cols + 1 to avoid division by zero
    df['doc_frequency'] = df.fillna(0).astype(bool).sum(axis=1) + 1 
    
    df['idf'] = np.log( corpus_size / df['doc_frequency'] )
    # tf-idf := tf * idf
    _cols = df.columns.difference(['idf', 'doc_frequency'])
    df[_cols] = df[_cols].multiply(df["idf"], axis="index")
    
    df.drop(columns=['doc_frequency', 'idf'], inplace=True)
    df.drop('total_words', inplace=True)
    
    return df


# In[235]:


df_tf_idf = get_tf_idf(df_frequency)
df_tf_idf

