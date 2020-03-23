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
# - [x] remove numbers from terms - done but not sure if it's good thing to do, maybe it's also important for relevancy of docs,
# like for example when there is year written?

# In[17]:


import pandas as pd
import numpy as np
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


np.random.seed(42)


# In[19]:


bp_data = pd.read_csv("articles.csv", header=0)


# In[20]:


bp_data.head(1)


# In[118]:


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
            text_words = [lemmatizer.lemmatize(word, pos="v").lower() for word in text_words
                          if word not in string.punctuation and word.lower() not in en_stop]
        else:
            text_words = [stemmer.stem(word).lower() for word in text_words
                         if word not in string.punctuation and word.lower() not in en_stop]
        
        preproccessed_docs.append({'words': text_words})
    
    pdocs = pd.DataFrame(preproccessed_docs)
    return pdocs


# In[119]:


preproccessed_docs = preprocess_docs(bp_data)
display(preproccessed_docs)


# In[128]:


def get_term_by_document_frequency(preprocessed_docs):
    document_by_term = {}
    
    for index, row in preprocessed_docs.iterrows():
        doc_id = index
        doc_words = row['words']
        
        document_by_term[doc_id] = {
            'total_words': len(doc_words)
        }
        
        
        for word in set(row['words']):
            document_by_term[doc_id][word] = doc_words.count(word)

    df = pd.DataFrame(document_by_term)
    
    return df


# In[129]:


df_frequency = get_term_by_document_frequency(preproccessed_docs)


# In[130]:


df_frequency


# In[131]:


def get_tf_idf(df_frequency):
    df = df_frequency.copy()
    # tf := word frequency / total frequency
    df = df.drop('total_words', inplace=False)[:] / df.loc['total_words']
    
    # idf := log ( len(all_documents) / len(documents_containing_word) )
    
    corpus_size = df.shape[1]

    # number of non-zero cols
    df['doc_frequency'] = df.fillna(0).astype(bool).sum(axis=1)
        
    df['idf'] = np.log( corpus_size / df['doc_frequency'] )
    
    # tf-idf := tf * idf
    _cols = df.columns.difference(['idf', 'doc_frequency'])
    df[_cols] = df[_cols].multiply(df["idf"], axis="index")
    
    df.drop(columns=['doc_frequency', 'idf'], inplace=True)
    
    return df


# In[132]:


df_tf_idf = get_tf_idf(df_frequency)
display(df_tf_idf)


# In[ ]:





# In[ ]:





# In[ ]:




