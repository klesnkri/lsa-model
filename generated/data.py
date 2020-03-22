#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpTokenizer


# In[4]:


np.random.seed(42)


# In[5]:


bp_data = pd.read_csv("articles.csv", header=0)


# In[6]:


bp_data.head(1)


# In[9]:


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
    
    for text in docs.loc[:,"text"]:
        text_words = tokenizer.tokenize(text)
        
        if use_lemmatizer:
            text_words = [lemmatizer.lemmatize(word, pos="v").lower() for word in text_words if word.lower() not in en_stop]
        else:
            text_words = [stemmer.stem(word).lower() for word in text_words if word.lower() not in en_stop]
        
        preproccessed_docs.append(text_words)
    
    return preproccessed_docs


# In[10]:


preproccessed_docs = preprocess_docs(bp_data)
preproccessed_docs[0]


# In[ ]:


def get_term_by_document(preprocess_docs):
    
    pass


# In[11]:


doc_0 = preproccessed_docs[0]


# In[12]:


doc_0.count('ai')

