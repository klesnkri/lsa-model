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


df_data = load_data(['articles.csv', 'recipes.csv'])


df_data.head(1)


preproccessed_docs = preprocess_docs(df_data)
preproccessed_docs


df_frequency = get_term_by_document_frequency(preproccessed_docs)


df_frequency


df_frequency.shape


df_reduced = reduce_terms(df_frequency, 0.75, 0.05,1200).sort_values('doc_frequency', ascending=False)
df_reduced


df_tf_idf = get_tf_idf(df_reduced)
display(df_tf_idf)


df_concept_by_doc, df_query_projection = transform_to_concept_space(df_tf_idf,20)
display(df_concept_by_doc)
pd.set_option('max_columns', None)
display(pd.DataFrame(df_concept_by_doc.idxmax()).transpose())
#display(df_query_projection)


best_match = get_n_nearest(df_tf_idf, df_concept_by_doc, df_query_projection, 2, 10)
best_match


lsa = LSA()


lsa.load()

