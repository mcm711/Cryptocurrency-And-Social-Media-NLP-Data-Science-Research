#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:51:59 2023

@author: matthewmcmurry
"""

import pandas as pd
from umap import umap_ as UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np


# might cause kernel to restart
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# import tensorflow_hub
# embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Step 2 - Reduce dimensionality
# umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')

#if previous line throws a "'module' object is not callable" error  use this line of code
umap_model = UMAP.UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()


def create_topic_model(df):
    for idx, row in df.iterrows():
        # create docs variable
        docs = row['merged'].text.values
        
        # create Topic_Model_xyz variable
        topic_model_name = 'Topic_Model_' + row['segment']
        globals()[topic_model_name] = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            n_gram_range=(1, 3),
            min_topic_size=25,
            calculate_probabilities=False,
            verbose=True
        )
        
        # fit the model
        topics, probs = globals()[topic_model_name].fit_transform(docs)
        
        # save the model to disk
        model_path = '/Users/matthewmcmurry/Documents/Text Data/Reddit/subreddits/' + topic_model_name
        globals()[topic_model_name].save(model_path)
        print("Saved " + topic_model_name)
        
        # add the model to the dataframe
        df.at[idx, 'Topic_Model'] = globals()[topic_model_name]
    return df

topics, probs = topic_model.fit_transform(docs)
topics_Q3_2020 = topic_model.get_topic_info()


topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=12)
topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

#update the topic model and reduce the topics
topic_model.update_topics(docs, n_gram_range=(1, 2))
topic_model.reduce_topics(docs, nr_topics=6)

topic_docs = {topic: [] for topic in set(topics)}
for topic, doc in zip(topics, docs):
     topic_docs[topic].append(doc)
    
    
