#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:14:00 2019

@author: antonio
"""
import gensim
import numpy as np
from spacy.lang.es import Spanish
import os
import string

np.random.seed(400)

###### Get files ###### 
def preprocess(text):
    # Tokenize, remove stopwords, numbers, emtpy spaces and punctuation and lemmatize
    tokenized = []
    nlp = Spanish()
    doc = nlp(text)
    token_list = []
    # Tokenize
    for token in doc:
        # Remove stopwords, numbers, emtpy spaces and punctuation and lemmatize
        if ((token.text not in nlp.Defaults.stop_words) & 
            (token.text not in string.punctuation) &
            (token.text.isalpha() == True)):
            token_list.append(token.lemma_)
    tokenized.append(token_list)
    return tokenized

def load_files(path):
    # Open a read files
    texts = []
    for root, dirs, files in os.walk(path):
         for file in files:
             texts.append(open(os.path.join(root,file), 'r').read())
    return texts


def unlist(texts):
    # flatten texts list
    unlisted_texts = []
    for sublist in texts:
        for item in sublist:
            unlisted_texts.append(item)
    return unlisted_texts


###### Distance ######
def get_topic_proportions(lda_model, bow_corpus, num_topics):
    # Get, for each document, the topic proportions assigned by LDA
    topic_prop=lda_model.get_document_topics(bow=bow_corpus)
    
    proportions = np.zeros((len(topic_prop), num_topics))
    
    for i in range(len(topic_prop)):
        doc_topics = topic_prop[i]
        for j in range(len(doc_topics)):
            topic = doc_topics[j]
            # topic[0] = number of topic
            # topic[1] = proportion of the topic in the document
            proportions[i, topic[0]] = topic[1]
            
    return proportions


def compute_distances(document_id, topic_proportions):
    # Compute Euclidean distance based on topic proportions of each document 
    # (probably not the best distance...)
    distances=[]
    for i in range(len(topic_proportions)):
        distance=np.linalg.norm(np.array(topic_proportions[i])-
                                np.array(topic_proportions[document_id]))
        distances.append(distance)
    return distances

    
def main():
    ###### Get and prepare texts ###### 
    texts = load_files('/home/antonio/data/corpora/SpaCCC/es/')
    clean_texts = [preprocess(text) for text in texts]
    unlisted_texts = unlist(clean_texts) # flatten nested list

    ###### Bag of words ######
    # Create a dictionary all texts
    dictionary = gensim.corpora.Dictionary(unlisted_texts)
   
    # We remove words that haven been seldom used
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=100000)
    
    # Checking dictionary created
    print('Check dictionary has been created')
    count = 0
    for k, v in dictionary.iteritems():
        count += 1
        if count < 7:
            print(k, v)
    
    # Bag of words on the dataset
    bow_corpus = [dictionary.doc2bow(doc) for doc in unlisted_texts]
    
    ###### LDA ######
    # Running LDA using Bag of Words
    num_topics = 20
    n_epoch = 50
    lda_20 = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, num_topics=num_topics, 
                                             update_every=0, id2word=dictionary,
                                             passes = n_epoch)
    
    ###### SHOW RESULTS ######
    # Print the 10 most probable words in every topic
    print('\n\nThe  10 most probable words in every topic are:')
    most_prob_word_topic=lda_20.show_topics(num_topics=20, num_words=10)
    #print(most_prob_word_topic)
    
    #print('\nNicer way to see the same:')
    for i in range(len(most_prob_word_topic)):
        clean_words=most_prob_word_topic[i][1].split(" + ")
        for j in range(len(clean_words)):
            clean_words[j]=clean_words[j][7:-1]
        print("Topic {}: {}".format(most_prob_word_topic[i][0], clean_words))
    
    
    # Print the proportion of topics in a document
    document_num=20
    print('\n\nThe topic proportion in document {} is:'.format(document_num))
    topic_prop_17320=lda_20.get_document_topics(bow=bow_corpus[document_num])
    for i in range(len(topic_prop_17320)):
        print("Topic {}: {}".format(topic_prop_17320[i][0], topic_prop_17320[i][1]))
        
    ####### Show similar documents ########
    topic_prop = get_topic_proportions(lda_20, bow_corpus, num_topics)
    
    # 2. Euclidean distance to selected document
    distances = compute_distances(document_num, topic_prop)
    
    indexes=sorted(range(len(distances)), key=lambda k: distances[k])
    print('\n\nThe most similar documents to document {} in ascending order are:'.format(document_num))
    for j in range(10):
        print("Document number {}: {}".format(indexes[j+1], texts[indexes[j+1]][0:70]))
        

if __name__ == '__main__':
    main()
