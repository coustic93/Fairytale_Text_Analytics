#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 21:43:17 2021

@author: yoonjae
"""

#%% Import Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
import urllib
import re
from bs4 import BeautifulSoup as soup
import numpy as np
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import operator

import math

from nltk import sent_tokenize, word_tokenize, PorterStemmer

#%% Read URl
url = urllib.request.urlopen('https://www.gutenberg.org/files/2591/2591-h/2591-h.htm#link2H_4_0026')
doc = url.read()

#%% Extracts the separate chapters, gets the text, and adds them to a main list
tree = soup( doc,'lxml' )
#Gets all the div = chapter classes
div = tree.find_all( 'div', { 'class': 'chapter' } )

#Takes the div and gets only the text without the header
#FIXME: Does not seem to be including the quotes
text_list = []
for i in range(len(div)):
    #Gets all paragraph tags for a given paragraph
    temp_list = div[i].find_all(['p','pre'])
    #Temporarily stores the full story to then be appended to the text list
    story_temp = ""
    #Loops through all paragraphs and concatenate them together
    for i in temp_list:
        i = i.getText().replace('\r',"").replace('\n',"").strip().split()
        i = ' '.join(i)
        story_temp = story_temp + ' ' + i
    story_temp = story_temp.replace("’","").replace("’","").replace("’","").replace('”','').replace("‘","")
    story_temp = story_temp.replace("\\","")
    #Append the full story to the final text list
    text_list.append(story_temp)

#Removes the title from the list so this is just the relevant chapters.
#TODO: Do this before all other processing just for logical order
del text_list[0]


#%%

article = re.split(r"""\? |\. |\.\' """,text_list[9])
pre_sentences = []

for sentence in article:
    pre_sentences.append(sentence.replace("[^a-zA-Z]", " "))

sentences = []
for sent in pre_sentences:
    if len(sent) > 50:
        sentences.append(sent)
    

#deletes the odd black space at the start of each story

# Tokenize Sentences
#sentences = sent_tokenize(text_list[1]) # NLTK function

total_documents = len(sentences)

#%%
# Create Frequency Matrix

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent] = freq_table

    return frequency_matrix

#%%
# Create Term Frequency
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

#%%
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

#%%
# Calculate IDF

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

#%%
# Score sentences

def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}
    
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

#%%
# Generate Summary
def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence in sentenceValue and sentenceValue[sentence] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def _best_ranks(sentences, sentenceValue):
    best_sent =[]
    sorted_d = dict(sorted(sentenceValue.items(), key=operator.itemgetter(1),reverse=True))
    for sent,num in sorted_d.items():
        if sent in sentences and len(best_sent) < 5:
              best_sent.append(sent)

    summary = []
    
    for sent in sentences:
        if sent in best_sent and sent not in summary:
            summary.append(sent)
    return summary

    
#%%

# 2 Create the Frequency matrix of the words in each sentence.
freq_matrix = _create_frequency_matrix(sentences)
#print(freq_matrix)

'''
Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
'''
# 3 Calculate TermFrequency and generate a matrix
tf_matrix = _create_tf_matrix(freq_matrix)
#print(tf_matrix)

# 4 creating table for documents per words
count_doc_per_words = _create_documents_per_words(freq_matrix)
#print(count_doc_per_words)

'''
Inverse document frequency (IDF) is how unique or rare a word is.
'''
# 5 Calculate IDF and generate a matrix
idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
#print(idf_matrix)

# 6 Calculate TF-IDF and generate a matrix
tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
#print(tf_idf_matrix)

# 7 Important Algorithm: score the sentences
sentence_scores = _score_sentences(tf_idf_matrix)
#print(sentence_scores)

# 8 Find the threshold
threshold = _find_average_score(sentence_scores)
#print(threshold)

# 9 Important Algorithm: Generate the summary
summary = _generate_summary(sentences, sentence_scores, 1.2 * threshold)
print(summary)
#summary = _best_ranks(sentences, sentence_scores)
#print("Summarize Text: \n" + ". \n".join(summary))



