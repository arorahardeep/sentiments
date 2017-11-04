#!/usr/bin/env python

"""
Load the news feed data
"""

import pandas as pd
import nltk
import os
import sys
import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class SentimentsData:

    def __init__(self):
        self.path = "amazon/"
        self.data_path = 'amazon/Reviews.csv'
        self.data = None
        self.data_train = None
        self.data_test = None
        self.dataY = None
        self.dataX = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.glove_embeddings_index = None
        self.embedding_matrix = None
        self.GLOVE_DIR = "./embeddings/"
        self.full_corpus = set()
        self.corpus_size = 0
        self.max_size = 0
        self.EMBEDDING_DIM = 100
        self.word2int = {}
        self.int2word = {}

    def train_tst_split(self):
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.1)


    def load_glove(self):
        print("Loading GLOVE embeddings...")
        embeddings_index = {}
        f = open(os.path.join(self.GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        self.glove_embeddings_index = embeddings_index

        print('Found %s word vectors.' % len(embeddings_index))

    def create_embeddings_matrix_full(self):
        print("Create embedding matrix of all glove embeddings...")
        self.load_glove()
        self.embedding_matrix = np.zeros((len(self.glove_embeddings_index) + 1, self.EMBEDDING_DIM))

        i = 0
        for key, value in self.glove_embeddings_index.items():
            self.embedding_matrix[i] = value
            i += 1
        self.corpus_size = i
        print(self.embedding_matrix.shape)

    def create_embeddings_matrix(self):
        print("Create embedding matrix...")
        self.load_glove()
        self.embedding_matrix = np.zeros((len(self.word2int) + 1, self.EMBEDDING_DIM))
        fd = 0
        for i, word in enumerate(self.word2int):
            embedding_vector = self.glove_embeddings_index.get(str(word))
            if embedding_vector is not None:
                fd += 1
                self.embedding_matrix[i] = embedding_vector # words not found in embedding index will be all-zeros.
        print(self.embedding_matrix.shape)
        print("Total Words %i Embeddings found = %i"%(i, fd))
        #sys.stdin.readline()

    def score2label(self, label):
        if label > 3:
            return 1
        else:
            return 0

    def create_corpus(self, text):
        self.full_corpus = self.full_corpus | set(text)

    def create_corpus_stats(self, df):
        print("Calculating corpus len...")
        df['Corpus_Len'] = df['Corpus'].apply(len)
        self.max_size = df['Corpus_Len'].max()
        print("Max Len %i"%(self.max_size))
        print("Generating all corpus...")
        df['Corpus'].apply(self.create_corpus)
        self.corpus_size = len(self.full_corpus)
        print("Total Vocabulary = %i"%(self.corpus_size))

    def tokenize_imp(self, df):
        print("Tokenizing...")
        print("Generating labels...")
        df['Label']  = df['Score'].apply(self.score2label)
        print("Generating corpus...")
        df['Text'] = df['Text'].apply(lambda x: x.lower())
        df['Corpus'] = df['Text'].apply(nltk.word_tokenize)
        print("Generating corpus stats...")
        self.create_corpus_stats(df)
        return df[['Label','Corpus']]

    def build_lookups(self):
        self.word2int = dict((c, i) for i, c in enumerate(self.full_corpus))
        self.int2word = dict((i, c) for i, c in enumerate(self.full_corpus))

    def encode(self, text):
        enc = []
        for c in text:
            enc.append(self.word2int[c])
        return enc

    def decode(self, text):
        dec = []
        for c in text:
            dec.append(self.int2word[c])
        return dec

    def preprocess_data(self):
        if not (os.path.exists(os.path.join(self.path,"train_data.csv"))):
            review_df = pd.read_csv(self.data_path)[100000:200000]
            print(review_df.shape)
            print("Dropping neutral reviews")
            review_df = review_df.drop(review_df[review_df.Score == 3].index)
            print(review_df.shape)

            review_df = self.tokenize_imp(review_df)

            self.build_lookups()
            self.data = review_df
            self.train_tst_split()
            self.data_train.to_csv(os.path.join(self.path,"train_data.csv"), index=False)
            self.data_test.to_csv(os.path.join(self.path,"test_data.csv"), index=False)
        else:
            print("Reading train file...")
            self.data_train = pd.read_csv(os.path.join(self.path,"train_data.csv"),converters={1:ast.literal_eval})
            print("Reading test file...")
            self.data_test = pd.read_csv(os.path.join(self.path,"test_data.csv"),converters={1:ast.literal_eval})
            print("Generating corpus stats...")
            self.create_corpus_stats(pd.concat([self.data_train,self.data_test]))
            self.build_lookups()

        self.trainX = self.data_train['Corpus'].values
        self.trainY = self.data_train['Label'].values
        self.testX = self.data_test['Corpus'].values
        self.testY = self.data_test['Label'].values

        for row in range(len(self.trainX)):
            self.trainX[row] = self.encode(self.trainX[row])

        for row in range(len(self.testX)):
            self.testX[row] = self.encode(self.testX[row])

        print(self.trainX.shape, self.trainY.shape)
        print(self.testX.shape, self.testY.shape)

    def load(self):
        self.preprocess_data()
        return (self.trainX, self.trainY), (self.testX, self.testY)


def main():
    sd = SentimentsData()
    sd.load()
    sd.create_embeddings_matrix()
    print(sd.max_size)
    print(sd.corpus_size)

if __name__ == "__main__":
    main()
