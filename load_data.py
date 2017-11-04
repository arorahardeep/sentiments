#!/usr/bin/env python

"""
Load the news feed data
"""

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class SentimentsData:

    def __init__(self):
        self.path = "stocknews/"
        self.data_path = 'stocknews/Combined_News_DJIA.csv'
        self.data = None
        self.data_train = None
        self.data_test = None
        self.dataY = None
        self.dataX = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.embeddings_index = None
        self.embedding_matrix = None
        self.GLOVE_DIR = "./embeddings/"
        self.full_corpus = []
        self.corpus_size = 0
        self.max_size = 0
        self.EMBEDDING_DIM = 100

    def train_tst_split(self):
        self.data_train, self.data_test = train_test_split(self.data, test_size=0.3)
        #self.data_train = self.data[self.data['Date'] < '2016-01-01']
        #self.data_test  = self.data[self.data['Date'] > '2015-12-31']


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
        self.embeddings_index = embeddings_index

        print('Found %s word vectors.' % len(embeddings_index))

    def create_embeddings_matrix(self):
        print("Create embedding matrix...")
        self.load_glove()
        self.embedding_matrix = np.zeros((len(self.full_corpus) + 1, self.EMBEDDING_DIM))
        for i, word in enumerate(self.full_corpus):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

    def tokenize(self, text):
        print("Tokenizing...")
        corpus = []
        corpus_len = []
        label = []
        date = []
        for row in range(0,len(text.index)):
            #all_news = CountVectorizer().build_tokenizer()(' '.join(str(x).lower() for x in text.iloc[row,:]))
            for x in text.iloc[row,2:27]:
                x = str(x).lower()
                label.append(text.iloc[row,1])
                date.append(text.iloc[row,0])
                all_news= CountVectorizer().build_tokenizer()(x)
                corpus.append(all_news)
                corpus_len.append(len(all_news))
                self.full_corpus = self.full_corpus + all_news
        self.full_corpus = set(self.full_corpus)
        self.corpus_size = len(self.full_corpus)
        self.max_size = max(corpus_len)
        print("Total Vocabulary = %i"%(self.corpus_size))
        return pd.DataFrame(
                {'Date': date,
                 'Label': label,
                 'Corpus': corpus
                })

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
            news_df = pd.read_csv(self.data_path)
            news_df = self.tokenize(news_df)
            self.build_lookups()
            self.data = news_df
            self.train_tst_split()
            self.data_train.to_csv(os.path.join(self.path,"train_data.csv"), index=False)
            self.data_test.to_csv(os.path.join(self.path,"test_data.csv"), index=False)
        else:
            self.data_train = pd.read_csv(os.path.join(self.path,"train_data.csv"))
            self.data_test = pd.read_csv(os.path.join(self.path,"test_data.csv"))

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
    print(sd.max_size)
    print(sd.corpus_size)

if __name__ == "__main__":
    main()
