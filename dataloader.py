from utils import *
import pandas as pd
import numpy as np 
from random import shuffle
import collections

class data_loader():
    def __init__(self, path="./train.tsv", model="BOW", m = 20000):
        self.rawdata = pd.read_csv(path, sep='\t')[0:m]
        self.phrase = self.rawdata["Phrase"].tolist()[0:m]
        self.sentiment = self.rawdata["Sentiment"].tolist()[0:m]
        self.index = 0
        self.length = len(self.rawdata)
        self.order = None
        self.p = 0
        self.model = load_BOW() if model == "BOW" else load_Ngrams
        self.data = self.get_matrix(0, m)

    def preprocess(self, sentence):
        x = self.model
        for word in process(sentence):
            x[word] += 1
        return np.array(list(x.values()))
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.index < self.length:
            raise StopIteration()
        self.index += 1
        return self.preprocess(self.phrase[self.index-1]), self.sentiment[self.index-1]

    def get_matrix(self, start, length):
        x = np.array([self.preprocess(i) for i in self.phrase[start:start + length]]).T
        y = np.array([onehot(5, j) for j in self.sentiment[start:start + length]]).T
        return [x, y]

    def get_shuffle_batch(self, batch_size=64):
        if self.p == 0 or self.p >= self.length:
            self.order = np.arange(self.length)
            shuffle(self.order)
            self.p = 0
        self.p += batch_size
        return self.get_matrix(self.p - batch_size, batch_size)

