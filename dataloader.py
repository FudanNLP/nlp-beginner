from utils import *
import pandas as pd
import numpy as np 
from random import shuffle

class data_loader():
    def __init__(self, path="./train.tsv", model="BOW"):
        self.data = pd.read_csv(path, sep='\t')[["Phrase", "Sentiment"]].values.tolist()
        self.index = 0
        self.length = len(self.data)
        self.order = None
        self.p = 1
        self.model = load_BOW() if model == "BOW" else load_Ngrams

    def preprocess(self, raw):
        sentence, sentiment= raw[0], raw[1]
        x = self.model
        for word in process(sentence):
            x[word] += 1
        y = np.zeros(5)
        y[sentiment-1] = 1
        return [np.array([list(x.values())]).reshape(19422,1), y.reshape(5,1)]

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        return self.preprocess(self.data[self.index-1])
    
    def get_shuffle_batch(self, batch_size=64):
        self.order = np.arange(self.length)
        if self.p == 1 or self.p > self.length:
            shuffle(self.order)
        self.p += batch_size
        return [self.preprocess(i) for i in self.data[self.p:self.p + batch_size] ]

