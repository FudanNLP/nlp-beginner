from utils import *
import pandas as pd
import numpy as np 
from random import shuffle
import collections
import pprint

class dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(self.x)
        self.num_of_inputs = len(x[0])
        self.order = [i for i in range(self.length)]      
        self.index = 0
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.index + 1 < self.length:
            self.index = 0
            raise StopIteration()
        self.index += 1
        return [self.x[self.index], self.y[self.index]]

    def get_batch(self, shuffled_batch=True, batch_size=64):
        if shuffled_batch and (self.p == 0 or self.p >= self.length):
            shuffle(self.order)
            x_new, y_new = [], []
            for i in self.order:
                x_new.append(self.x[i])
                y_new.append(self.y[i])
            self.p = 0
        self.p += batch_size
        x = np.array(self.x[self.p - batch_size : self.p])
        y = np.array(self.y[self.p - batch_size : self.p])

        return np.array(x).T, np.array(y).squeeze(2).T


class data_loader():
    def __init__(self, path="./train.tsv", model="BOW", length=20000, classes=5, ratio=0.8):
        self.path=path
        self.length=length
        self.classes=classes
        self.ratio = ratio
        self.model = None
    def preprocess(self, sentence):
        return BOWTransform(sentence, self.model)

    def load(self):
        print("Start Loading dataset...")
        rawdata = pd.read_csv(self.path, sep='\t')
        rawphrase = rawdata["Phrase"].tolist()

        self.model = generate_BOW(rawphrase)

        rawsentiment = rawdata["Sentiment"].tolist()
        mean_count = {i:0 for i in range(self.classes)}
        x, y = [], []
        for i  in range(len(rawdata)):
            if mean_count[rawsentiment[i]] < self.length // self.classes:
                mean_count[rawsentiment[i]] += 1
                x.append(rawphrase[i])
                y.append(np.array([onehot(5, rawsentiment[i])]).T)

        x = self.preprocess(x)

        print("Finished Loading !")

        return dataset(x[:int(self.ratio*self.length)], y[:int(self.ratio*self.length)]), \
            dataset(x[int(self.ratio*self.length):], y[int(self.ratio*self.length):])


def test_dataloader(dataset):
    print("Dataloader Test Starts!")
    
    count = 0
    for i in dataset: 
        pass
    batchsize = 128

    for i in range(dataset.length // batchsize + 1):
        x, y = dataset.get_batch(batchsize)
        print(x.shape, y.shape)
    print("Dataloader Test pass !")

def check_dataset(path="./train.tsv"):
    rawdata = pd.read_csv(path, sep="\t")["Sentiment"].tolist()
    from collections import Counter
    print(Counter(rawdata[:1000]))
    print(Counter(rawdata[:10000]))
    print(Counter(rawdata[:20000]))


if __name__ == "__main__":
    #check_dataset()

    dataloader = data_loader(path="./train.tsv", model="BOW", length=20, classes=5, ratio= 0.8)
    
    train_dataset, test_dataset = dataloader.load()

    test_dataloader(train_dataset);test_dataloader(test_dataset)


