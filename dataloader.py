from utils import *
import pandas as pd
import numpy as np 
from random import shuffle
import collections


class dataset():
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)        
        self.index = 0
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.index + 1 < self.length:
            self.index = 0
            raise StopIteration()
        self.index += 1
        return self.data[self.index]

    def get_batch(self, shuffled_batch=True, batch_size=64):
        if shuffled_batch and self.p == 0 or self.p >= self.length:
            shuffle(self.data)
            self.p = 0
        self.p += batch_size

        x, y = list(map(list, zip(*self.data[self.p - batch_size : self.p])))

        return np.array(x).reshape(len(x[0]), len(x)), np.array(y).reshape(len(y[0]), len(y))


class data_loader():
    def __init__(self, path="./train.tsv", model="BOW", length=20000, classes=5, ratio=0.8):
        self.path=path
        self.model=load_BOW() if model == "BOW" else load_Ngram()
        self.length=length
        self.classes=classes
        self.ratio = ratio

    def preprocess(self, sentence):
        model = self.model
        for word in process(sentence):
            model[word] += 1
        return np.array([[value] for value in model.values()])

    def load(self):
        print("Start Loading dataset...")
        rawdata = pd.read_csv(self.path, sep='\t')
        rawphrase = rawdata["Phrase"].tolist()
        rawsentiment = rawdata["Sentiment"].tolist()
        mean_count = {i:0 for i in range(self.classes)}
        data = []
        for i  in range(len(rawdata)):
            if mean_count[rawsentiment[i]] < self.length // self.classes:
                mean_count[rawsentiment[i]] += 1
                x = np.array(self.preprocess(rawphrase[i]))
                y = np.array([onehot(5, rawsentiment[i])]).T
                data.append([x, y])

        print("Finished Loading !")
        return dataset(data[:int(self.ratio*self.length)]), dataset(data[int(self.ratio*self.length):])


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
    check_dataset()

    dataloader = data_loader(path="./train.tsv", model="BOW", length=20, classes=5, ratio= 0.8)
    
    train_dataset, test_dataset = dataloader.load()

    test_dataloader(train_dataset);test_dataloader(test_dataset)


