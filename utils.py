import os
import pandas as pd
import numpy as np 
import copy


def onehot(length, y):
    tmp = np.zeros(length)
    tmp[y-1] = 1
    return tmp


def process(text):
    text = text.lower()
    text = text.replace(',', ' ')
    text = text.replace('/', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('.', ' ')
    return text.split() 


def generate_BOW(data):
    book = {}
    for sentence in data:
        for word in process(sentence):
            if not word in book:
                book[word] = 0

    np.save("./phrase_BOW.npy", book)
    return book

def BOWTransform(data, book):
    x = []
    for sentence in data:
        vector = copy.deepcopy(book)
        for word in process(sentence):
            vector[word] += 1
        x.append(list(vector.values()))
    return x

def load_BOW(path="./phrase_BOW.npy"):
    return np.load(path, allow_pickle=True).item()

def generate_Ngram(path=["./test.tsv", "./train.tsv"], n=2):
    if os.path.exists("./phrase_ngram.npy"):
        print("phrase_ngram.npy already exists!")
        return 
    
    data = pd.concat([pd.read_csv(p, sep='\t') for p in path])
    book = {}

    for sentence in data["Phrase"]:
        tmp = process(sentence)
        for i in range(len(tmp) - n):
            if not tuple(tmp[i:i+n]) in book:
                book[tuple(tmp)] = 0

    np.save("./phrase_Ngram.npy", book)
    print("Done !")


def load_Ngram(path="./phrase_Ngram.npy"):
    return np.load(path, allow_pickle=True).item()



if __name__ == "__main__":
    generate_BOW()
    generate_Ngram()

    rawdata = load_BOW()
    
    print(len(rawdata))

    print(len(load_Ngram()))
