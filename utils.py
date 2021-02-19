import os
import pandas as pd
import numpy as np 


def process(text):
    text = text.lower()
    text = text.replace(',', ' ')
    text = text.replace('/', ' ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = text.replace('.', ' ')
    return text.split() 


def generate_BOW(path=["./test.tsv", "./train.tsv"], type="bag-of-words"):
    if os.path.exists("./phrase_dict.npy"):
        print("phrase_dict.npy already exists!")
        return 
    
    data = pd.concat([pd.read_csv(p, sep='\t') for p in path])
    book = {}

    for sentence in data["Phrase"]:
        for word in process(sentence):
            if not word in book:
                book[word] = 0

    np.save("./phrase_dict.npy", book)
    print("Done !")


def load_BOW(path="./phrase_dict.npy"):
    return np.load(path, allow_pickle=True).item()


def generate_ngrams(words_list, n=2):
    ngrams_list = []
 
    for num in range(0, len(words_list)):
        ngram = ' '.join(words_list[num:num + n])
        ngrams_list.append(ngram)
 
    np.save("./phrase_ngrams.npy", ngrams_list)


def load_Ngrams(path="./phrase_ngrams.npy"):
    return np.load(path).item()


if __name__ == "__main__":
    generate_BOW()
    #generate_ngrams()

    rawdata = load_BOW()

    print(len(rawdata))

