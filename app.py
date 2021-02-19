import numpy as np
import utils
from model import Softmax
import dataloader


def test_softmax():
    model = Softmax(19422)

    test = [np.random.random((19422, 1)), np.array([0,0,0,0,1]).reshape(5, 1)]

    model.fit([test])

    print("Softmax Test pass !")


def test_datalaoder():
    dataset = dataloader.data_loader()
    count = 0
    for i in dataset:
        if count > 5 :
            break
        count += 1

    print("Dataloader Test pass !")

def train():
    epoch = 1
    
    model = Softmax(19422)
    
    dataset = dataloader.data_loader()
    
    for i in range(epoch):
        
        for j in range(100):
            batch = dataset.get_shuffle_batch(64)
            loss, w = model.fit(batch)
            print(i, j,loss) 


if __name__ == "__main__":

    utils.generate_BOW()
    #generate_ngrams()

    rawdata = utils.load_BOW()

    print(len(rawdata))

    test_softmax()

    test_datalaoder()

    train()
