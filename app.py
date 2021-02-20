import numpy as np
import utils
from model import Softmax
import dataloader
import matplotlib.pyplot as plt
import pickle


def test_softmax():
    print("Softmax Test Starts!")

    model = Softmax(19422, 5, 0.01)

    batch = [np.random.random((19422, 3)), np.array([np.array([0,0,0,1,0])]*3).reshape(5,3)]
    
    for i in range(20):
        loss, _ = model.fit(batch)
        print(i, loss)
    assert  loss < 10, "Softmax Test Failed!"
    print("Softmax Test Pass!")
        

def test_datalaoder():
    print("Dataloader Test Starts!")
    
    dataset = dataloader.data_loader()
    count = 0
    for i in dataset:
        count += 1

    batchsize = 128

    for i in range(dataset.length // batchsize + 1):
        dataset.get_shuffle_batch(batchsize)

    print("Dataloader Test pass !")


def train(path="softmax_model.pickle"):
    epoch = 10
    
    model = Softmax(19422,5,lr = 0.001)
    
    dataset = dataloader.data_loader(m=1000)
    
    batchsize = 1000

    per_batch_loss = []

    for i in range(epoch):
        
        for j in range((dataset.length-1) // batchsize +1):
            batch = dataset.get_shuffle_batch(batch_size=batchsize)
            loss, _ = model.fit(batch)
            print("epoch:"+ str(i), " step:" + str(j*batchsize % dataset.length) + ":" + str((j+1)*batchsize % dataset.length),loss)
            per_batch_loss.append(loss.flatten())

    print("Train Finished!")
    plt.plot(per_batch_loss)
    plt.savefig("loss.jpg")
    plt.show()
    np.savetxt("loss.txt", per_batch_loss)
    pickle.dump(model, open(path, "wb"))


def test(path="softmax_model.pickle", ):
    model = pickle.load(open(path, "rb")) 


if __name__ == "__main__":

    utils.generate_BOW()

    rawdata = utils.load_BOW()

    print(len(rawdata))

    test_softmax()

    #test_datalaoder()

    train()

    test()
