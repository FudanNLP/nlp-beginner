import numpy as np
import utils
import model
import dataloader as loader
import matplotlib.pyplot as plt
import pickle
import collections


def test_softmax(use_torch=False):
    print("Softmax Test Starts!")

    net = model.Softmax_torch(19422, 5) if use_torch else model.Softmax(19422, 5, 0.01)

    x, y = [np.random.random((19422, 3)), np.array([np.array([0,0,0,1,0])]*3).reshape(5,3)]
    
    for i in range(20):       
        loss = net.fit(x, y)
        print(i, loss)
    assert  loss < 10, "Softmax Test Failed!"
    print("Softmax Test Pass!")
        

def test(net, dataset):
    x, y = dataset.get_batch(batch_size=dataset.length)
    y_ = net.predict(x)
    total_acc = collections.Counter(list(y.argmax(axis=0) - y_.argmax(axis=0)))[0] / dataset.length
    print("Total_acc: ", total_acc)


def train(path="softmax_model.pickle", dataset=None, use_torch=False, epoch=20, batchsize=64, lr=0.001):
    
    net = model.Softmax(dataset.num_of_inputs, 5)

    dataset = dataset

    per_batch_loss = []

    for i in range(epoch):
        for j in range((dataset.length+1) // batchsize ):            
            x, y = dataset.get_batch(batch_size=batchsize)

            outputs = net.fit(x, y)
            
            y_ = net.predict(x)
  
            loss = outputs
            per_batch_loss.append(loss)
            if j == dataset.length // batchsize - 1:
                print("epoch:"+str(i), " step:"+str(j*batchsize%dataset.length)+":"+str((j+1)*batchsize%dataset.length), "loss: ", loss)
        
    test(net=net, dataset=dataset)

    print("Train Finished!")
    plt.plot(per_batch_loss)
    plt.title("Loss")
    plt.savefig("loss.jpg")
    plt.show()
    np.savetxt("loss.txt", per_batch_loss)
    pickle.dump(net, open(path, "wb"))


def load_softmax(path="softmax_model.pickle", ):
    return pickle.load(open(path, "rb")) 


if __name__ == "__main__":

    test_softmax()

    m = 2000

    dataloader = loader.data_loader(path="./train.tsv", model="BOW", length=m, classes = 5, ratio = 0.8)    
    train_dataset, test_dataset = dataloader.load()
    pickle.dump(train_dataset, open("./train_dataset.pickle", "wb"))
    pickle.dump(test_dataset, open("./test_dataset.pickle", "wb"))

    #train_dataset, test_dataset = pickle.load(open("./train_dataset.pickle", "rb")), pickle.load(open("./test_dataset.pickle", "rb"))

    train(path="softmax_model.pickle", dataset=train_dataset,  epoch=50, batchsize=200, lr = 0.001)

    test(load_softmax("softmax_model.pickle"),  dataset=test_dataset)



