import numpy as np 
import copy
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

h = 1e-5

class Softmax_torch(nn.Module):
    def __init__(self, num_of_inputs, num_of_outputs):
        super(Softmax_torch, self).__init__()
        self.fc = nn.Linear(num_of_inputs, num_of_outputs)

    def forward(self, x, y):
        x = self.fc(x)
        return x

    def predict(self, x):
        return torch.argmax(F.softmax((self.fc(x))), dim=1)


class Softmax():
    def __init__(self, num_of_input=19422, num_of_output=5, lr=1):
        """
        x:
            (19422, 1)
        y:
            (5, 1)
        """
        self.num_of_class = None
        self.w = np.random.randn(num_of_input, num_of_output)
        self.lr = lr
        self.x = None
        self.y_ = None
        self.grad = 0
        self.loss = 0

    def softmax(self, x):
        z = np.exp(x - np.max(x,axis=0))
        return z / (np.sum(z))

    def cross_entroy(self, y):
        return np.sum(np.sum(-1 * y * (np.log2(self.y_ + h)), axis=0))
    
    def forward(self, x):
        self.x = x
        self.y_ = self.softmax(self.w.T.dot(x))  
        return self.y_

    def criterion(self, y):
        self.loss += self.cross_entroy(y)
    
    def backward(self, y):
        self.grad += self.x.dot((y - self.y_).T)
    
    def step(self, bs):
        self.w += self.lr * self.grad / bs
        self.loss = 0 
        self.grad = 0

    def predict(self, x):
        return self.forward(x)
    
    def fit(self, x, y, debug=False):
        
        self.forward(x)
        self.criterion(y)
        self.backward(y)
        
        loss = self.loss

        N = x.shape[-1]

        self.step(N)
        
        true, predict =  y.argmax(axis=0), self.y_.argmax(axis=0)
        
        acc = collections.Counter(true - predict)[0] / N
        
        #print("true:", true, "predict", predict, "acc:",acc)
        print("acc: ", acc)

        return loss / N


if __name__ == "__main__":
    model = Softmax_torch(19422, 5)

    inputs = torch.randn(16, 19422)

    labels = torch.zeros(16)

    for i in range(labels.shape[0]):
        labels[i] = i % 3

    print(labels)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

    criterion = nn.CrossEntropyLoss()

    for i in range(18):
        y_ = model(inputs, labels.long())
        
        loss = criterion(y_, labels.long())

        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()
        
        print(np.float(loss))
        print(model.predict(inputs))


    print("Done!")

