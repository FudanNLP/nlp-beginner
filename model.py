import numpy as np 
import copy
import collections

h = 1e-5

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
        self.debug = False

    def softmax(self, x):
        z = np.exp(x - np.max(x,axis=0))
        return z / (np.sum(z))

    def cross_entroy(self, y):
        #print(y, self.y_)
        return np.sum(np.sum(-1 * y * (np.log2(self.y_ + h)), axis=0))
    
    def forward(self, x):
        self.x = x
        
        if self.debug:    
            print(self.w.T.dot(x))
            print(self.softmax(self.w.T.dot(x)))
            print(self.softmax(self.w.T.dot(x)).argmax(axis=0))
        
        self.y_ = self.softmax(self.w.T.dot(x))  

    def criterion(self, y):
        self.loss += self.cross_entroy(y)
    
    def backward(self, y):
        self.grad += self.x.dot((y - self.y_).T)
        
        if self.debug:
            print("x", self.x)
            print("y - y_", (y - self.y_).T)
            print("grad", self.x.dot((y - self.y_).T))
    
    def step(self, bs):
        if self.debug:
            print("step:", self.lr * self.grad / bs)
            print("before:", self.w)
        
        self.w += self.lr * self.grad / bs
        
        if self.debug:
            print("after", self.w)
        
        self.loss = 0 
        self.grad = 0

    def predict(self, x):
        return self.forward(x).argmax(axis=0)
    
    def fit(self, batch, debug=False):
        assert type(batch) == list, "Input shall be batch (<class 'list'>)"

        if debug:
            self.debug = True

        x, y = batch
        #print(x.shape, y.shape)
        self.forward(x)
        self.criterion(y)
        self.backward(y)
        
        loss = self.loss

        N = x.shape[-1]

        self.step(N)
        
        true, predict =  y.argmax(axis=0), self.y_.argmax(axis=0)
        
        acc = collections.Counter(true - predict)[0] / N
        
        print("true:", true, "predict", predict, "acc:",acc)
        
        return loss / N, self.w


   

