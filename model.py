import numpy as np 

e = 0.000001

class Softmax():
    def __init__(self, num_of_input=19422, num_of_output=5, lr=0.0001):
        self.num_of_class = None
        self.w = np.random.randn(num_of_input, num_of_output)
        self.lr = lr
        self.x = None
        self.y_ = None
        self.grad = 0
        self.loss = 0

    def softmax(self, x):
        z = np.exp(x - np.max(x,axis=0))
        return z / (np.sum(z) + e)

    def cross_entroy(self, y):
        return -1 * y.T.dot(np.log2(self.y_ + e)) / 5
    
    def forward(self, x):
        self.x = x
        #print(self.w.T.dot(x))
        #print(self.softmax(self.w.T.dot(x)))
        #print(self.softmax(self.w.T.dot(x)).argmax(axis=0))
        self.y_ = self.softmax(self.w.T.dot(x))

    def criterion(self, y):
        self.loss += self.cross_entroy(y)
    
    def backward(self, y):
        self.grad += -1 * self.x.dot((y - self.y_).T)
    
    def step(self, bs):
        self.w += self.lr * self.grad / bs
        self.grad = 0
        self.loss = 0
        self.y_ = None
    

    def fit(self, batch):
        assert type(batch) == list, "Input shall be batch (<class 'list'>)"
        for x, y in batch:
            self.forward(x)
            self.criterion(y)
            self.backward(y)
            #print(self.y_.flatten(), y.flatten())
        tmp = self.loss
        self.step(len(batch))
        return tmp, self.w

