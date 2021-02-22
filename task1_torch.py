
import pandas as pd
import utils
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Softmax_torch(nn.Module):
    def __init__(self, num_of_inputs, num_of_outputs):
        super(Softmax_torch, self).__init__()
        self.fc1= nn.Linear(num_of_inputs, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_of_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        return torch.argmax(F.softmax(self.forward(x) ), dim=1)


def run():
    length = 20000
    ratio = 0.8

    train_length, test_length = int(ratio * length), int((1-ratio)*length)
    
    data = pd.read_csv("./train.tsv", sep="\t")[:length]
    rawx, rawy = data["Phrase"].tolist(), data["Sentiment"].tolist()

    BOW_model = utils.load_BOW()

    x, y = [], rawy
    
    order = [i for i in range(length)]
    
    for sentence in rawx:
        for word in utils.process(sentence):
            BOW_model[word] += 1
        x.append(list(BOW_model.values()))

    shuffle(order)
    x, y = [x[i] for i in order], [y[i] for i in order]

    net = Softmax_torch(19422, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002)

    running_loss = 0.0

    batch_size = 1600

    epochs = 10

    for epoch in range(epochs):
        for i in range((train_length+1) // batch_size):
            inputs, labels = (
                            torch.Tensor(x[i * batch_size : (i+1) * batch_size]),
                            torch.Tensor(y[i * batch_size: (i +1)*batch_size])
                            )

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i == 9:
                print("epoch: ", epoch, "loss: ", running_loss / train_length)
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    from collections import Counter
    print(Counter(y[:train_length]))
    print(Counter(y[train_length:]))
    acc_test = Counter((net.predict(torch.Tensor(x[train_length:])) - torch.Tensor(y[train_length:])).tolist())[0] / test_length
    acc_train = Counter((net.predict(torch.Tensor(x[:train_length])) - torch.Tensor(y[:train_length])).tolist())[0] / train_length
    print("acc in train: ", acc_train)
    print("acc in test: ", acc_test)


if __name__ == "__main__":
    run()

