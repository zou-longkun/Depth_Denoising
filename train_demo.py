import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.dataloader import Dataset
from utils import add_noise
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Model_demo import Model
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_data = datasets.MNIST('./data/mnist_data', train=True, download=True)
test_data = datasets.MNIST('./data/mnist_data', train=False)

noises = ["gaussian", "speckle"]
noise_ct = 0
noise_id = 0
traindata = np.zeros((60000, 28, 28))
xtrain = np.zeros((60000, 28, 28))
ytrain = np.zeros(60000)
for idx in tqdm(range(len(train_data))):
    xtrain[idx] = train_data[idx][0]
    ytrain[idx] = train_data[idx][1]
    if noise_ct < (len(train_data) / 2):
        noise_ct += 1
        traindata[idx] = add_noise(train_data[idx][0], noise_type=noises[noise_id])
    else:
        print("\n{} noise addition completed to training images".format(noises[noise_id]))
        noise_id += 1
        noise_ct = 0
print("{} noise addition completed to training images".format(noises[noise_id]))

noise_ct = 0
noise_id = 0
testdata = np.zeros((10000, 28, 28))
xtest = np.zeros((10000, 28, 28))
ytest = np.zeros(10000)
for idx in tqdm(range(len(test_data))):
    xtest[idx] = test_data[idx][0]
    ytest[idx] = test_data[idx][1]
    if noise_ct < (len(test_data) / 2):
        noise_ct += 1
        x = add_noise(test_data[idx][0], noise_type=noises[noise_id])
        testdata[idx] = x
    else:
        print("\n{} noise addition completed to test images".format(noises[noise_id]))
        noise_id += 1
        noise_ct = 0
print("{} noise addition completed to test images".format(noises[noise_id]))

tsfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = Dataset(traindata, xtrain, ytrain, tsfms)
testset = Dataset(testdata, xtest, ytest, tsfms)

batch_size = test_batch_size = 32
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, **kwargs)
testloader = DataLoader(testset, batch_size=1, shuffle=True, **kwargs)

model = Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
epochs = 10
scheduler = CosineAnnealingLR(optimizer, epochs)
lenth = len(trainloader)
losslist = list()
epochloss = 0
running_loss = 0
for epoch in range(epochs):
    model.train()
    print("Entering Epoch: ", epoch)
    for dirty, clean, label in tqdm((trainloader)):
        dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor).to(device)
        clean = clean.view(clean.size(0), -1).type(torch.FloatTensor).to(device)

        # -----------------Forward Pass----------------------
        output = model(dirty)
        loss = criterion(output, clean)
        # -----------------Backward Pass---------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epochloss += loss.item()
    # -----------------Log-------------------------------
    losslist.append(running_loss / lenth)
    running_loss = 0
    scheduler.step()
    print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))

plt.plot(range(len(losslist)), losslist)
plt.show()
torch.save(model.state_dict(), './model.pt')
