import torch
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
device = torch.device("cuda" if use_cuda else "cpu")
test_batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
test_data = datasets.MNIST('./data/mnist_data', train=False)

noises = ["gaussian", "speckle"]
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
testset = Dataset(testdata, xtest, ytest, tsfms)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

model = Model()
model.load_state_dict(torch.load('./model.pt'))
model = model.to(device)

f, axes = plt.subplots(6, 3, figsize=(10, 10))
axes[0, 0].set_title("Original Image")
axes[0, 1].set_title("Dirty Image")
axes[0, 2].set_title("Cleaned Image")

test_imgs = np.random.randint(0, 10000, size=6)
with torch.no_grad():
    model.eval()
    for idx in range((6)):
        dirty = testset[test_imgs[idx]][0]
        clean = testset[test_imgs[idx]][1]
        label = testset[test_imgs[idx]][2]
        dirty = dirty.view(dirty.size(0), -1).type(torch.FloatTensor).to(device)
        output = model(dirty)

        output = output.view(1, 28, 28)
        output = output.permute(1, 2, 0).squeeze(2)
        output = output.detach().cpu().numpy()

        dirty = dirty.view(1, 28, 28)
        dirty = dirty.permute(1, 2, 0).squeeze(2)
        dirty = dirty.detach().cpu().numpy()

        clean = clean.permute(1, 2, 0).squeeze(2)
        clean = clean.detach().cpu().numpy()

        axes[idx, 0].imshow(clean, cmap="gray")
        axes[idx, 1].imshow(dirty, cmap="gray")
        axes[idx, 2].imshow(output, cmap="gray")
    plt.show()

