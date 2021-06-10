import torch
import os
import cv2
import random
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, datasetnoised, datasetclean, labels, transform=None):
        self.noise = datasetnoised
        self.clean = datasetclean
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, idx):
        xNoise = self.noise[idx]
        xClean = self.clean[idx]
        y = self.labels[idx]
        if self.transform is not None:
            xNoise = self.transform(xNoise)
            xClean = self.transform(xClean)

        return xNoise, xClean, y


class ImageDataset(Dataset):
    def __init__(self, imagedir, labelfile, classify_num, train=True):
        self.imagedir = imagedir
        self.labelfile = labelfile
        self.classify_num = classify_num
        self.img_list = []
        with open(self.labelfile, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                filepath = os.path.join(self.imagedir, line.split(";")[0].replace('\\', '/'))
                label = line.split(";")[1].strip('\n')
                self.img_list.append((filepath, label))
        if not train:
            self.img_list = random.sample(self.img_list, 50)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        _int_label = int(self.img_list[item][1])
        label = torch.tensor(_int_label, dtype=torch.long)
        img = self.ProcessImgResize(self.img_list[item][0])
        return img, label

    def ProcessImgResize(self, filename):
        _img = cv2.imread(filename)
        _img = _img.transpose((2, 0, 1))
        _img = _img / 255
        _img = torch.from_numpy(_img)
        _img = _img.to(torch.float32)
        return _img
