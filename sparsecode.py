import torch
from torchvision.datasets import MNIST
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
from sklearn.decomposition import MiniBatchDictionaryLearning,SparseCoder
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F

# 滑窗采样函数
def sliding_window_sampling(image_batch, window_size=4, stride=2):
    batch_size, channel, height, width = image_batch.shape

    blocks = image_batch.unfold(2, window_size, stride).unfold(3, window_size, stride)

    blocks = blocks.contiguous()
    shape = blocks.shape
    blocks = blocks.reshape(batch_size, channel*shape[2] * shape[3], *shape[4:])
    return blocks


class sparse_code():
    def __init__(self, n_components):
        self.n_components = n_components
        self.dictionary = None
        self.is_trained = False  # 新增训练状态标志

    def forward(self, image, train=False):
        X = image.reshape(-1, image.size(-2) * image.size(-1)).numpy()

        if train and not self.is_trained:
            dict_learner = MiniBatchDictionaryLearning(n_components=self.n_components,
                batch_size=10,
                dict_init=self.dictionary
            )
            self.dictionary = dict_learner.fit(X).components_
            self.is_trained = True

        coder = SparseCoder(
            dictionary=self.dictionary,
            transform_algorithm='omp',
            #transform_algorithm='lasso_lars',
            transform_alpha=0.01,)
            #transform_max_iter=100)
        return coder.transform(X), self.dictionary


def sparse_coding(images, sc1, sc2, sc3, train_sc1=False, train_sc2=False, train_sc3=False):

    sample_blocks = sliding_window_sampling(images)
    image_transformed1, _ = sc1.forward(sample_blocks, train=train_sc1)
    image_reconstructed1 = image_transformed1.reshape(images.size(0), 20, 13, 13)
    image_reconstructed1 = torch.Tensor(image_reconstructed1)


    sample_blocks2 = sliding_window_sampling(image_reconstructed1)
    image_transformed2, _ = sc2.forward(sample_blocks2, train=train_sc2)
    image_reconstructed2 = image_transformed2.reshape(images.size(0), 400, 5, 5)
    image_reconstructed2 = torch.Tensor(image_reconstructed2)


    image_transformed3, _ = sc3.forward(image_reconstructed2, train=train_sc3)# 第三次稀疏编码
    image_transformed3 = image_transformed3.reshape(images.shape[0], 400, 49)
    image_activated = F.relu(torch.Tensor(image_transformed3))
    image_zero = (image_activated > 1e-6)

    return image_zero

def main(train_epochs, encode_test):

    sc1 = sparse_code(20)
    sc2 = sparse_code(20)
    sc3 = sparse_code(49)


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128,
        shuffle=False
    )


    print(" 1")
    for epoch in range(train_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            sample_blocks = sliding_window_sampling(images)
            image_transformed1, _ = sc1.forward(sample_blocks, train=True)
            #_ = sparse_coding(images, sc1, sc2, sc3,
                              #train_sc1=True, train_sc2=False, train_sc3=False)


    print(" 2")
    for epoch in range(train_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            sample_blocks = sliding_window_sampling(images)
            image_transformed1, _ = sc1.forward(sample_blocks, train=False)
            image_reconstructed1 = image_transformed1.reshape(images.size(0), 20, 13, 13)
            image_reconstructed1 = torch.Tensor(image_reconstructed1)

            sample_blocks2 = sliding_window_sampling(image_reconstructed1)
            image_transformed2, _ = sc2.forward(sample_blocks2, train=True)
            #_ = sparse_coding(images, sc1, sc2, sc3,
             #                 train_sc1=False, train_sc2=True, train_sc3=False)


    print(" 3")
    for epoch in range(train_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            sample_blocks = sliding_window_sampling(images)
            image_transformed1, _ = sc1.forward(sample_blocks, train=False)
            image_reconstructed1 = image_transformed1.reshape(images.size(0), 20, 13, 13)
            image_reconstructed1 = torch.Tensor(image_reconstructed1)

            sample_blocks2 = sliding_window_sampling(image_reconstructed1)
            image_transformed2, _ = sc2.forward(sample_blocks2, train=False)
            image_reconstructed2 = image_transformed2.reshape(images.size(0), 400, 5, 5)
            image_reconstructed2 = torch.Tensor(image_reconstructed2)

            image_transformed3, _ = sc3.forward(image_reconstructed2, train=True)

            #_ = sparse_coding(images, sc1, sc2, sc3,
                             # train_sc1=False, train_sc2=False, train_sc3=True)



    all_spikes, all_labels = [], []
    for batch_idx, (images, labels) in enumerate(train_loader):
        spikes = sparse_coding(images, sc1, sc2, sc3, train_sc1=False, train_sc2=False, train_sc3=False)
        all_spikes.append(spikes)
        all_labels.append(labels)


    torch.save({'spikes': torch.cat(all_spikes), 'labels': torch.cat(all_labels)},
               './train_spikes.pt')

    if encode_test:

        all_spikes_test, all_labels_test = [], []
        for batch_idx, (images, labels) in enumerate(test_loader):
            spikes = sparse_coding(images, sc1, sc2, sc3, train_sc1=False, train_sc2=False, train_sc3=False)
            all_spikes_test.append(spikes)
            all_labels_test.append(labels)


        torch.save({'spikes': torch.cat(all_spikes_test), 'labels': torch.cat(all_labels_test)},
                   './test_spikes.pt')


# 数据集类
class MNISTSpikeDataset(Dataset):
    def __init__(self, file_path):
        data = torch.load(file_path)
        self.spikes = torch.cat((data['spikes'],))
        self.labels = torch.cat((data['labels'],))

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, idx):
        spikes = self.spikes[idx]
        labels = self.labels[idx]
        return spikes, labels



def get_mnist(data_path, batch_size):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    train_path = data_path + '/train_spikes.pt'
    test_path = data_path + '/test_spikes.pt'
    trainset = MNISTSpikeDataset(train_path)
    testset = MNISTSpikeDataset(test_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainloader, testloader


if __name__ == "__main__":
    with torch.no_grad():
        main(train_epochs=9, encode_test=True)
