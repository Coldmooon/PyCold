import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from datasets.LMDB.caffedb import *
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import warp
from affine_transformer.numpy.interpolation import *


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transforms.ToTensor())

canvas = [28, 28]

train_data = train_dataset.train_data.numpy()
train_data_transformed = np.zeros((train_data.shape[0], canvas[0], canvas[1]))
# randomly transform the training data
print('Transforming the training data ...')
for i in range(train_data.shape[0]):

    im = train_data[i]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(im.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Cluttered MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi
    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(10/3, 1)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[scale, 0,  0],
                  [0, scale,  0],
                  [0,   0,    1]])

    T = (np.dot(S, R))[0:2,:]

    im = affine_transformer(im[None, :,:, None], S)

    # train_data[i,:,:] = rotate(train_data[i], rotation, reshape=False)

    # plt.figure(figsize=(7, 7))
    # plt.imshow(im[0,:,:,0].reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Cluttered MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()


    placed_h = np.random.randint(0 - 0.1, canvas[0] - 28)
    placed_w = np.random.randint(0 - 0.1, canvas[1] - 28)

    train_data_transformed[i,placed_h:placed_h+28, placed_w:placed_w+28] = im[0,:,:,0]
    # print('RTS shape: ', train_data_transformed[i].shape)
    #
    # plt.figure(figsize=(7, 7))
    # plt.imshow(train_data_transformed[i].reshape(canvas[0], canvas[1]), cmap='gray', interpolation='none')
    # plt.title('Cluttered MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()


train_data_transformed = np.expand_dims(train_data_transformed, axis=1)
train_labels = train_dataset.train_labels.numpy()

test_data = test_dataset.test_data.numpy()
test_data_transformed = np.zeros((test_data.shape[0], canvas[0], canvas[1]))

print('Transforming the test data ...')

for i in range(test_data.shape[0]):

    im = test_data[i]

    plt.figure(figsize=(7, 7))
    plt.imshow(im.reshape(28, 28), cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST', fontsize=20)
    plt.axis('off')
    plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi
    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(10/3, 1)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[scale, 0,  0],
                  [0, scale,  0],
                  [0,   0,    1]])

    T = (np.dot(S, R))[0:2,:]
    im = affine_transformer(im[None, :,:, None], S)

    # test_data[i,:,:] = rotate(test_data[i], rotation, reshape=False)

    plt.figure(figsize=(7, 7))
    plt.imshow(im.reshape(28, 28), cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST', fontsize=20)
    plt.axis('off')
    plt.show()

    placed_h = np.random.randint(0 - 0.1, canvas[0] - 28)
    placed_w = np.random.randint(0 - 0.1, canvas[1] - 28)
    #
    test_data_transformed[i,placed_h:placed_h+28, placed_w:placed_w+28] = im[0,:,:,0]
    # print('RTS shape: ', test_data_transformed[i].shape)

    plt.figure(figsize=(7, 7))
    plt.imshow(test_data_transformed[i].reshape(canvas[0], canvas[1]), cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST', fontsize=20)
    plt.axis('off')
    plt.show()


test_data_transformed = np.expand_dims(test_data_transformed, axis=1)
test_labels = test_dataset.test_labels.numpy()

print('training data:', train_data_transformed.shape)
print('test data: ', test_data_transformed.shape)

dataset = {'train': [train_data_transformed, train_labels], 'test': [test_data_transformed, test_labels]}

numpy2lmdb(dataset)
