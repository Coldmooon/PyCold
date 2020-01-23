import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
# from datasets.LMDB.caffedb import *
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

canvas = [42, 42]

train_data = train_dataset.train_data.numpy()
train_data_transformed = np.zeros((train_data.shape[0], canvas[0], canvas[1]), dtype=np.uint8)
train_data_center_only = np.zeros((train_data.shape[0], canvas[0], canvas[1]), dtype=np.uint8)
train_data_rand_placed = np.zeros((train_data.shape[0], canvas[0], canvas[1]), dtype=np.uint8)
# randomly transform the training data
print('Transforming the training data ...')
for i in range(train_data.shape[0]):

    img = train_data[i]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Original MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi
    # scale = np.random.uniform(0.7, 1.2)
    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(0.83, 1.42) # corresponding to STN (1/0.7, 1/1.2)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[scale, 0,  0],
                  [0, scale,  0],
                  [0,   0,    1]])

    T = (np.dot(S, R))[0:2,:]

    im = affine_transformer(img[None, :,:, None], T)
    # train_data[i,:,:] = rotate(train_data[i], rotation, reshape=False)
    #
    # plt.figure(figsize=(7, 7))
    # plt.imshow(im[0,:,:,0].reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Transformed MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    placed_h_random = np.random.randint(-1 + 0.1, canvas[0] - 28 + 1)
    placed_w_random = np.random.randint(-1 + 0.1, canvas[1] - 28 + 1)
    placed_h_center = int((canvas[0] - 28) / 2)
    placed_w_center = int((canvas[1] - 28) / 2)

    train_data_transformed[i, placed_h_random:placed_h_random + 28, placed_w_random:placed_w_random + 28] = im[0,:,:,0]
    train_data_center_only[i, placed_h_center:placed_h_center + 28, placed_w_center:placed_w_center + 28] = img[:,:]
    train_data_rand_placed[i, placed_h_random:placed_h_random + 28, placed_w_random:placed_w_random + 28] = img[:,:]
    #
    # plt.figure(figsize=(7, 7))
    # plt.imshow(train_data_transformed[i].reshape(canvas[0], canvas[1]), cmap='gray', interpolation='none')
    # plt.title('center only', fontsize=20)
    # plt.axis('off')
    # plt.show()

train_data_transformed = np.expand_dims(train_data_transformed, axis=1)
train_data_center_only = np.expand_dims(train_data_center_only, axis=1)
train_data_rand_placed = np.expand_dims(train_data_rand_placed, axis=1)
train_labels = train_dataset.train_labels.numpy()

print('Transforming the test data ...')
test_data = test_dataset.test_data.numpy()
test_data_transformed = np.zeros((test_data.shape[0], canvas[0], canvas[1]), dtype=np.uint8)
for i in range(test_data.shape[0]):

    img = test_data[i]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Original MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi

    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(0.83, 1.42) # corresponding to STN (1/0.7, 1/1.2)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[scale, 0,  0],
                  [0, scale,  0],
                  [0,   0,    1]])

    T = (np.dot(S, R))[0:2,:]
    im = affine_transformer(img[None, :,:, None], T)

    # test_data[i,:,:] = rotate(test_data[i], rotation, reshape=False)
    #
    # plt.figure(figsize=(7, 7))
    # plt.imshow(im.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Transformed MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    placed_h = np.random.randint(-1 + 0.1, canvas[0] - 28 + 1)
    placed_w = np.random.randint(-1 + 0.1, canvas[1] - 28 + 1)

    test_data_transformed[i,placed_h:placed_h + 28, placed_w:placed_w + 28] = im[0,:,:,0]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(test_data_transformed[i].reshape(canvas[0], canvas[1]), cmap='gray', interpolation='none')
    # plt.title('Placed MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()


test_data_transformed = np.expand_dims(test_data_transformed, axis=1)
test_labels = test_dataset.test_labels.numpy()

print('train_data_transformed shape:', train_data_transformed.shape)
print('train_data_center_only shape:', train_data_center_only.shape)
print('train_data_rand_placed shape:', train_data_rand_placed.shape)
print('test_data_transformed shape: ', test_data_transformed.shape)


# dataset = {'train': [train_data_transformed, train_labels], 'test': [test_data_transformed, test_labels]}
# for caffe
# numpy2lmdb([train_data_transformed, train_labels], 'train_transformed')
# numpy2lmdb([train_data_center_only, train_labels], 'train_data_center_only')
# numpy2lmdb([train_data_rand_placed, train_labels], 'train_rand_placed')
# numpy2lmdb([test_data_transformed, test_labels], 'test_transformed')

# for torch
np.save('train_transformed.npy', train_data_transformed)
np.save('train_center_only.npy', train_data_center_only)
np.save('train_rand_placed.npy', train_data_rand_placed)
np.save('train_labels.npy', train_labels)
np.save('test_data.npy', test_data_transformed)
np.save('test_labels.npy', test_labels)
