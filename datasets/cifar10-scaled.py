import torch
import torchvision
import torchvision.transforms as transforms
from affine_transformer.numpy.interpolation import *
from affine_transformer.numpy.utils import *
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# im_to_show = torchvision.utils.make_grid(images)
# imshow(im_to_show)
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


train_data = trainset.train_data
train_labels = np.array(trainset.train_labels) + 1

canvas = [48, 48]

train_data_transformed = np.zeros((train_data.shape[0], canvas[0], canvas[1], 3), dtype=np.uint8)
train_data_center_only = np.zeros((train_data.shape[0], canvas[0], canvas[1], 3), dtype=np.uint8)
train_data_rand_placed = np.zeros((train_data.shape[0], canvas[0], canvas[1], 3), dtype=np.uint8)
# randomly transform the training data
print('Transforming the training data ...')
for i in range(train_data.shape[0]):

    img = train_data[i]

    # plt.figure(figsize=(2, 2))
    # plt.imshow(img.reshape(32, 32, 3), interpolation='none')
    # plt.title('Original CIFAR10 ', fontsize=20)
    # plt.axis('off')
    # plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi
    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(2/3, 2)
    # scale = np.random.uniform(0.83, 1.42) # corresponding to STN (1/0.7, 1/1.2)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[1, 0,  0],
                  [0, 1,  0],
                  [0, 0,  1]])

    T = (np.dot(S, R))[0:2,:]

    out_shape = np.random.randint(16, 48)

    im = affine_transformer(img[None, :, :, :], S, out_shape, out_shape)
    # train_data[i,:,:] = rotate(train_data[i], rotation, reshape=False)
    # array_to_img(im[0]).show()

    # plt.figure(figsize=(2, 2))
    # plt.imshow(np.uint8(im[0]), interpolation='none')
    # plt.title('Transformed CIFAR10', fontsize=20)
    # plt.axis('off')
    # plt.show()

    placed_h_random = np.random.randint(0 - 0.1, canvas[0] - out_shape)
    placed_w_random = np.random.randint(0 - 0.1, canvas[1] - out_shape)
    placed_h_center = int((canvas[0] - 32) / 2)
    placed_w_center = int((canvas[1] - 32) / 2)

    train_data_transformed[i, placed_h_random:placed_h_random + out_shape, placed_w_random:placed_w_random + out_shape, :] = im[0,:,:,:]
    train_data_center_only[i, placed_h_center:placed_h_center + 32, placed_w_center:placed_w_center + 32, :] = img[:,:, :]
    # train_data_rand_placed[i, placed_h_random:placed_h_random + 32, placed_w_random:placed_w_random + 32, :] = img[:,:, :]

    # plt.figure(figsize=(2, 2))
    # plt.imshow(train_data_transformed[i].reshape(canvas[0], canvas[1], 3), cmap='gray', interpolation='none')
    # plt.title('no scale', fontsize=20)
    # plt.axis('off')
    # plt.show()

# train_data_transformed = np.expand_dims(train_data_transformed, axis=1)
# train_data_center_only = np.expand_dims(train_data_center_only, axis=1)
# train_data_rand_placed = np.expand_dims(train_data_rand_placed, axis=1)

train_data_transformed = np.transpose(train_data_transformed, (0,3,1,2))
train_data_center_only = np.transpose(train_data_center_only, (0,3,1,2))
train_data_rand_placed = np.transpose(train_data_rand_placed, (0,3,1,2))


print('Transforming the test data ...')
test_data = testset.test_data
test_data_transformed = np.zeros((test_data.shape[0], canvas[0], canvas[1], 3), dtype=np.uint8)
for i in range(test_data.shape[0]):

    img = test_data[i]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Original MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    rotation = np.random.uniform(-np.pi/4, np.pi/4) # * 180 / np.pi
    # scale = np.random.uniform(0.7, 1.2)
    scale = np.random.uniform(0.7, 10/3)
    # scale = np.random.uniform(0.83, 1.42) # corresponding to STN (1/0.7, 1/1.2)

    R = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                  [np.sin(rotation),  np.cos(rotation), 0],
                  [      0,                  0,         1]])
    S = np.array([[1, 0,  0],
                  [0, 1,  0],
                  [0,   0,    1]])

    T = (np.dot(S, R))[0:2,:]

    out_shape = np.random.randint(16, 48)
    im = affine_transformer(img[None, :, :, :], S, out_shape, out_shape)

    # test_data[i,:,:] = rotate(test_data[i], rotation, reshape=False)

    # plt.figure(figsize=(7, 7))
    # plt.imshow(im.reshape(28, 28), cmap='gray', interpolation='none')
    # plt.title('Transformed MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()

    placed_h = np.random.randint(0 - 0.1, canvas[0] - out_shape)
    placed_w = np.random.randint(0 - 0.1, canvas[1] - out_shape)

    test_data_transformed[i,placed_h:placed_h + out_shape, placed_w:placed_w + out_shape] = im[0,:,:,:]

    # plt.figure(figsize=(7, 7))
    # plt.imshow(test_data_transformed[i].reshape(canvas[0], canvas[1]), cmap='gray', interpolation='none')
    # plt.title('Placed MNIST', fontsize=20)
    # plt.axis('off')
    # plt.show()


# test_data_transformed = np.expand_dims(test_data_transformed, axis=1)
test_data_transformed = np.transpose(test_data_transformed, (0, 3, 1, 2))
test_labels = np.array(testset.test_labels) + 1

print('train_data_transformed shape:', train_data_transformed.shape)
print('train_data_transformed mean & std:', np.mean(train_data_transformed, axis=(0,2,3)), np.std(train_data_transformed, axis=(0,2,3)))

print('train_data_center_only shape:', train_data_center_only.shape)
print('train_data_center_only mean & std:', np.mean(train_data_center_only, axis=(0,2,3)), np.std(train_data_center_only, axis=(0,2,3)))

# print('train_data_rand_placed shape:', train_data_rand_placed.shape)
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
# np.save('train_rand_placed.npy', train_data_rand_placed)
np.save('train_labels.npy', train_labels)
np.save('test_data.npy', test_data_transformed)
np.save('test_labels.npy', test_labels)