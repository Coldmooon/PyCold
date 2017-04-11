# Created by coldmooon
import lmdb
import numpy as np
import caffe
import torch
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua


def load_t7file():

    dataset = load_lua('/media/coldmoon/ExtremePro960G/Datasets/MNIST/mnist-rot-12k/mnist-rot-12k.t7')
    training_data = dataset['train']['data'].numpy()
    training_labels = dataset['train']['labels'].numpy() - 1
    training_labels = training_labels[:,0]
    # data = np.lib.pad(data, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    # data = data.transpose(0,1,3,2)

    test_data = dataset['val']['data'].numpy()
    test_labels = dataset['val']['labels'].numpy() - 1

    dataset = {'train': [training_data, training_labels], 'test': [test_data, test_labels]}

    return dataset

def numpy2lmdb(dataset=None, save_path='./'):

    if dataset is None:
        dataset = load_t7file()

    sets = ['train', 'test']

    for set in sets:
        print('Processing...', set)
        data = dataset[set][0]
        print('data shape: ', data.shape)
        labels = dataset[set][1]

        X = np.zeros((data.shape), dtype=np.uint8)
        y = np.zeros(data.shape[0], dtype=np.int64)

        map_size = X.nbytes * 10
        env = lmdb.open(save_path + set, map_size=1e12)
        # env = lmdb.open(cifar_caffe_directory, map_size=50000 * 1000 * 5)
        txn = env.begin(write=True)

        count = 0
        for i in range(data.shape[0]):
            datum = caffe.io.array_to_datum(data[i], int(labels[i]))
            str_id = '{:08}'.format(count)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            count += 1
            if count % 1000 == 0:
                print('already handled with {} pictures'.format(count))
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
        env.close()

        # with env.begin(write=True) as txn:
        #     # txn is a Transaction object
        #     for i in range(data.shape[0]):
        #         if i % 1000 == 0:
        #             print('processed: ',i)
        #         datum = caffe.proto.caffe_pb2.Datum()
        #         datum.channels = data.shape[1]
        #         datum.height = data.shape[2]
        #         datum.width = data.shape[3]
        #
        #         # print('label: ', int(labels[i] - 1))
        #         # plt.figure(figsize=(7, 7))
        #         # plt.imshow(data[i].reshape(32, 32), cmap='gray', interpolation='none')
        #         # plt.title('Cluttered MNIST', fontsize=20)
        #         # plt.axis('off')
        #         # plt.show()
        #
        #         datum.data = data[i].tobytes()  # or .tostring() if numpy < 1.9
        #         datum.label = int(int(labels[i]) - 1)
        #         str_id = '{:08}'.format(i)
        #
        #         # The encode is only essential in Python 3
        #         txn.put(str_id.encode('ascii'), datum.SerializeToString())
        #
        # env.close();