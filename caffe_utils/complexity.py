import caffe
import numpy as np

def params_network(prototxt_file):
    net = caffe.Net(prototxt_file, caffe.TEST)
    print('Number of parameters: ', np.sum([np.prod(v[0].data.shape) for k, v in (net.params.items())]))


def params_solver(caffe_solver):
    print('Number of parameters:')
    print(np.sum([np.prod(v[0].data.shape) for k, v in caffe_solver.net.params.items() if 'conv' in k if 'bn' not in k if 'shared' not in k]))
    print('shared parameters:')
    for k,v in caffe_solver.net.params.items():
        if 'shared' in  k:
            print(np.prod(v[0].data.shape))
            break