import scipy.io as sio
import caffe
import numpy as np
def get_lambda(features, scales, remove_negative=0):
    kp = np.array([np.amax(features[:, 0])])
    toselect = np.zeros((1, features.shape[0]), dtype=np.int)
    counterpart = kp[toselect].transpose() / 50
    counterpart = counterpart[:, 0]
    selected = features[:, 0] > counterpart
    features = features[selected, :]
    scales1 = scales[1:]
    nscale = scales1.shape
    O = np.zeros((nscale), dtype=np.int)
    fs_1_end = features[:, 1:]
    fs_0 = features[:, O]
    rs = fs_1_end / fs_0
    rs_mean = np.mean(rs, axis=0)

    print('rs_mean before:', rs_mean)
    marked = rs_mean > 0
    scales1 = scales1[marked]
    print(scales1.shape)
    O_t = np.array([O]).T + 1
    O_t = O_t[marked]
    print(O_t.shape)
    rs_mean = rs_mean[marked]
    print('rs_mean after:', rs_mean)

    # print('rs_mean:',rs_mean)
    scale_log = np.array([-np.log2(scales1)]).T
    left = np.concatenate((O_t, scale_log), axis=1)
    mus_log = np.log2(rs_mean)
    out = np.linalg.lstsq(left, mus_log)[0]
    lamb = out[1]
    return lamb


def computing_lambda(model_file, data_net, feature_net_proto, scales, channel_mean):
    data_net.forward()
    num = data_net.blobs['data'].data.shape[0]
    nScales = scales.shape[0]
    #     for blob in data_net.blobs.keys():
    #         if 'resample' in blob:
    #             print(blob,': ', data_net.blobs[blob].data.shape)
    if len(channel_mean.shape) < 3:
        channel_mean = channel_mean.reshape(3, 1, 1)

    lambda_net = caffe.Net(feature_net_proto, model_file, caffe.TEST)
    features = np.zeros((num, nScales))
    scale_counter = 0;
    sio.savemat('input_data.mat', mdict={'data_for_lambda': data_net.blobs['data'].data[...]})
    for blob in data_net.blobs.keys():
        if 'resample' in blob:
            data_for_lambda = []
            data_for_lambda = data_net.blobs[blob].data[...]
            data_for_lambda = data_for_lambda - channel_mean
            lambda_net.blobs['data'].reshape(*data_for_lambda.shape)
            lambda_net.blobs['data'].data[...] = data_for_lambda
            lambda_net.forward()
            feature = lambda_net.blobs['average'].data.squeeze()
            features[:, scale_counter] = feature
            scale_counter += 1

    sio.savemat('features.mat', mdict={'features': features})
    print('features: ', features.shape)
    lamb = get_lambda(features, scales)
    print('Initialization done!')
    return lamb


# v2 resample using python built-in funciton not caffe resample layer.
def computing_lambda_v2(model_file, data_net, feature_net_proto, scales, channel_mean):
    data_net.forward()
    num = data_net.blobs['data'].data.shape[0]
    input_data = data_net.blobs['data'].data[...]
    sio.savemat('input_data.mat', mdict={'data_for_lambda': data_net.blobs['data'].data[...]})
    nScales = scales.shape[0]

    #     for blob in data_net.blobs.keys():
    #         if 'resample' in blob:
    #             print(blob,': ', data_net.blobs[blob].data.shape)
    if len(channel_mean.shape) < 3:
        channel_mean = channel_mean.reshape(3, 1, 1)

    lambda_net = caffe.Net(feature_net_proto, model_file, caffe.TEST)
    features = np.zeros((num, nScales))

    scale_counter = 0
    for scale in scales:
        if scale == 1.0:
            scaled_data = input_data
            scaled_data -= channel_mean
        else:
            scaled_data = skimage.transform.rescale(input_data, (scale, scale), mode='constant', order=1, clip=False,
                                                    cval=0)
            scaled_data -= channel_mean
        lambda_net.blobs['data'].reshape(*scaled_data.shape)
        lambda_net.blobs['data'].data[...] = scaled_data
        lambda_net.forward()
        feature = lambda_net.blobs['average'].data.squeeze()
        features[:, scale_counter] = feature
        scale_counter += 1

    sio.savemat('features.mat', mdict={'features': features})
    lamb = get_lambda(features, scales)
    return lamb

