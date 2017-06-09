from powerlaw import *
import os
os.getcwd()
from pylab import *
import time
import caffe
import shutil


scales = np.array([1.0, 0.875, 0.75, 0.625, 1.375, 1.25, 1.125])
channel_mean = np.array([-1.84252e-08, -8.76298e-09, -2.27323e-08])

for k in [1]:
    path = './data/'
    foldername = 'mstnet_elu_gauss_cifar10aug4pad_' + time.strftime(
        "%Y%m%d_%H_%M") + '_halfpyramidv2_pixelscale_decay0_conv_concat_ave_fc_deploy'
    destination = path + foldername

    if not os.path.exists(destination):
        os.makedirs(destination)

    filename = destination + '/' + foldername + '.txt'
    f = open(filename, 'w')
    lambda_net_proto = path + 'mstnet_elu_gauss_halfpyramidv2_pixelwise_scale_decay0_conv_concat_ave_fc_deploy.prototxt'
    model_file = destination + '/' + 'computing_lambda.caffemodel'

    solver = None
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(path + 'solver_elu.prototxt')
    for it in range(120001):
        # computing lambda
        if (it == 0) and (it % 1000 == 0):
            print('\nIteration', it, ', computing lambda')
            solver.net.save(model_file)
            solver.test_nets[1].share_with(solver.net)
            power_lambda = computing_lambda(model_file, solver.test_nets[1], lambda_net_proto, scales, channel_mean)
            print('lambda computed is:', power_lambda)
            factors = pow(scales, -power_lambda)
            print('New factors: \n', factors)
            for scale in scales[1:]:
                position = np.where(scales[:] == scale)[0][0]
                # print('now assign ', scales[position],':',factors[position], ' to ', 'layer: scale-' + str(scale)+':', \
                #                                                    solver.net.params['scale-' + str(scale)][0].data[0])
                solver.net.params['scale-' + str(scale)][0].data[:] = factors[position]
            print('lambda initialized..')
        solver.step(1)

        if it % 100 == 0:
            training_loss = solver.net.blobs['loss'].data
            training_acc = solver.net.blobs['accuracy'].data
            f.write('\niteration:' + str(it) + '\n')
            f.write('   train loss: ' + str(training_loss))
            f.write('   train acc : ' + str(training_acc))
        if (it >= 100000) and (it % 1000 == 0):
            solver.net.save(destination + '/' + 'mstnet_elu_gauss_cifar10aug4pad_' + str(it) + '.caffemodel')
    print("done!")
    source = os.listdir(path)
    for files in source:
        if (files.endswith(".caffemodel") or files.endswith(".solverstate")) and ('gpu0' in files):
            print('moving ', files, 'to', destination)
            shutil.move(path + files, destination)
    f.close()