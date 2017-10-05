# Usefulness: This file helps me to get and inspect the layer parameters from a particular layer of CNN saved as caffemodel

import numpy as np
import caffe

model = '/raid5/hasnat/caffe/Deploy_Resnet_vMFML_usk.prototxt'
weights = '/raid5/hasnat/2016/CMs/CM_Net_vMFML_usk_16_var3_Casia_iter_70000.caffemodel'

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(model, weights, caffe.TEST)

print('==============================')
print(net.params['kappa_mul'][0].data) # kappa_mul is the layer name
print('==============================')




