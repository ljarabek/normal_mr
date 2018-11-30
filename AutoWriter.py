import torchfile
import tensorflow as tf
from pprint import pprint
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os


def auto_writer(t7name = './tf-AdaIN-master/vgg_normalised.t7', outputpy='output.py', outputmat='weights.mat'):
    np.set_printoptions(threshold=np.nan)
    model = torchfile.load(t7name, force_8bytes_long = True)
    h = 'import scipy.io as sio\n' \
        'import tensorflow as tf\n' \
        'import numpy as np\n' \
        'layers = []\n' \
        'weights=sio.loadmat("{}")\n' \
        'input = tf.placeholder(shape=(1, 512, 512, 3), dtype=tf.float32)\n' \
        'net = tf.identity(input)\n' \
        'with tf.name_scope("{}"):'.format(outputmat,t7name)
    t7name+="/"
    if os.path.isfile(outputmat):
        weights=sio.loadmat(outputmat)
    else:
        weights = {}
    for idx, module in tqdm(enumerate(model.modules)):
        if module._typename == b'nn.SpatialReflectionPadding':
            #print("lol")
            left = module.pad_l
            right = module.pad_r
            top = module.pad_t
            bottom = module.pad_b
            h = h + """
    net = tf.pad(net, [[0 ,0], [{}, {}], [{}, {}], [0 ,0]], 'REFLECT')""".format(str(top), str(bottom), str(left), str(right))
        elif module._typename == b'nn.SpatialConvolution':
            weight = module.weight.transpose([2, 3, 1, 0])
            weights["{}conv2d_".format(t7name)+ str(idx)] = weight
            bias = module.bias
            weights['{}conv2d_'.format(t7name)+str(idx)+'b'] = bias
            # strides = [1, module.dH, module.dW, 1]  # Assumes 'NHWC'
            h = h + """
    weight = tf.Variable({}, trainable=True, dtype=tf.float32)
    bias = tf.Variable({},trainable=True,dtype=tf.float32)
    net = tf.nn.conv2d(net, weight, [1, {},{}, 1], padding='VALID')
    net = tf.nn.bias_add(net, bias[0,:])""".format( "weights['{}conv2d_{}']".format(t7name,str(idx)),
                                  "weights['{}conv2d_{}b']".format(t7name,str(idx)),
                                  str(module.dH), str(module.dW))
        elif module._typename == b'nn.ReLU':
            h = h + """
    net = tf.nn.relu(net)"""
        elif module._typename == b'nn.SpatialUpSamplingNearest':

            h = h + """
    d = tf.shape(net)
    size = [d[1] * {}, d[2] * {}]
    net = tf.image.resize_nearest_neighbor(net, size)""".format(str(module.scale_factor), str(module.scale_factor))

        elif module._typename == b'nn.SpatialMaxPooling':
            h = h + """
    net = tf.nn.max_pool(net, ksize=[1, {},{} , 1], strides=[1, {}, {}, 1],
                         padding='VALID', name = str({}, 'utf-8'))""".format(module.kH, module.kW, module.dH, module.dW, module.name)
        else:
            raise NotImplementedError(module._typename)

    sio.savemat('weights.mat',weights)
    #print(h)
    weights=sio.loadmat('weights.mat')
    for key in weights:
        print(key)
        print(np.array(weights[key]).shape)


    text_file = open(outputpy, "w")

    text_file.write(h)

    text_file.close()
auto_writer(t7name = './tf-AdaIN-master/decoder-content-similar.t7', outputpy='outputDecoderCS.py', outputmat='weights.mat')