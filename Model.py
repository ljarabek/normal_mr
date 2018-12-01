import tensorflow as tf
from trainingUtils import save, load
from layers import adaptive_instance_norm
import scipy.io as sio
import numpy as np

weights = sio.loadmat("weights.mat")


class Model:
    def __init__(self, sess, ckpt=None):
        """
        :param sess: tf.Session()
        :param ckpt: str
        """
        self.sess = sess
        self.saver = tf.train.Saver()
        self.weights_encoder = \
            {
                'conv2d_2': tf.Variable(
                    np.expand_dims(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_2'][:, :, 1, :], 2),
                    trainable=True, dtype=tf.float32),
                'conv2d_2b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_2b'],
                                         trainable=True, dtype=tf.float32),
                'conv2d_5': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_5'], trainable=True,
                                        dtype=tf.float32),
                'conv2d_5b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_5b'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_9': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_9'], trainable=True,
                                        dtype=tf.float32),
                'conv2d_9b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_9b'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_12': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_12'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_12b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_12b'], trainable=True,
                                          dtype=tf.float32),
            }
        self.weights_decoder = \
            {
                'conv2d_18':tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18'], trainable=True,
                            dtype=tf.float32),
                'conv2d_18b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18b'], trainable=True,
                           dtype=tf.float32),
                'conv2d_21': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21'], trainable=True,
                             dtype=tf.float32),
                'conv2d_21b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21b'], trainable=True,
                           dtype=tf.float32)
                # TODO: TUKI SI OSTAL
            }


        if ckpt:
            self.saver.restore(self.sess, save_path=ckpt)
            print("restored from checkpoint " + str(ckpt))

    def encoder(self, input):
        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_2'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_2b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_5'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_5b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_9'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_9b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_12'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_12b'][0, :])
        return tf.nn.relu(net)

    def decoder(self, input):
        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_18'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_18b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_21'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_21b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        weight = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25'], trainable=True,
                             dtype=tf.float32)
        bias = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25b'], trainable=True,
                           dtype=tf.float32)
        net = tf.nn.conv2d(net, weight, [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, bias[0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        weight = tf.Variable(
            np.expand_dims(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28'][:, :, :, 1], -1),
            trainable=True, dtype=tf.float32)
        bias = tf.Variable(np.expand_dims(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28b'][:, 1], -1),
                           trainable=True, dtype=tf.float32)
        net = tf.nn.conv2d(net, weight, [1, 1, 1, 1], padding='VALID')
        return tf.nn.bias_add(net, bias[0, :])

