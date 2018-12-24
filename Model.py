import tensorflow as tf
from trainingUtils import save, load
from layers import AdaIN
import scipy.io as sio
import numpy as np
from preprocessing import DataStream

weights = sio.loadmat("weights.mat")


class Model:
    def __init__(self, sess, batch_size, root_dir = "C:/MR slike/healthy-axis2-slice100/",ckpt=None):
        """
        :param sess: tf.Session()
        :param ckpt: str
        :param batch_size: int
        """
        self.data = DataStream(batch_size=batch_size, root_dir=root_dir)
        self.sess = sess

        self.weights_encoder = \
            {
                'conv2d_0': tf.Variable(
                    np.expand_dims(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_0'][:, :, 1, :], 2),
                    trainable=True, dtype=tf.float32),
                'conv2d_0b': tf.Variable(),#TODO: TU SI OSTAL - dej bias še iz .mat, potem dodaj še cov2d_0 v encoder, potem naredi downsampling in upsampling, potem pa compute moments, potem loss iz momentov, loss iz rekonstrukcij.. to je to:)
                'conv2d_2': tf.Variable(
                    weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_2'],#[:, :, 1, :], 2),
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
                           dtype=tf.float32),
                # TODO: TUKI SI OSTAL
                'conv2d_25': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_25b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_28': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_28b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28b'],
                                          trainable=True,
                                          dtype=tf.float32)
            }


        self.saver = tf.train.Saver()
        if ckpt:
            self.saver.restore(self.sess, save_path=ckpt)
            print("restored from checkpoint " + str(ckpt))

    def encode(self, input):
        activations = []

        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_2'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_2b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_5'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_5b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_9'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_9b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_encoder['conv2d_12'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_encoder['conv2d_12b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        return net, activations

    def decode(self, input):
        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_18'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_18b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_21'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_21b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_25'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_25b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_28'], [1, 1, 1, 1], padding='VALID')

        return tf.nn.bias_add(net, self.weights_decoder['conv2d_28b'][0, :])


    def build_model(self):
        self.lr = tf.Variable(0.003,False,name='learning_rate')
        
        
        ids, batch = self.data.get_batch()
        self.input_c = tf.Variable(np.float32(batch), trainable=False, name='input_content')
        self.input_s = tf.Variable(np.float32(batch), trainable=False, name='input_style')

        self.latent_c, act_c = self.encode(input=self.input_c)

        self.latent_s, act_s = self.encode(input=self.input_c)

        self.latent_c_s = AdaIN(self.latent_c,self.latent_s, 1.0)

        self.output = self.decode(input=self.latent_c_s)

        self.output_recon_c = self.encode(input=self.output)

        tf.global_variables_initializer().run()

        idsc, batchc = self.data.get_batch()
        idss, batchs = self.data.get_batch()
        self.numbers = self.sess.run(self.output, feed_dict={self.input_c:batchc, self.input_s:batchs})


with tf.Session() as sess:

    mod = Model(sess=sess, batch_size=1)  #BS drugo kot 1 ne deluje!!
    #lr = tf.Variable(0.003, False, name='learning_rate')
    #tf.global_variables_initializer().run()
    mod.build_model()
    print(np.array(mod.numbers).shape)

