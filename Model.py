import tensorflow as tf
from trainingUtils import save, load
from layers import AdaIN
import scipy.io as sio
import numpy as np
from preprocessing import DataStream
import matplotlib.pyplot as plt

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
        self.ids, self.batch = self.data.get_batch()
        self.weights_encoder = \
            {
                'conv2d_0': tf.Variable(
                    np.expand_dims(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_0'][:, :, 1, :], 2),
                    trainable=True, dtype=tf.float32),
                'conv2d_0b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_0b'], trainable=True,
                                         dtype=tf.float32),#TODO: TU SI OSTAL - dej bias še iz .mat, potem dodaj še cov2d_0 v encoder, potem naredi downsampling in upsampling, potem pa compute moments, potem loss iz momentov, loss iz rekonstrukcij.. to je to:)
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
                'conv2d_16': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_16'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_16b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_16b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_19': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_19'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_19b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_19b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_22': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_22'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_22b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_22b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_25': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_25'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_25b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_25b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_29': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_29'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_29b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_29b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_32': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_32'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_32b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_32b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_35': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_35'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_35b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_35b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_38': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_38'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_38b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_38b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_42': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_42'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_42b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_42b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_45': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_45'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_45b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_45b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_48': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_48'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_48b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_48b'], trainable=True,
                                          dtype=tf.float32),
                'conv2d_51': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_51'], trainable=True,
                                         dtype=tf.float32),
                'conv2d_51b': tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_51b'], trainable=True,
                                          dtype=tf.float32)
            }
        self.weights_decoder = \
            {
                'conv2d_1': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_1'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_1b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_1b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_5': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_5'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_5b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_5b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_8': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_8'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_8b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_8b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_11': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_11'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_11b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_11b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_14': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_14'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_14b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_14b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_18':tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18'], trainable=True,
                            dtype=tf.float32),
                'conv2d_18b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18b'], trainable=True,
                           dtype=tf.float32),
                'conv2d_21': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21'], trainable=True,
                             dtype=tf.float32),
                'conv2d_21b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21b'], trainable=True,
                           dtype=tf.float32),
                'conv2d_25': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25'],
                                         trainable=True,
                                         dtype=tf.float32),
                'conv2d_25b': tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25b'],
                                          trainable=True,
                                          dtype=tf.float32),
                'conv2d_28': tf.Variable(np.expand_dims(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28']
                                                        [:,:,:,0],3),  # expand output_dim to 1
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


        with tf.name_scope('encoder'):
            net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_0'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_0b'][0, :])  # 3 channel input!
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_2'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_2b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_5'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_5b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name=str(b'pool1', 'utf-8'))
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
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name=str(b'pool2', 'utf-8'))
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_16'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_16b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_19'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_19b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_22'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_22b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_25'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_25b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name=str(b'pool3', 'utf-8'))
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_29'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_29b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_32'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_32b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_35'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_35b'][0, :])
            net = tf.nn.relu(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_38'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_38b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='VALID', name=str(b'pool4', 'utf-8'))
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_42'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_42b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_45'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_45b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_48'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_48b'][0, :])
            net = tf.nn.relu(net)
            activations.append(net)
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            net = tf.nn.conv2d(net, self.weights_encoder['conv2d_51'], [1, 1, 1, 1], padding='VALID')
            net = tf.nn.bias_add(net, self.weights_encoder['conv2d_51b'][0, :])
            #net = tf.nn.relu(net)
            activations.append(net)

        return net, activations

    def decode(self, input):

        #net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')

        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_1'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_1b'][0, :])
        net = tf.nn.relu(net)
        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_5'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_5b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_8'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_8b'][0, :])
        net = tf.nn.relu(net)

        ### MOD
        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)
        ### MOD END

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_11'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_11b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_14'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_14b'][0, :])
        net = tf.nn.relu(net)

        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_18'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_18b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_21'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_21b'][0, :])
        net = tf.nn.relu(net)
        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_25'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_25b'][0, :])
        net = tf.nn.relu(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_28'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, [self.weights_decoder['conv2d_28b'][0, 0]])
        net = tf.image.resize_nearest_neighbor(net, [self.batch.shape[1],self.batch.shape[2]])
        return net


    def build_model(self):
        self.lr = tf.Variable(0.003,False,name='learning_rate')
        
        
        ids, batch = self.data.get_batch()
        self.input_c = tf.Variable(np.float32(batch), trainable=False, name='input_content')
        self.input_s = tf.Variable(np.float32(batch), trainable=False, name='input_style')

        self.latent_c, act_c = self.encode(input=self.input_c)

        self.latent_s, act_s = self.encode(input=self.input_s)

        self.latent_c_s = AdaIN(self.latent_c,self.latent_s, 1.0)

        self.output = self.decode(input=self.latent_c_s)
        #

        self.output_recon_c, act_r_c = self.encode(input=self.output)

        print(self.output.get_shape())
        print(self.latent_c.get_shape())
        print(self.output_recon_c.get_shape())
        self.loss = tf.nn.l2_loss(self.latent_c - self.output_recon_c)    # TODO: ERROR!! niso istih dimenzij...
        self.minimizeLoss = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        tf.global_variables_initializer().run()

        #idsc, batchc = self.data.get_batch()
        #idss, batchs = self.data.get_batch()
        #self.iterate = self.sess.run([self.output, self.loss,self.minimizeLoss], feed_dict={self.input_c:batchc, self.input_s:batchs})


with tf.Session() as sess:

    mod = Model(sess=sess, batch_size=1)  #BS drugo kot 1 ne deluje - niti nima smisla, saj ni nikjer BN*...
    #lr = tf.Variable(0.003, False, name='learning_rate')
    #tf.global_variables_initializer().run()
    mod.build_model()

    for i in range(5000):
        idsc, batchc = mod.data.get_batch()
        idss, batchs = mod.data.get_batch()
        [_, loss, output] = sess.run([mod.minimizeLoss,mod.loss,mod.output], feed_dict={mod.input_c:batchc, mod.input_s:batchs})
        print(loss)

    #print(np.array(mod.numbers).shape)

plt.imshow(np.array(output)[0,:,:,0])
plt.show()