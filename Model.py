import tensorflow as tf
from trainingUtils import save, load
from layers import AdaIN
import scipy.io as sio
import numpy as np
import os
from preprocessing import DataStream
import matplotlib.pyplot as plt
from multi_slice_viewer import multi_slice_viewer
from init_weights import encoder_weights, decoder_weights
from tqdm import tqdm
weights = sio.loadmat("weights.mat")


class Model:
    def __init__(self, sess, batch_size, root_dir="C:\MR slike/healthy-axis2-slice100/", ckpt=None):
        """
        :param sess: tf.Session()
        :param ckpt: str
        :param batch_size: int
        """
        self.data = DataStream(batch_size=batch_size, root_dir=root_dir)
        self.sess = sess
        self.ids, self.batch = self.data.get_batch()

        self.weights_encoder = encoder_weights()

        self.weights_decoder = decoder_weights()

        self.build_model()

        self.saver = tf.train.Saver()
        if ckpt:
            self.saver.restore(self.sess, save_path=ckpt)
            print("restored from checkpoint " + str(ckpt))

    def save(self, dir):
        save(self.saver, self.sess, logdir=dir)
    def encode(self, input):
        """
        :param input: tf.Tensor (b h w c)
        :return: activations of relu_1,2,3,4 ; output
        """
        activations = []

        with tf.name_scope('encoder'):  # TODO: UPORABLJAJO SD IN MEAN OD RELU1_1,2_1,3_1,4_1;; popravi L2 Losses...
            net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            activations.append(net)
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
            activations.append(net) ## TODO: TUKAJ JE ZADNJA AKTIVACIJA ZA STYLE
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
        return net, activations

    def decode(self, input):
        activations  = []
        net = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_1'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_1b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_5'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_5b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_8'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_8b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_11'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_11b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_14'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_14b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)

        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_18'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_18b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_21'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_21b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)

        d = tf.shape(net)
        size = [d[1] * 2, d[2] * 2]
        net = tf.image.resize_nearest_neighbor(net, size)

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_25'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_25b'][0, :])
        net = tf.nn.relu(net)
        activations.append(net)

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        net = tf.nn.conv2d(net, self.weights_decoder['conv2d_28'], [1, 1, 1, 1], padding='VALID')
        net = tf.nn.bias_add(net, self.weights_decoder['conv2d_28b'][0, :])
        activations.append(net)
        return activations, net

    def build_model(self):
        self.lr = tf.Variable(0.003, False, name='learning_rate')

        self.input_c = tf.Variable(np.float32(self.batch), trainable=False, name='input_content')
        self.input_s = tf.Variable(np.float32(self.batch), trainable=False, name='input_style')

        self.latent_c, self.act_c = self.encode(input=self.input_c)

        self.latent_s, self.act_s = self.encode(input=self.input_s)

        self.latent_c_s = AdaIN(self.latent_c, self.latent_s, 1.0)

        self.act_d, self.output = self.decode(input=self.latent_c_s)

        self.output_recon_c, self.act_r_c = self.encode(input=self.output)

        with tf.name_scope('loss'):
            self.loss_content = tf.reduce_mean(
                tf.square(self.output_recon_c - self.latent_c_s))  # prej je blo reduce mean - tf.square (MSE)
            m_std_loss = []
            for orig_s, recon_s in zip(self.act_s, self.act_r_c):
                s_mean, s_var = tf.nn.moments(orig_s, [1, 2])
                r_mean, r_var = tf.nn.moments(recon_s, [1, 2])

                mean_l = tf.reduce_mean(tf.square(s_mean - r_mean))  # prej je blo reduce sum - tf.square
                std_l = tf.reduce_mean(tf.square(tf.sqrt(s_var) - tf.sqrt(r_var)))

                m_std_loss.append([mean_l, std_l])
            self.loss_style = tf.reduce_mean(m_std_loss)  # 5e-8 *

            self.loss = self.loss_content + self.loss_style *0.3#* 0.03  # + self.loss_content  # +self.loss_style)#self.loss_style #self.loss_content 0.001*  #

        self.minimizeLoss = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(self.loss)  # 0.0000003
        tf.global_variables_initializer().run()


    #def train_step(self, same_style = True):
    #    idsc, batchc = self.data.get_batch()
    #    batchs = self.data.get_fixed_batch()
    #    self.sess.run(self.minimizeLoss, feed_dict={})



with tf.Session() as sess:
    mod = Model(sess=sess, batch_size=7, ckpt= "C:\MR_normalization\ckpts3\model.ckpt")
    # lr = tf.Variable(0.003, False, name='learning_rate')
    # tf.global_variables_initializer().run()
    idss, batchs = mod.data.get_fixed_batch()
    for i in tqdm(range(1)):
        idsc, batchc = mod.data.get_batch()
        if idsc == idss:
            continue
        output, rc, _, ls, lc, act_decoder, act_encoder = sess.run(
            (mod.output, mod.output_recon_c, mod.minimizeLoss, mod.loss_style, mod.loss_content, mod.act_d, mod.act_s),
            feed_dict={mod.input_c: batchc, mod.input_s: batchs})

        #if (i == 1):
        #    plt.imshow(batchs[0, :, :, 0])
        #    plt.show()
        #    plt.imshow(output[0, :, :, 0])
        #    plt.show()
        print(str(i) + " Ls {} + Lc {}  ".format(ls, lc))  # str(i) + str(idss) + str(idsc) +
        # print(tf.trainable_variables())
    #mod.save("C:/MR_normalization/ckpts3/")

    # print(np.array(mod.numbers).shape)

for d in range(20):
    try:
        #print("{} layer of decoder: {}".format(8-d, np.array(act_decoder[8-d]).shape))
        print("{} layer of encoder: {}".format(d, np.array(act_encoder[d]).shape))
    except:
        print("it's broken bro")


for d in range(20):
    try:
        print("{} layer of decoder: {}".format(d, np.array(act_decoder[d]).shape))
        #print("{} layer of encoder: {}".format(d, np.array(act_encoder[d]).shape))
    except:
        print("it's broken bro")

#multi_slice_viewer(np.array(output)[:,:,:,0])
#multi_slice_viewer(np.array(batchc)[:,:,:,0])
# plt.imshow(np.array(batchs)[0,:,:,0])
# plt.show()"""
