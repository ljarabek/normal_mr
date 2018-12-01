import tensorflow as tf
from preprocessing import DataStream
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
Fully-convolutional autoencoder
"""

data = DataStream(10)
_, batch = data.get_batch()
weights = sio.loadmat("weights.mat")


#input = tf.Variable(initial_value=batch, dtype = tf.float32)
input = tf.Variable(initial_value=batch, dtype = tf.float32,trainable=False)
start = tf.identity(input)

#weight = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_0'], trainable=True, dtype=tf.float32) #1133
#bias = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_0b'], trainable=True, dtype=tf.float32)
#net = tf.nn.conv2d(net, weight, [1, 1, 1, 1], padding='VALID')
#net = tf.nn.bias_add(net, bias[0, :])
net = tf.pad(start, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
weight = tf.Variable(np.expand_dims(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_2'][:,:,1,:],2), trainable=True, dtype=tf.float32) #3 3 3 64
bias = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_2b'], trainable=True, dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1, 1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0, :])
net = tf.nn.relu(net)
net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_5'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_5b'], trainable=True, dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1, 1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0, :])
net = tf.nn.relu(net)
net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_9'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_9b'],trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])
net = tf.nn.relu(net)
net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_12'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/vgg_normalised.t7/conv2d_12b'],trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])
net = tf.nn.relu(net)

#mean_std = tf.nn.moments(net,axes=[1,2])
#net = tf.nn.batch_normalization()

net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_18b'],trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])
net = tf.nn.relu(net)
net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_21b'],trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])
net = tf.nn.relu(net)
net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25'], trainable=True, dtype=tf.float32)
bias = tf.Variable(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_25b'],trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])
net = tf.nn.relu(net)
net = tf.pad(net, [[0 ,0], [1, 1], [1, 1], [0 ,0]], 'REFLECT')
weight = tf.Variable(np.expand_dims(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28'][:,:,:,1],-1), trainable=True, dtype=tf.float32)
bias = tf.Variable(np.expand_dims(weights['./tf-AdaIN-master/decoder-content-similar.t7/conv2d_28b'][:,1],-1),trainable=True,dtype=tf.float32)
net = tf.nn.conv2d(net, weight, [1, 1,1, 1], padding='VALID')
net = tf.nn.bias_add(net, bias[0,:])


loss = tf.reduce_mean(tf.abs(input-net))
train = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in tqdm(range(1000)):
        _, batch = data.get_batch()
        #print(batch[0,50,50,0])
        _, l= sess.run((train,loss), feed_dict={input:batch})
        print(l)

    plt.imshow(np.array(sess.run(net, feed_dict={input:np.array(batch, np.float32)}))[0,:,:,0])
    plt.show()
    plt.imshow(np.array(batch, np.float32)[0,:,:,0])
    plt.show()