from Model import Model
import tensorflow as tf
from multi_slice_viewer import multi_slice_viewer


with tf.Session() as sess:
    model = Model(sess,7,ckpt="C:\MR_normalization\ckpts3\model.ckpt")
    idc, batchc = model.data.get_batch()
    ids, batchs = model.data.get_fixed_batch()
    lol = sess.run(model.output,feed_dict={model.input_c:batchc, model.input_s:batchs})



multi_slice_viewer(lol[:,:,:,0])