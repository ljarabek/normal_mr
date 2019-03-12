from Model import Model
import tensorflow as tf
from multi_slice_viewer import multi_slice_viewer


with tf.Session() as sess:
    model = Model(sess,7,ckpt=None)
    idc, batchc = model.data.get_batch()
    ids, batchs = model.data.get_fixed_batch()
    print(batchc.shape)
    lol = sess.run(model.output,feed_dict={model.input_s:batchs, model.act_c[1]:batchc})



#multi_slice_viewer(lol[:,:,:,0])