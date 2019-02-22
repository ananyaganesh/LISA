import tensorflow as tf
import numpy as np

vn_target = tf.constant([[[3,3,1,1,1,1,1], [3,3,1,1,1,1,1], [2,1,1,1,1,1,1], [4,1,1,1,1,1], [1,1,1,1,1,1], [1,2,1,1,1,1], [1,4,1,1,1,1,1]]])

predicate_predictions = tf.constant([[0,0,1,0,0,1,0]])

with tf.Session() as sess:
    sess.run()

print('done')