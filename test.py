import tensorflow as tf
import numpy as np
mat = np.array([[1,2,2],[0,1,2],[2,0,1]])
mat_tf = tf.convert_to_tensor(mat, tf.int32)
one_hot_mat = tf.one_hot(indices=mat_tf, depth=3)

sample_weights = tf.reduce_sum(tf.multiply(one_hot_mat, [1,0,1]), 2)
with tf.Session() as sess:
	print('hot mat')
	print(one_hot_mat.eval())
	print('weights')
	print(sample_weights.eval())
