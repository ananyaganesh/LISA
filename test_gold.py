import tensorflow as tf
import numpy as np

# batch_size x seq_len x numframes(=seq_len)
vn_target = tf.constant([[[3,3,1,1,1,1,1], [3,3,1,1,1,1,1], [2,1,1,1,1,1,1], [4,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,2,1,1,1,1,1], [1,4,1,1,1,1,1]]], shape=(1,7,7))

# batch x seq_len
predicate_predictions = tf.constant([[0,0,1,0,0,1,0]])
num_classes = 5

# batch x 1
trigger_counts = tf.reduce_sum(predicate_predictions, -1) # indicates num_preds in each entry in batch
seq_mask = tf.sequence_mask(tf.reshape(trigger_counts, [-1]))
vn_targets_indices = tf.where(seq_mask)
#vn_targets_indices = tf.Print(vn_targets_indices, [seq_mask, trigger_counts], "seq mask", summarize=100)
vn_targets_gathered = tf.gather_nd(tf.transpose(vn_target, [0,2,1]), vn_targets_indices)
vn_targets_one_hot = tf.one_hot(vn_targets_gathered, depth=num_classes, axis=-1)
vn_scores = tf.reshape(vn_targets_one_hot, [2*7, num_classes])
vn_scores = tf.Print(vn_scores, [vn_scores], "Scores", summarize=100)

with tf.Session() as sess:
    sess.run(vn_scores)

print('done')