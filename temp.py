import tensorflow as tf

a = tf.cast(tf.ones((2, 5, 7)), tf.bool)
b = tf.ones((2, 2, 4))
c = tf.ones((2, 2, 4))

out = tf.where(tf.equal(a, True), b, c)

tf.sequence_mask()