import tensorflow as tf

in0 = tf.constant([
    [1,2],
    [3,4],
])
in1 = tf.constant([
    [5,6,7],
    [8,9,10],
])
concat = tf.concat(1, [in0,in1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(concat))

in0 = tf.placeholder(dtype=tf.int32, shape=[None])
h0 = tf.one_hot(in0, 3, axis=-1, dtype=tf.int32)
h1 = tf.one_hot(in0, 4, axis=-1, dtype=tf.int32)
concat = tf.concat(1, [h0,h1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(concat,feed_dict={in0:[0,1]}))

in0 = tf.placeholder(dtype=tf.int32, shape=[None])
h0 = tf.one_hot(in0, 3, axis=-1, dtype=tf.int32)
in1 = tf.placeholder(dtype=tf.int32, shape=[None,2])
in1 = tf.reshape(in1,[-1,2])
concat = tf.concat(1, [h0,in1,in1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(concat,feed_dict={in0:[0,1],in1:[[2,3],[4,5]]}))
