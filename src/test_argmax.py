import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,2,2])

outData0 = tf.arg_max(inData0) # arg_max() missing 1 required positional argument: 'dimension'

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData0,feed_dict={inData0:[[[0,4],[1,3]]]})
print(output)
