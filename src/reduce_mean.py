import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,3])

outData0 = tf.reduce_mean(inData0)
outData1 = tf.reduce_mean(inData0,reduction_indices=[1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData0,feed_dict={inData0:[[1,3,9],[2,4,6],[1,3,7]]})
print(output)

output = sess.run(outData1,feed_dict={inData0:[[1,3,9],[2,4,6],[1,3,7]]})
print(output)
