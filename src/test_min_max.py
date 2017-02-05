import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,3])

outData0 = tf.minimum(inData0, 5)
outData1 = tf.maximum(inData0, 5)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run([outData0,outData1],feed_dict={inData0:[[1,3,9],[2,4,6],[1,3,7]]})
print(output)
