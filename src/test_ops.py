import tensorflow as tf

inData = tf.placeholder(tf.float32, [None,3])
outData = inData * 3

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData,feed_dict={inData:[[1,3,9]]})
print(output)



inData = tf.placeholder(tf.float32, [None,3])
outData = tf.maximum(inData, 3)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData,feed_dict={inData:[[1,3,9]]})
print(output)
