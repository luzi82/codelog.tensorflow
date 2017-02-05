import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,3,3])
outData0 = tf.reshape(inData0, [-1,9])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData0,feed_dict={inData0:[[[1,3,9],[2,4,10],[3,5,11]],[[-1,0,1],[1,-1,0],[0,1,-1]]]})
print(output)

inData0 = tf.placeholder(tf.float32, [None,3,3])
outData0 = tf.reshape(inData0, [-1,9,1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print(sess.run(outData0,feed_dict={inData0:[[[1,3,9],[2,4,10],[3,5,11]],[[-1,0,1],[1,-1,0],[0,1,-1]]]}))
