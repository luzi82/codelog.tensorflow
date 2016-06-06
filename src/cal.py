import tensorflow as tf

inData = tf.placeholder(tf.float32, [None,3])

w = tf.constant([
 [-1,0,1],
 [1,-1,0],
 [0,1,-1]
],dtype=tf.float32,shape=[3,3])

outData = tf.matmul(inData,w)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData,feed_dict={inData:[[1,3,9]]})
print(output)

output = sess.run(outData,feed_dict={inData:[[1,3,9],[3,9,1],[9,1,3]]})
print(output)
