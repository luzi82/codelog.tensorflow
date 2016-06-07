import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,3])

outData0 = inData0 + inData0
outData1 = outData0 + inData0

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run([outData0,outData1],feed_dict={inData0:[[1,3,9],[2,4,6],[1,3,7]]})
print(output)
print(output[0])
print(output[1])

a,b = sess.run([outData0,outData1],feed_dict={inData0:[[1,3,9],[2,4,6],[1,3,7]]})
print(a)
print(b)
