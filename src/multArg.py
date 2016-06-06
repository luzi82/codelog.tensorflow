import tensorflow as tf

inData0 = tf.placeholder(tf.float32, [None,3])
inData1 = tf.placeholder(tf.float32, [None,3])
inData2 = tf.placeholder(tf.float32, [None,3])

outData1 = inData0 + inData1
outData2 = outData1 + inData2

sess = tf.Session()
sess.run(tf.initialize_all_variables())

output = sess.run(outData1,feed_dict={inData0:[[1,3,9]],inData1:[[-1,-2,-3]]})
print(output)

output = sess.run(outData2,feed_dict={inData0:[[1,3,9]],inData1:[[-1,-2,-3]],inData2:[[3,2,1]]})
print(output)
