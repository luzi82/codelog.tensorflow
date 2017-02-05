import tensorflow as tf

inData = tf.placeholder(tf.bool, [None,3])
w = tf.where(inData)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print(sess.run(w,feed_dict={inData:[[True,False,False]]}))
print(sess.run(w,feed_dict={inData:[[True,False,True]]}))

print(sess.run(w,feed_dict={inData:[
    [True,False,True],
    [True,False,False],
    [False,True,False]
]}))
