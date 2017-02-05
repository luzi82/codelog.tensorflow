import tensorflow as tf

with tf.device('/cpu:0'):
    inData = tf.placeholder(tf.bool, [None,3])
    w = tf.cast(inData,tf.float32)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    print(sess.run(w,feed_dict={inData:[[True,False,False]]}))
    print(sess.run(w,feed_dict={inData:[[True,False,True]]}))
    
    print(sess.run(w,feed_dict={inData:[
        [True,False,True],
        [True,False,False],
        [False,True,False]
    ]}))
