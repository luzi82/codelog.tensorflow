import tensorflow as tf

for _ in range(10):

    outData = tf.nn.dropout(tf.ones([10],dtype=tf.float32),0.5)
    #outData = outData + outData
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    output = sess.run(outData)
    print(output)
