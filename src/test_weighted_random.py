import tensorflow as tf

inData = tf.placeholder(tf.float32, [None,3])
sss = tf.reduce_sum(inData,reduction_indices=[1])

high = tf.cumsum(inData, axis=1, exclusive=False)
low = tf.cumsum(inData, axis=1, exclusive=True)
sss0 = tf.reshape(sss,[-1,1])

high0 = high / sss0
low0 = low / sss0

rand = tf.random_uniform(tf.shape(sss0), dtype=tf.float32)
# 
high1 = tf.less(rand, high0)
low1 = tf.less_equal(low0, rand)
# 
good = tf.logical_and(high1,low1)
good0 = tf.to_float(good)
# 
choice = tf.argmax(good0,dimension=1)
# 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 
# for i in range(10):
#     

print(sess.run(
    [
        rand,
        high1,
        low1,
        good,
        choice,
    ],feed_dict={inData:[[1.,2.,3.],[4.,5.,6.]]}))
