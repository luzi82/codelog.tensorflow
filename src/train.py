import tensorflow as tf

var0 = tf.Variable(tf.zeros([3]))
outData0 = var0 + tf.constant([-1,-2,-3],dtype=tf.float32,shape=[3])
loss0 = tf.reduce_sum(tf.abs(outData0))
train0 = tf.train.GradientDescentOptimizer(0.5).minimize(loss0)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#print(var0.eval(sess))

#sess.run(train0)
#print(var0.eval(sess))

for _ in range(10):
	sess.run(train0)
	print(var0.eval(sess))

#print(var0.eval(sess))
