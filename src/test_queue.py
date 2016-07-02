import tensorflow as tf

q = tf.RandomShuffleQueue(20,10,tf.float32)
a0 = tf.placeholder(tf.float32,[1])

qpush = q.enqueue([a0])
qpull = q.dequeue()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#print("fdsa")

for i in range(20):
	sess.run(qpush,feed_dict={a0:[i]})
	#out = sess.run(qpull)
	#print (out)

for i in range(10):
	out = sess.run(qpull)
	print (out)
