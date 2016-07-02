import tensorflow as tf

rand = tf.random_normal([3],stddev=0.1)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for _ in range(10):
    output = sess.run(rand)
    print(output)
