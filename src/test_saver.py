import tensorflow as tf

def create_train():
    var0 = tf.Variable(tf.zeros([3]),dtype=tf.float32)
    inData = tf.placeholder(tf.float32, [None,3])
    outData0 = var0 - inData
    loss0 = tf.reduce_sum(tf.abs(outData0))
    train0 = tf.train.GradientDescentOptimizer(0.5).minimize(loss0)
    return var0, inData, train0

var0, inData, train0 = create_train()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for _ in range(1000):
    sess.run(train0,feed_dict={inData:[[86.,11.,28.]]})

print(var0.eval(sess)) # [86.,11.,28.]

saver = tf.train.Saver({'var0':var0})
saver.save(sess,"/tmp/FZZACAGZOU")

for _ in range(1000):
    sess.run(train0,feed_dict={inData:[[20.,51.,73.]]})

print(var0.eval(sess)) # [20.,51.,73.]

saver.restore(sess,"/tmp/FZZACAGZOU")

print(var0.eval(sess)) # [86.,11.,28.]

sess.close()
sess = None
saver = None

var0, inData, train0 = create_train()
sess = tf.Session()
saver = tf.train.Saver({'var0':var0})
saver.restore(sess,"/tmp/FZZACAGZOU")

print(var0.eval(sess)) # [86.,11.,28.]
