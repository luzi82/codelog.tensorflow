import tensorflow as tf
import time

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

sess.close()
sess = None
saver = None

print('NAPJNHPV tf.Graph().as_default() test')

for _ in range(100):
    t0 = int(time.time()*1000)

    with tf.Graph().as_default(): # without this, the restore time will become very slow in loop
        var0, inData, train0 = create_train()
        with tf.Session() as sess:
            saver = tf.train.Saver({'var0':var0})

            t1 = int(time.time()*1000)
            saver.restore(sess,"/tmp/FZZACAGZOU")
    #         print(var0.eval(sess)) # [86.,11.,28.]
            saver = None
            t2 = int(time.time()*1000)

    t3 = int(time.time()*1000)

    print("{},{},{}".format(t1-t0,t2-t1,t3-t2))

print('TKCPICEE tf.reset_default_graph() test')

for _ in range(100):
    t0 = int(time.time()*1000)
    
    tf.reset_default_graph()

    var0, inData, train0 = create_train()
    
    sess = tf.Session()
    saver = tf.train.Saver({'var0':var0})

    t1 = int(time.time()*1000)
    saver.restore(sess,"/tmp/FZZACAGZOU")
    saver = None
    t2 = int(time.time()*1000)
    
    sess.close()
    sess = None

    t3 = int(time.time()*1000)

    print("{},{},{}".format(t1-t0,t2-t1,t3-t2))
