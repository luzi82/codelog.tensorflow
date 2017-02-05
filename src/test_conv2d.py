import tensorflow as tf

test_input = tf.constant(
    [
        [[[0],[1],[0]],[[1],[0],[0]],[[0],[0],[1]]],
        [[[2],[0],[0]],[[0],[2],[0]],[[0],[0],[2]]],
    ],
    dtype=tf.float32
)
test_filter = tf.constant([[[[1]],[[0]],[[0]]],[[[0]],[[0]],[[0]]],[[[0]],[[0]],[[0]]]],dtype=tf.float32)
test_conv2d = tf.nn.conv2d(test_input,test_filter,[1,1,1,1],"SAME")

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print(sess.run(test_conv2d))
