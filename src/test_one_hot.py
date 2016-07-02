import tensorflow as tf

in_0 = tf.placeholder(tf.float32, [None,3])
idx_0 = tf.placeholder(tf.int64, [None])
out_0 = tf.placeholder(tf.float32, [None])

mask = tf.one_hot(idx_0, 3, axis=-1)
in_mask = in_0 * mask
mid = tf.reduce_sum(in_mask, reduction_indices=[1])
diff = out_0 - mid
diff_abs = tf.abs(diff)
diff_abs_mean = tf.reduce_mean(diff_abs)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

a,b,c,d,e,f = sess.run([mask,in_mask,mid,diff,diff_abs,diff_abs_mean],feed_dict={in_0:[[1,3,9],[2,4,6],[1,3,7]],idx_0:[0,1,2],out_0:[10,100,1000]})
print({
    'mask':a,
    'in_mask':b,
    'mid':c,
    'diff':d,
    'diff_abs':e,
    'diff_abs_mean':f,
})
