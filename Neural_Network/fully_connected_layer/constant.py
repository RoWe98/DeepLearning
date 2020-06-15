# 利用张量的方式实现全连接层

import tensorflow as tf 

# O = X@W+b
# 创建w,b张量
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1 # x@w1+b
o1 = tf.nn.relu(o1)  # o1 = relu(x@w1+b)

