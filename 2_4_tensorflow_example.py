import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x_data = np.linspace(-3,3,100)
# noise = np.random.normal(0,0.2,100)
# y_data = np.sin(x_data) + noise
x_data = np.linspace(-3,3,100)
noise = np.random.normal(0,0.2,100)
y_data = 3*x_data + 0.2

plt.plot(x_data,y_data)
plt.show()

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 构建二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))
    W,B = sess.run([k,b])
    plt.scatter(x_data,y_data)
    plt.plot(x_data,W*x_data+B,'r-',lw=2)
    plt.show()