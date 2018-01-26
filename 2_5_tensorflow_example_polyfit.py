import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# prepare data
N = 300
xs = np.linspace(-3,3,N)
ys= np.sin(xs) + np.random.uniform(-0.2,0.2,N)

# define placeholder
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

# initialize the weigths and biases
w1 = tf.Variable(tf.random_normal([1]),name='w1')
w2 = tf.Variable(tf.zeros([1]),name='w2')
w3 = tf.Variable(tf.zeros([1]),name='w3')
bias = tf.Variable(tf.random_normal([1]),name='bias')

# the output/prediction
prediction = tf.add(tf.multiply(X,w1),bias)
prediction = tf.add(tf.multiply(tf.pow(X,2),w2),prediction)
prediction = tf.add(tf.multiply(tf.pow(X,3),w3),prediction)

# create the loss function
loss = tf.reduce_mean(tf.square(prediction -Y))/N


# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)

#minimize the loss function
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for  i  in range(200):
        sess.run(train,feed_dict={X:xs,Y:ys})
        if i %20 == 0:
            print('the loss of iter {0} is {1} '.format(i,sess.run(loss,feed_dict={X:xs,Y:ys})))
        # total = 0
        # for x,y in zip(xs,ys):
        #     __,l = sess.run([train,loss],feed_dict={X:x,Y:y})
        #     total += l
        # if i%10 == 0:
        #     print('the loss of Iter {0} is {1}'.format(i,total))
    W1,W2,W3,B = sess.run([w1,w2,w3,bias])


    plt.scatter(xs,ys)
    plt.plot(xs,W1*xs+W2*xs*xs+W3*xs*xs*xs + B,'r-.',lw = 2)
    plt.show()
