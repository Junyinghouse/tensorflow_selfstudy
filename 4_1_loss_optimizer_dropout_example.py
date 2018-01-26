import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size = 100

n_batch =mnist.train.num_examples// batch_size

x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

lr = tf.Variable(1e-3,dtype=tf.float32,name='learning_rate')

w1 = tf.Variable(tf.truncated_normal([784,400],stddev=0.1))
b1 = tf.Variable(tf.zeros([400])+0.1)
l1 = tf.nn.tanh(tf.add(tf.matmul(x,w1),b1))
l1_drop = tf.nn.dropout(l1,keep_prob)

w2 = tf.Variable(tf.truncated_normal([400,100],stddev=0.1))
b2= tf.Variable(tf.zeros([100])+0.1)
l2 = tf.nn.tanh(tf.add(tf.matmul(l1_drop,w2),b2))
l2_drop = tf.nn.dropout(l2,keep_prob)

w = tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b = tf.Variable(tf.zeros([10])+0.1)

y_pred = tf.nn.softmax(tf.matmul(l2_drop,w) +b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits= y_pred))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(21): # 迭代次数， 不断调整权重w 和 偏置b
        sess.run(tf.assign(lr,1e-3*(0.95**i)))
        for _ in range(n_batch):   #  训练模型，
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1})

        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1}) # 测试已经训练好的模型
        LR = sess.run(lr)
        if i%4 == 0:
            print('Iter {0}, testing accuracy: {1}, learning rate : {2}'.format(i,test_acc,LR))



