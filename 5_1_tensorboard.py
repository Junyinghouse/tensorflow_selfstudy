import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

batch_size = 100

n_batch =mnist.train.num_examples// batch_size

# 命名空间
with tf.name_scope('input'):
    #

    x = tf.placeholder(tf.float32,shape=[None,784],name='x-input')
    y = tf.placeholder(tf.float32,shape=[None,10],name='y-input')
    lr = tf.Variable(1e-3, dtype=tf.float32, name='learning-rate')

keep_prob = tf.placeholder(tf.float32)


with tf.name_scope('layer'):
    with tf.name_scope('Input_Layer'):
        with tf.name_scope('weights-L1'):
            w1 = tf.Variable(tf.truncated_normal([784,400],stddev=0.1),name='w1')
        with tf.name_scope('bias-L1'):
            b1 = tf.Variable(tf.zeros([400])+0.1,name='b1')
        with tf.name_scope('xw1_plus_b1'):
            l1 = tf.nn.tanh(tf.add(tf.matmul(x,w1),b1),name='tanh1_l1')
        l1_drop = tf.nn.dropout(l1,keep_prob)

    with tf.name_scope('Hidden_Layer'):
        with tf.name_scope('weights-L2'):
            w2 = tf.Variable(tf.truncated_normal([400,100],stddev=0.1),name='weights2')
        with tf.name_scope('bias-L2'):
            b2= tf.Variable(tf.zeros([100])+0.1,name='b2')
        with tf.name_scope('L1w2_plus_b2'):
            l2 = tf.nn.tanh(tf.add(tf.matmul(l1_drop,w2),b2),name='tanh2_l2')
        l2_drop = tf.nn.dropout(l2,keep_prob)

    with tf.name_scope('Output_Layer'):
        with tf.name_scope('w3'):
            w = tf.Variable(tf.truncated_normal([100,10],stddev=0.1),name='w3')
        with tf.name_scope('b3'):
            b = tf.Variable(tf.zeros([10])+0.1)
        with tf.name_scope('softmax_output'):
            y_pred = tf.nn.softmax(tf.matmul(l2_drop,w) +b,name='softmax_L')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits= y_pred),name='loss')

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
with tf.name_scope('train'):
    with tf.name_scope('optimizer-adam'):
        optimizer = tf.train.AdamOptimizer(lr)
    with tf.name_scope('train-step'):
        train_step = optimizer.minimize(loss)

with tf.name_scope('test'):
    with tf.name_scope('correct-prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1),name='correctPrediction')
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name='accuracy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer= tf.summary.FileWriter('.graphy/mnist',sess.graph)
    for i in range(5): # 迭代次数， 不断调整权重w 和 偏置b
        sess.run(tf.assign(lr,1e-3*(0.95**i)))
        for _ in range(n_batch):   #  训练模型，
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}) # 测试已经训练好的模型
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        LR = sess.run(lr)
        if i%2 == 0:
            print('Iter {0}, testing accuracy: {1}, training accuracy: {3}，learning rate : {2}'.format(i,test_acc,LR,train_acc))


    writer.close()
