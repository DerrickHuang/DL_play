import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils

import numpy as np
import os
from IPython import embed

import cifar_lenet5_forward
from dataprocess import load_CIFAR10, Data_Generator, augmentation


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZE = 1e-8
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME = "cifar_model"

def backward(Xtr, Ytr):
    x = tf.placeholder(tf.float32, [
    BATCH_SIZE,
    cifar_lenet5_forward.IMAGE_SIZE,
    cifar_lenet5_forward.IMAGE_SIZE,
    cifar_lenet5_forward.NUM_CHANNEL])
    #y_ = tf.placeholder(tf.float32, [None, cifar_lenet5_forward.OUTPUT_NODE])
    y_ = tf.placeholder(tf.int32, [BATCH_SIZE,])
    y = cifar_lenet5_forward.forward(x,True,REGULARIZE)
    global_step = tf.Variable(0, trainable = False)

    #ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_ , 1))
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = y_ )
    cem = tf.reduce_mean(ce)
    loss = cem
    #loss += tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        50000 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')

    saver = tf.train.Saver()

    data_generator = Data_Generator(Xtr, Ytr)
    batch_generator = data_generator.batch_gen(100)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            #image_batch, label_batch = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=BATCH_SIZE)
            image_batch, label_batch = next(batch_generator)
            image_batch = augmentation(image_batch)
            #print('Training data shape: ', image_batch.shape)
            #print('Training data shape: ', label_batch.shape)
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict = {x: image_batch, y_: label_batch})
            if i%10 == 0:
                print("After %d training step(s), loss on training batch is %g"%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main():
    cifar10_dir = 'cifar-10-batches-py'
# 看看数据集中的一些样本：每个类别展示一些
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)


    backward(X_train, y_train)

if __name__ == '__main__':
    main()











