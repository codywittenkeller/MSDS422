# MSDS 422 Assignment 6 for Cody Wittenkeller

# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 1

# although we standardize X and y variables on input,
# we will fit the intercept term in the models
# Expect fitted values to be close to zero
SET_FIT_INTERCEPT = True

# import base packages into the namespace for this program
import pandas as pd  # data frame operations  
import numpy as np  # arrays and math functions
import time
from datetime import datetime  # use for time-stamps in activity log
import tensorflow as tf

# import MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


# Specify general machine learning meta-parameters
N_EPOCHS = 10
LEARNING_RATE = 0.1   # meta-parameter in machine learning
batch_size = 100

# Define logdir to be unique to current time
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf-logs'
logdir = 'tmp/regression-demo-{}/'.format(root_logdir, now)

    # create first model
print('\n-----------------------------------')
print('Two-Layer 100 Neuron Model')
print('-----------------------------------')
start_time_1 = time.time()

tf.reset_default_graph()

n_inputs = 28*28  # MNIST
n_hidden = 100 # to be adjusted in future models
n_outputs = 10

#define placeholder for x
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64,shape=(None),name="y")

# define hidden layers (2 and 3 hidden layers will be tested below)
with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2",activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
   
# define the cross entropy loss function to guide the gradient descent
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")
    
# define how the gradient descent will be conducted to minimize cross entropy
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training_op = optimizer.minimize(loss)

# define how the accuracy will be measured and defined for all models
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#execution phase
# set up numpy array for storing results
model_result_1 = np.zeros(N_EPOCHS)

def  shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# train the NN mmodel, output the final result
with tf.Session() as sess:
    init.run()
    for epoch in range(N_EPOCHS):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            model_result_1[epoch] = accuracy.eval(feed_dict={X: X_test, y:y_test})     
    save_path = saver.save(sess, "./my_model_final.ckpt")

# print the accuracy after the training and calibrating
print("Two-Layer 100 Neuron Accuracy:",model_result_1[N_EPOCHS-1])
# print the run time of the model
print("--- %s seconds ---" % (time.time() - start_time_1))
sess.close()


# all models follow the same code as above with different neurons and hidden layers
print('\n-----------------------------------')
print('Two-Layer 300 Neuron Model')
print('-----------------------------------')
tf.reset_default_graph()

start_time_2 = time.time()
n_inputs = 28*28  # MNIST
n_hidden = 300
n_outputs = 10

#define placeholder for x
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64,shape=(None),name="y")

with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2",activation=tf.nn.relu)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
   

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")
    
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# set up numpy array for storing results
model_result_2 = np.zeros(N_EPOCHS)

def  shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# train the NN mmodel, output the final result
with tf.Session() as sess:
    init.run()
    for epoch in range(N_EPOCHS):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            model_result_2[epoch] = accuracy.eval(feed_dict={X: X_test, y:y_test})     
    save_path = saver.save(sess, "./my_model_final.ckpt")

print("Two-Layer 300 Neuron Accuracy:",model_result_2[N_EPOCHS-1])
print("--- %s seconds ---" % (time.time() - start_time_2))
sess.close()

print('\n-----------------------------------')
print('Three-Layer 100 Neuron Model')
print('-----------------------------------')
tf.reset_default_graph()

start_time_3 = time.time()
n_inputs = 28*28  # MNIST
n_hidden = 100
n_outputs = 10

#define placeholder for x
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64,shape=(None),name="y")

with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2",activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, n_hidden, name="hidden3",activation=tf.nn.relu)
        logits = tf.layers.dense(hidden3, n_outputs, name="outputs")
   

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")
    
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# set up numpy array for storing results
model_result_3 = np.zeros(N_EPOCHS)

def  shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# train the NN mmodel, output the final result
with tf.Session() as sess:
    init.run()
    for epoch in range(N_EPOCHS):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            model_result_3[epoch] = accuracy.eval(feed_dict={X: X_test, y:y_test})     
    save_path = saver.save(sess, "./my_model_final.ckpt")

print("Three-Layer 100 Neuron Accuracy:",model_result_3[N_EPOCHS-1])
print("--- %s seconds ---" % (time.time() - start_time_3))
sess.close()

print('\n-----------------------------------')
print('Three-Layer 300 Neuron Model')
print('-----------------------------------')
tf.reset_default_graph()

start_time_4 = time.time()
n_inputs = 28*28  # MNIST
n_hidden = 300
n_outputs = 10

#define placeholder for x
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int64,shape=(None),name="y")

with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2",activation=tf.nn.relu)
        hidden3 = tf.layers.dense(hidden2, n_hidden, name="hidden3",activation=tf.nn.relu)
        logits = tf.layers.dense(hidden3, n_outputs, name="outputs")
   

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")
    
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# set up numpy array for storing results
model_result_4 = np.zeros(N_EPOCHS)

def  shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# train the NN mmodel, output the final result
with tf.Session() as sess:
    init.run()
    for epoch in range(N_EPOCHS):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            model_result_4[epoch] = accuracy.eval(feed_dict={X: X_test, y:y_test})     
    save_path = saver.save(sess, "./my_model_final.ckpt")

print("Three-Layer 300 Neuron Accuracy:",model_result_4[N_EPOCHS-1])
print("--- %s seconds ---" % (time.time() - start_time_4))
sess.close()

# results were recorded after running each individual model one at a time. 
# TF models proved to be difficult to iterate or run multiple models within a 
# single execution of code. Results were documented within the report table. 
# errors were experienced with tensorboard, so the elements assocaited with 
# tensorboard were removed for this submission until the specific errors can be fixed.


