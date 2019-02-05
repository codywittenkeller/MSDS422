# MSDS Assignment 7 by Cody Wittenkeller
# DNN for computer vision
# Image classification for cats and dogs


# import base packages into the namespace for this program 
import numpy as np  # arrays and math functions
import time
from datetime import datetime  # use for time-stamps in activity log
import tensorflow as tf

RANDOM_SEED = 1

def reset_graph(seed=RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

print('\n-----------------------------------')
print('Data Preparation Phase')
print('-----------------------------------')

# import data for images
# image data was provided in .npy format based on provided source code
cats_64_3 = np.load('cats_1000_64_64_3.npy') #64 pixels in color
cats_64_1 = np.load('cats_1000_64_64_1.npy') #64 pixels in grayscale
cats_128_3 = np.load('cats_1000_128_128_3.npy') #124 pixels in color
cats_128_1 = np.load('cats_1000_128_128_1.npy') #124 pixels in grayscale


dogs_64_3 = np.load('dogs_1000_64_64_3.npy') #64 pixels in color
dogs_64_1 = np.load('dogs_1000_64_64_1.npy') #64 pixels in grayscale
dogs_128_3 = np.load('dogs_1000_128_128_3.npy') #124 pixels in color
dogs_128_1 = np.load('dogs_1000_128_128_1.npy') #124 pixels in grayscale


# create labels for the cat and dog images
y_cats = np.zeros(1000)
y_dogs = np.ones(1000)
y_data = np.concatenate((y_cats,y_dogs))

#combine datasets for 2x2 design methodology
data_64_3 = np.concatenate((cats_64_3,dogs_64_3))
data_64_1 = np.concatenate((cats_64_1,dogs_64_1))
data_128_3 = np.concatenate((cats_128_3,dogs_128_3))
data_128_1 = np.concatenate((cats_128_1,dogs_128_1))


from sklearn.preprocessing import MinMaxScaler
def data_scale(data):
    data = data.reshape(-1,width*height)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = data.reshape(-1,width,height,channels)

# create list of row# and labels for storage
row = 0
row_and_class = []
for row in range(2000):
    row_and_class.append((row,y_data[row]))

# create train and test sets
from sklearn.model_selection import train_test_split
row_and_class_train, row_and_class_test = train_test_split(row_and_class,
                                                           test_size=0.2,
                                                           random_state = RANDOM_SEED)


# define function to return a batch from dataset
from random import sample

# create a function that creates random batches for training and testing
def prepare_batch(dataset,row_and_class,batch_size):
    batch_row_and_class = sample(row_and_class,batch_size)
    X_batch = dataset[[x[0] for x in batch_row_and_class]]
    y_batch = y_data[[x[0] for x in batch_row_and_class]]
    return X_batch, y_batch


print('\n-----------------------------------')
print('TensorFlow Graph Construction Phase')
print('-----------------------------------')

# define start time for run time calculation
start_time = time.time()

# Made with the assistance from Geron Chapter 13 source code

# define shape values
data = data_128_1 # update this field for all 4 data sets for 2 by 2 methodology
height = data.shape[1]
width = data.shape[2]
channels = data.shape[3]

# define conv layer values
filters1 = 32
filters2 = 64
filters3 = 64
kernel_size = 5
pool_size = 2
strides = 2
reshape = int((height/8)*(width/8)*(filters3))

# define DNN values
n_outputs = 2 # only two outputs cat or dog
n_hidden = 1000 # number of hidden nodes
dropout_rate = 0.5 # add dropout for regularization
LEARNING_RATE = 0.0005   # meta-parameter in machine learning
scale = .001

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height,width,channels], name="X")
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

with tf.name_scope("Conv_and_pool"):
    conv1 = tf.layers.conv2d(X, filters=filters1, kernel_size=kernel_size, 
                             padding='same', activation=tf.nn.relu, name="conv1")
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=pool_size,strides=strides)
    conv2 = tf.layers.conv2d(pool1, filters=filters2, kernel_size=kernel_size, 
                             padding='same', activation=tf.nn.relu, name="conv2")
    pool2 = tf.layers.max_pooling2d(conv2,pool_size=pool_size,strides=strides)
    conv3 = tf.layers.conv2d(pool2, filters=filters3, kernel_size=kernel_size, 
                             padding='same', activation=tf.nn.relu, name="conv3")
    pool3 = tf.layers.max_pooling2d(conv3,pool_size=pool_size,strides=strides)
    pool3_flat = tf.reshape(pool3,[-1,reshape])

# define deep neural network after the conv. layers
with tf.name_scope("dnn"):
    #hidden1 = tf.layers.dense(pool3_flat, n_hidden, name="hidden1", 
                              #activation=tf.nn.relu,
                              #kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
    #hidden2 = tf.layers.dense(hidden1, n_hidden, name="hidden2",
                              #activation=tf.nn.relu,
                              #kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))
    #hidden2_dropout = tf.layers.dropout(hidden2,dropout_rate,training=training)
    logits = tf.layers.dense(pool3_flat, n_outputs, name="outputs")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    base_loss = tf.reduce_mean(xentropy)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #add L1 regularization for early stopping
    loss = tf.add_n([base_loss]+reg_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

print('\n-----------------------------------')
print('TensorFlow Model Training Phase')
print('-----------------------------------')
# model hyperparamters 
n_epochs = 50
batch_size = 128
n_iterations_per_epoch = len(row_and_class_train) // batch_size

# time the training session
start_time = time.time()

with tf.Session() as sess:
    init.run()
    # define start time for run time calculation
    start_time = time.time()
# iterate through all epochs, reporting accuracy on training set
    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(data,row_and_class_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        # return the accuracy for each epoch    
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("Train accuracy:", acc_train)
        save_path = saver.save(sess, "./my_catdog_model")
# print the run time of the training
print("--- %s seconds ---" % (time.time() - start_time))
sess.close()


print('\n-----------------------------------')
print('TensorFlow Model Evaluation Phase')
print('-----------------------------------')

# evaluate the model developed above on the test set
X_test, y_test = prepare_batch(data,row_and_class_test, len(row_and_class_test))

n_test_batches = 100
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_catdog_model")

    print("Computing final accuracy on the test set")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)

sess.close()

