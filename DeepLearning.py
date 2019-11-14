#########################################################################
#               Recurrent NN with keras Example of sentiment analysis 
#########################################################################
#from keras.preprocessing import sequence
#from keras.models import Sequential
#from keras.layers import Dense, Embedding, LSTM
#from keras.datasets import imdb

#print("Loading Data...")
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)
#print(x_train[0])





#########################################################################
#               Convolutional NN with keras examples of mnist
#########################################################################
import keras
import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

# import the data
#(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
##(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = fashion_mnist.load_data()


## get the data into the shape we need <length, width, color_channel> or color channel first
#if K.image_data_format() == 'channels_first':
#    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
#    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
#    input_shape = (1, 28, 28)
#else:
#    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
#    test_images = mnist_test_images.reshape(mnist_test_images.shape[0],28, 28, 1)
#    input_shape = (28, 28, 1)

#train_images = train_images.astype('float32')
#test_images = test_images.astype('float32')
#train_images /= 255
#test_images /= 255

#train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
#test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

#def build_model(filepath=None):
#    if(filepath != None):
#        model = keras.models.load_model(filepath)
#    else:
#        # build the CNN model
#        model = Sequential()
#        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
#        # 64 3x3 kernels
#        model.add(Conv2D(64, (3,3), activation='relu'))
#        # reduce by taking the max of each 2x2 block
#        model.add(MaxPooling2D(pool_size=(2,2)))
#        # Dropout to avoid overfitting
#        model.add(Dropout(0.25))
#        # Flatten the results to one dimension for passing into our final layer
#        model.add(Flatten())
#        # Add a hidden layer to learn with
#        model.add(Dense(128, activation='relu'))
#        # Add another dropout
#        model.add(Dropout(0.5))
#        # Final Categorization from 0-9 with softmax
#        model.add(Dense(10, activation='softmax'))

#        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
#    return model

#def display_sample(num):
#    print(train_labels[num])
#    label = train_labels[num].argmax(axis=0)
#    image = train_images[num].reshape([28,28])
#    plt.title("Sample: %d  Label: %d"%(num, label))
#    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#    plt.show()

#model = build_model("MNIST_CNN.model")
### Takes about an hour to run
##history = model.fit(train_images, train_labels, 
##                    batch_size=32, 
##                    epochs=10, 
##                    verbose=2, 
##                    validation_data=(test_images, test_labels))


#score = model.evaluate(test_images, test_labels, verbose=0)
#print('Test loss: ', score[0])
#print('Test accuracy: ', score[1])

# Save the model
#model.save("MNIST_CNN.model")


#########################################################################
#               Keras examples of mnist
#########################################################################
# Using Keras to do the mnist problem

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adadelta
import matplotlib.pyplot as plt

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0,2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=100, epochs=15, verbose=2, validation_data=(test_images,test_labels))
score = model.evaluate(test_images, test_labels, verbose=0)
print("Test_loss: ", score[0])
print("Test accuracy: ", score[1])
print("Saving Model")
#model.save('mnist_nn.h5')

for x in range(1000):
    test_image = test_images[x,:].reshape(1,784)
    predicted_cat = model.predict(test_image).argmax()
    label = test_labels[x].argmax()
    if(predicted_cat != label):
        plt.title('Prediction: %d Label: %d' % (predicted_cat, label))
        plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()

#########################################################################
#               Tensorflow examples of mnist
#########################################################################
# Using tensorflow for the neural network example of mnist

#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
#import numpy as np

#sess = tf.InteractiveSession()
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#input_images = tf.placeholder(tf.float32, shape=[None, 784])
#target_labels = tf.placeholder(tf.float32, shape=[None, 10])

#hidden_nodes = 512

#input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
#input_biases = tf.Variable(tf.zeros([hidden_nodes]))

#hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
#hidden_biases = tf.Variable(tf.zeros([10]))

#input_layer = tf.matmul(input_images, input_weights)
#hidden_layer = tf.nn.relu(input_layer + input_biases)
#digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

#loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))
#optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)
#correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels, 1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#tf.global_variables_initializer().run()

#for x in range(2000):
#    batch = mnist.train.next_batch(100)
#    optimizer.run(feed_dict={input_images:batch[0], target_labels:batch[1]})
#    if((x+1) % 100 == 0):
#        print("Training epoch: " + str(x+1))
#        print("Accuracy: " + str(accuracy.eval(feed_dict={input_images:mnist.test.images, target_labels:mnist.test.labels})))

#def display_sample(num):
#    print(mnist.train.labels[num])
#    label = mnist.train.labels[num].argmax(axis=0)
#    image = mnist.train.images[num].reshape([28,28])
#    plt.title("Sample: %d  Label: %d"%(num, label))
#    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#    plt.show()

## display_sample(1235)
