import keras.preprocessing.image
import numpy as np
import matplotlib.pyplot as plt
from emnist import list_datasets, extract_training_samples, extract_test_samples
from tensorflow.keras.datasets import mnist
from keras import layers
from tensorflow import data as tf_data

x_train, labels_train = extract_training_samples('digits')
x_test, labels_test = extract_test_samples('digits')

#(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
#print(list_datasets())
#print(trainX.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
from tensorflow.keras.utils import  to_categorical
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape (x_test.shape[0], 28, 28, 1)

data_augmentation_layers = [
    layers.RandomRotation(factor=0.03),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1, 'constant'),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

#x_train_augmented = data_augmentation(x_train)

plt.figure(figsize=(20, 2))
for i in range(0,20):
    ax=plt.subplot(2,20,i+1)
    plt.imshow(x_train[i,:], cmap=plt.get_cmap('gray_r'))
    plt.title(labels_train[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#plt.figure(figsize=(20, 2))
#for i in range(0,20):
#    ax=plt.subplot(2,20,i+1)
#    plt.imshow(x_train_augmented[i,:], cmap=plt.get_cmap('gray_r'))
#    plt.title(labels_train[i])
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization

net = Sequential()

net.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
net.add(BatchNormalization())
net.add(Conv2D(32,kernel_size=3,activation='relu'))
net.add(BatchNormalization())
net.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
net.add(BatchNormalization())
net.add(Dropout(0.4))

net.add(Conv2D(64,kernel_size=3,activation='relu'))
net.add(BatchNormalization())
net.add(Conv2D(64,kernel_size=3,activation='relu'))
net.add(BatchNormalization())
net.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
net.add(BatchNormalization())
net.add(Dropout(0.4))

net.add(Flatten())
net.add(Dense(128, activation='relu'))
net.add(BatchNormalization())
net.add(Dropout(0.4))
net.add(Dense(10, activation='softmax'))

net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(net.summary())
from tensorflow.keras.utils import plot_model
plot_model(net, to_file='network_structure_2.png', show_shapes=True)
net.compile(loss='categorical_crossentropy', optimizer='adam')

history = net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=256)

# net.save("network_for_mnist_monday_final.h5")

plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#from tensorflow.keras.models import load_model
#net = load_model("network_for_mnist_2.h5")

outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)

plt.figure(figsize=(8, 2))
for i in range(0,8):
    ax = plt.subplot(2,8,i+1)
    plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
for i in range(0,8):
    output = net.predict(x_test[i,:].reshape(1, 28,28,1))
    output=output[0,0:]
    plt.subplot(2,8,8+i+1)
    plt.bar(np.arange(10.),output)
    plt.title(np.argmax(output))
plt.show()