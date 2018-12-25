import keras
import os
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.datasets import cifar10 #to find out more about the dataset - > https://www.cs.toronto.edu/~kriz/cifar.html
import matplotlib.image as mpimg
from skimage.io import imsave
import scipy.misc
from scipy import ndimage, misc
import numpy as np
from keras.models import model_from_json
from PIL import Image

# dir = "/Users/ishgupta/Documents/Fall 2018/185c/project/dataset/tiny-imagenet-200/val/images/"
dir = "/trainingdata/"
# dir = '/Users/ishgupta/Documents/Fall 2018/185c/dataset/helen_1/'
X = []
Y = []


def save(model):
    # Save model to disk
    model_json = model.to_json()
    with open("finalmodel5.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("finalmodel5.h5")
    print("Saved model to disk")


for image in os.listdir(dir):
    if '.jpg' in image:
        train_image = img_to_array(load_img(dir + image))
        # train_image = scipy.misc.imresize(train_image, (64, 64))
        # misc.imsave(image, train_image)
        train_image = train_image / 255
        X.append(rgb2lab(train_image)[:, :, 0])
        Y.append((rgb2lab(train_image)[:, :, 1:]) / 128)

X = np.array(X, dtype=float)  # this converts it into a huge vectors
Y = np.array(Y, dtype=float)


X = X.reshape(len(X), 64, 64, 1)
Y = Y.reshape(len(Y), 64, 64, 2)

split = int(0.80 * len(X))
train_images = X[:split]
train_labels = Y[:split]
test_images = X[split:]
test_labels = Y[split:]



model = Sequential()
model.add(InputLayer(input_shape=(64, 64, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='relu', padding='same'))


# Finish model
model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])

history = model.fit(x=train_images, y=train_labels,validation_split=0.1, batch_size=25, epochs=500, verbose=1)

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('Train_Acc_val.png')
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.savefig('Train_Acc_loss.png')

save(model)
