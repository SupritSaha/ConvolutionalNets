# CIFAR 10 classification using Alexnet-like CNN
# =============================================
# Author : Suprit Saha

# Loading required packages
# =============================================
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical,plot_model

# Loading CIFAR 10 dataset
# ==============================================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
img_height, img_width, channel = x_train.shape[1],x_train.shape[2],x_train.shape[3]

# Setting hyperparameters
# =============================================
batch_size = 128
num_classes = len(np.unique(y_train))
epochs = 10
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)

# Original Alexnet trained on 1.2 million high-resolution images
# Images were downsampled and cropped to 256*256
# CIFAR10 has very small images hence no rescaling/cropping

# Preprocess
# ==============================================
x_train[:,:,:,0] = x_train[:,:,:,0] - x_train[:,:,:,0].mean()
x_train[:,:,:,1] = x_train[:,:,:,1] - x_train[:,:,:,1].mean()
x_train[:,:,:,2] = x_train[:,:,:,2] - x_train[:,:,:,2].mean()


# One hot encoding of target variable
# ==============================================
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Creating model like Alexnet
# ===============================================
model = Sequential()
# model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,))) for original Alexnet trained on ImageNet
model.add(Conv2D(96, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
plot_model(model, to_file='alexnet-cifar.png', show_shapes=True)


# Specifying target variable, loss function, accuracy metric
                                              # =================================================
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Train and validate the network
# ==================================================
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Predicting and getting test accuracy
# ====================================================
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Comments
# ====================================================
# 1. Images were downsampled and cropped to 256*256 in training Imagenet
# 2. Preprocessing : centered raw pixel - subtract train mean from raw pixel
# 3. Usage of ReLU as activation function
# 4. Multiple GPU training with cross gpu parallelization
# 5. Local Response Normalization - no longer used. Batch Norm better alternative
# ReLU neurons have unbounded activations and we need LRN to normalize that : https://www.quora.com/What-is-local-response-normalization
# 6. Overlapping pooling - prevents overfitting
# 7. 5 CNN + 3 FC layers - 10. LRN follow 1,2 conv layer, max pooling follow both LRN and fifth conv layer
# 8. Data augmentation -
# image translations and horizontal reflections
# random 224x224 patches + horizontal reflections from the 256x256 images
# Testing: five 224x224 patches + horizontal reflections
# Change the intensity of RGB channels
# PCA on the set of RGB pixel values throughout the ImageNet training set
# To each RGB image pixel factor added
# 9. Dropout
# 10. mini batch gradient descent with batch size = 128
# 11. Initial LR = 0.01, decreased by factor of 10 if validation deosn't improve
# 12. Momentum = 0.9,weight decay = 0.0005
# 13. Weights initialized from N(0,0.01). Bias initalization : conv 2,4,5 and FCS with 1 and rest 0
# 14. 5 CNN + 2 pretrained CNN(Imagenet 2011) gives best performance
# 15. Image similarity - Euclidean distance between last 4096-D FC layer
# could be made efficient by training an auto-encoder to
# compress these vectors to short binary codes
# 16. Network’s performance degrades if a single convolutional layer is removed
# 17. 60 million parameters
