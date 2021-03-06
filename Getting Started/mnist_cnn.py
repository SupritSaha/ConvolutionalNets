# MNIST digit classification using CNN
# =============================================
# Author : Suprit Saha

# Loading required packages
# =============================================
import numpy as np
from keras.datasets import mnist  # Importing MNIST dataset
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical,plot_model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

# Loading MNIST dataset
# ==============================================
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print("Training set dimensions :", x_train.shape) # 60000 images of size 28*28
print("Test set dimensions :", x_test.shape) # 10000 images of size 28*28

# Setting hyperparameters
# =============================================
num_epochs = 20
batch_size = 128
num_classes = len(np.unique(y_train))
hidden_units = 256
dropout_prob = 0.3
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
validation_ratio = 0.1
accuracy_metric = ['accuracy']
img_rows,img_cols,img_channels = x_train.shape[1],x_train.shape[1],1
kernel_size = 3 # 3*3 filter size
pool_size = 2 # max pooling size
filters = 64 # number of filters
input_shape= (img_rows, img_cols, img_channels) # Assuming fixed input size

# Resize
# ==============================================
x_train = np.reshape(x_train,[-1,*input_shape])
x_test = np.reshape(x_test,[-1,*input_shape])
print("Reshaped training set dimensions :", x_train.shape)
print("Reshaped test set dimensions :", x_test.shape)

# Normalize
# ==============================================
x_train = x_train.astype('float32') / 255 # Normalization accelerates your training speed
x_test = x_test.astype('float32') / 255

# One hot encoding of target variable
# ==============================================
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("OHE train target variable dimensions :", y_train.shape)
print("OHE test target variable dimensions :", y_test.shape)

# Model development
# ==============================================
# model is a stack of CNN-ReLU-MaxPooling
model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))

model.add(Flatten()) # convertibg matrix to array for FC layer
model.add(BatchNormalization()) # Batch normalization
model.add(Activation('relu')) # adding non linearity
model.add(Dropout(dropout_prob))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# Specifying target variable, loss function, accuracy metric
                                              # =================================================
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=accuracy_metric)


# Train and validate the network
# ==================================================
earlystop = EarlyStopping(monitor='val_acc',
                          min_delta=0.0001,
                          patience=5,
                          verbose=1,
                          mode='auto') # early stopping is a form of regularization
                          # used to avoid overfitting when training a learner with
                          # an iterative method, such as gradient descent

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=5,
                              min_lr=0.001) # Reduce learning rate when a metric has stopped improving

callbacks_list = [earlystop,reduce_lr]


model.fit(x=x_train,
          y=y_train,
          epochs=num_epochs,
          batch_size=batch_size,
          callbacks=callbacks_list,
          validation_split=validation_ratio)


# Predicting and getting test accuracy
# ====================================================
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Test accuracy: %.1f%%" % (100.0 * acc)) # 99.0 %

# Comments
# =====================================================
# 1. Sparse interactions - Running kernel over image can detect meaningful features
# This means less parameters to store , reducing memory requirements and fewer operations
# 2. Parameter sharing - Sliding same set of learned weights over image
# 3. Equivariant to translation
# 4. Not equivariant to change in scale or image rotation
# 5. Hierarchical feature learning
# 6. Very good feature extractors useful in transfer learning
