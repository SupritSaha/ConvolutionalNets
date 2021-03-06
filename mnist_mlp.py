# Loading required packages
# =============================================
import numpy as np
from keras.datasets import mnist  # Importing MNIST dataset
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization
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
input_size = x_train.shape[1]*x_train.shape[1] # Assuming fixed input size

# Resize
# ==============================================
x_train = np.reshape(x_train,[-1,input_size])
x_test = np.reshape(x_test,[-1,input_size])
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
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size)) # adding 1st hidden layer
model.add(BatchNormalization()) # Batch normalization
model.add(Activation('relu')) # adding non linearity
model.add(Dropout(dropout_prob)) # regularizing using dropout
model.add(Dense(hidden_units)) # adding 2nd hidden layer
model.add(BatchNormalization()) # Batch normalization
model.add(Activation('relu')) # adding non linearity
model.add(Dropout(dropout_prob)) # regularizing using dropout
model.add(Dense(num_classes)) # Final layer
model.add(Activation('softmax')) # Adding softmax to get predicted probabilities
model.summary() # Helps in viewing number of parameters


plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

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
                          mode='auto') # early stopping is a form of regularization used to avoid overfitting when training a learner with an iterative method, such as gradient descent

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
print("Test accuracy: %.1f%%" % (100.0 * acc)) # 97.9%
