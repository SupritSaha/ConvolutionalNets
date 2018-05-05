# CIFAR-10 classification using CNN
# =============================================
# Author : Suprit Saha

# Loading required packages
# =============================================
import numpy as np
from keras.datasets import cifar10  # Importing MNIST dataset
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical,plot_model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Creating class for CIFAR-10 classification
# ==============================================
class cifarClassifier:
    """
    The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class
    """
    def __init__(self):
        """
        Load data and initialize models
        """
        (self.x_train, self.y_train),(self.x_test, self.y_test) = cifar10.load_data()
        self.models = self.all_models()
        self.trained_model = None

    def normalize(self,x_train,x_test):
        """
        Rescale pixel values between 0 and 1
        """
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        return x_train, x_test

    def oneHotEncode(self,y_train,y_test):
        """
        One hot encoding of target variable
        """
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return y_train, y_test

    def trainMLP(self):
        """
        Train CIFAR using MLP
        """
        (x_train, x_test) = self.normalize(self.x_train, self.x_test)
        (y_train, y_test) = self.oneHotEncode(self.y_train, self.y_test)

        num_epochs = 10
        batch_size = 128
        validation_ratio = 0.1

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

        model = self.models['mlp']
        model.fit(x=x_train,
                  y=y_train,
                  epochs=num_epochs,
                  batch_size=batch_size,
                  callbacks=callbacks_list,
                  validation_split=validation_ratio)

        loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print("Test accuracy: %.1f%%" % (100.0 * acc))

        return model

    def trainCNN(self):
        """
        Train CIFAR using CNN
        """
        (x_train, x_test) = self.normalize(self.x_train, self.x_test)
        (y_train, y_test) = self.oneHotEncode(self.y_train, self.y_test)

        num_epochs = 10
        batch_size = 128
        validation_ratio = 0.1


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

        model = self.models['cnn']

        model.fit(x=x_train,
                  y=y_train,
                  epochs=num_epochs,
                  batch_size=batch_size,
                  callbacks=callbacks_list,
                  validation_split=validation_ratio)


        loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print("Test accuracy: %.1f%%" % (100.0 * acc))

        return model

    def all_models(self):
        """
        Two models - CNN & MLP
        """
        dropout_prob = 0.3
        optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
        accuracy_metric = ['accuracy']
        hidden_units = 256
        num_classes = 10

        mlp = Sequential()
        mlp.add(Flatten(input_shape=self.x_train.shape[1:])) # adding 1st hidden layer
        mlp.add(BatchNormalization()) # Batch normalization
        mlp.add(Activation('relu')) # adding non linearity
        mlp.add(Dropout(dropout_prob)) # regularizing using dropout
        mlp.add(Dense(hidden_units)) # adding 2nd hidden layer
        mlp.add(BatchNormalization()) # Batch normalization
        mlp.add(Activation('relu')) # adding non linearity
        mlp.add(Dropout(dropout_prob)) # regularizing using dropout
        mlp.add(Dense(num_classes)) # Final layer
        mlp.add(Activation('softmax')) # Adding softmax to get predicted probabilities

        mlp.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

        kernel_size = 3 # 3*3 filter size
        pool_size = 2 # max pooling size
        filters = 64 # number of filters
        img_rows,img_cols,img_channels = self.x_train.shape[1],self.x_train.shape[1],3
        input_shape= (img_rows, img_cols, img_channels)

        cnn = Sequential()
        cnn.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu',
                         input_shape=input_shape))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size))

        cnn.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size))

        cnn.add(Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         activation='relu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size))

        cnn.add(Flatten())
        cnn.add(BatchNormalization()) # Batch normalization
        cnn.add(Activation('relu')) # adding non linearity
        cnn.add(Dropout(dropout_prob))
        cnn.add(Dense(num_classes))
        cnn.add(Activation('softmax'))

        cnn.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy']
                    )
        return {'cnn' : cnn, 'mlp': mlp}

    def train(self, method):
        """
        Wrapper method to call the actual training method
        """
        if method in self.models:
            method = ''.join(['train', method.upper()])
            method = getattr(self, method)
            self.trained_model = method()
        else:
            raise("Not a Available method, methods available are \n 1.mlp \n2.cnn")

    def get_model(self, method):
        """
        get the model of a given method
        """
        if method in self.models:
            """ return  actual model"""
            return self.models[method]

    def get_test_data(self):
        """Get test data"""
        return (self.x_test, self.y_test)

def train_models():
    """
    Train both cnn and mlp
    """
    print("\nTraining MLP\n")
    cc = cifarClassifier()
    cc.train('mlp')
    #loss: 1.5691 - acc: 0.4495 - val_loss: 1.5196 - val_acc: 0.4598
    #Test accuracy: 46.8%

    print("Training CNN\n")
    cc.train('cnn')
    #loss: 0.8377 - acc: 0.7057 - val_loss: 0.8415 - val_acc: 0.7074
    #Test accuracy: 69.5%

if __name__ == '__main__':
    train_models()

# Comments
# ==========================================================
# 1. ConvNets perform much better than MLP
# 2. Better to use pretrained architectures especially on small datasets
