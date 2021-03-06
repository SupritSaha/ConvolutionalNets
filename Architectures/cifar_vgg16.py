# CIFAR 10 classification using VGG16-like CNN
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

# Preprocess
# ==============================================
x_train[:,:,:,0] = x_train[:,:,:,0] - x_train[:,:,:,0].mean()
x_train[:,:,:,1] = x_train[:,:,:,1] - x_train[:,:,:,1].mean()
x_train[:,:,:,2] = x_train[:,:,:,2] - x_train[:,:,:,2].mean()


# One hot encoding of target variable
# ==============================================
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Creating model like VGG16
# ===============================================
model = Sequential()
# Block 1
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())

# Block 2
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())

# Block 3
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(BatchNormalization())

# part of model been modified for CIFAR 10 Dataset
# Block 4
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# Block 5
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
plot_model(model, to_file='vgg16-cifar.png', show_shapes=True)

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
# =====================================================
# 1. Focusses of more depth by stacking more conv layers and small filters
# 2. Preprocessing like Alexnet
# 3. 5 configurations proposed(VGG-11,.....VGG16,VGG19)
# 4. VGG16 has 13 conv + 3 FC layers
# 5. Start with small # channels - 64 and increased gradually to 512,stride = 1
# 6. RELU follow all hidden layers
# 7. Max pooling window - 2*2
# 8. Use small filters for enabling deeper architecture.
    # a. 1 7*7 filter has same receptive filed as 3 3*3 conv layers
    # b. Decrease in number of parameters (3*(3^2C^2)= 27C^2 vs (7^2C^2)= 49C^2)
    # c. More number of relu's
# 9. Mini batch GD, batch size = 256,momentum = 0.9,weight decay = 0.0005
# 10. Dropout = 0.5 in 2 FC layers
# 11. Initial LR =0.01 ,decreased by factor of 10 if validation deosn't improve
# 12. 138 million parameters
# 13. Initialisation important due to depth
# 14. VGG11 randomly initialised N(0,0.01),bias =0
# 15. For other configs, first 4 conv + 3 FC layers initalised with VGG-11 weights,other layers randomly initailized
# 16. Alternatively can use Glorot and Bengio initialisation
# 17. Input fixed 224x224 image. Randomly cropped from isotropically rescaled images
# 18. Data augmentation : random horizontal flipping and RGB colour shift
# 19. Single scale training : Each training image is rescaled s/t shortest dimension=S. Now S = 256 or S = 384
# 20. Multi scale training : Training set augmentation by scale jittering
# Sample S randomly in range [Smin, Smax]; Smin = 256, Smax= 512
# 21. Multiscale models are trained by finetuning all layers of a single scale model
# 22. Single scale evaluation: Each test image scaled isotropically s/t smallest side is Q. Here S = Q (fixed scale) & Q=0.5(Smin+Smax) for jittered scale
# 23. Multiscale evaluation:Try different Qs= {S-32, S, S+32} for a single S. Multi-crop evaluation/Dense evaluation
# 24. FC layers converted to convolutional layers
# FC-1000: 4096x1000 params into 1000 filters size 1x1x4096
# Spatial pooling to obtain scores for 1000 classes
# 25. Scale jittering at test/training leads to better performance
# 26. Multicrop performs slightly better than dense evaluation
# 27. Ensemble of convnets (VGG16 + VGG19) with scale jittered training and multi scale eval gives best performance
# 28. Extracted features from VGG can be used in other tasks
