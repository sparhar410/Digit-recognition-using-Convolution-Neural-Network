import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.core import Dense
from keras.layers import Activation,BatchNormalization
from keras.optimizers import Adam
from keras.metrics import sparse_categorical_crossentropy

import matplotlib.pyplot as plt
import numpy as np


from mlxtend.data import loadlocal_mnist

x_train, y_train = loadlocal_mnist(
        images_path='C:/Users/AviD/AppData/Local/Programs/Python/Python37/MNIST_data/train-images.idx3-ubyte', 
        labels_path='C:/Users/AviD/AppData/Local/Programs/Python/Python37/MNIST_data/train-labels.idx1-ubyte')

x_test,y_test=loadlocal_mnist(
    images_path='C:/Users/AviD/AppData/Local/Programs/Python/Python37/MNIST_data/t10k-images.idx3-ubyte',
    labels_path='C:/Users/AviD/AppData/Local/Programs/Python/Python37/MNIST_data/t10k-labels.idx1-ubyte'
    )
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
    
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

#print(y_train[0])
#plt.imshow(x_test[0],cmap='gray')
#plt.show()


model = Sequential()
model.add(Conv2D(28,(3,3), input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x=x_train,y=y_train,epochs=100,verbose=2)
y=model.evaluate(x_test,y_test,verbose=1)
print(y)



