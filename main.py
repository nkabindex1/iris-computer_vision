import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

image_size = 227
training_data =[]
datadir = os.getcwd()
regex = 'iris-'
categories = [category for category in os.listdir() if regex in category ]

print(categories)
hog = cv2.HOGDescriptor()


# image handler 

# model

# pipeline
def create_training_data():
    for i, category in enumerate(categories):
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for image in os.listdir(path):
            image_array = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(image_array, (image_size,image_size))
            
            
            # sobel
            sobel_x = cv2.Sobel(new_array, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(new_array, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute the magnitude of the gradients
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize the gradient magnitude to [0, 255]
            gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
                
            #plt.imshow(new_array, cmap='gray')
            #plt.show()
            training_data.append([gradient_magnitude_normalized, class_num])
    random.shuffle(training_data)
create_training_data()


# perform sobel edge detection

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    
X = np.array(X).reshape(-1, image_size, image_size, 1) # for color images, the last part is 3
y = categorical_labels = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# normalize data before feeding to cnn

X_train = X_train / 255
X_test = X_test / 255



dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

#{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense
NAME = f'Iris-cnn-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs{NAME}')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


model = Sequential()
model.add(Conv2D(128, (3,3),input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(128, (3,3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
    
model.add(Flatten())

#for dense in range(dense_layer):
    #model.add(Dense(layer_size))

model.add(Dense(3))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X,y,epochs=10,batch_size=32,validation_split=0.1,callbacks=[tensorboard])




    
    


    