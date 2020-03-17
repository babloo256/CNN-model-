# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3,input_shape =(64 , 64 , 3) , activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2,2))) 
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128 , activation = 'relu' ))
classifier.add(Dense(output_dim = 1 , activation = 'softmax' ))

classifier.compile(optimizer = 'adam' ,loss= 'categorical_crossentropy' , metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training set',
                                                target_size=(64, 64),
                                                batch_size=20,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test set',
                                            target_size=(64, 64),
                                            batch_size=20,
                                            class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2389,
        epochs=2,
        validation_data=test_set,
        validation_steps=100)