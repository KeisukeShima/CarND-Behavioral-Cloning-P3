import csv
import cv2
import numpy as np

def get_data(path):
    images=[]
    measurements=[]
    lines=[]
    with open(path +'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = path + 'IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    return images, measurements

# Data
lines=[]
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        lines.append(line)
        
images=[]
measurements=[]
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Data2
lines=[]
with open('../data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data2/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Data3
lines=[]
with open('../data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data3/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Cropping2D

# set up lambda layer

# model start here
model = Sequential()

# Define End-to-End driving from NVidia's article
dropout_rate = 0.3
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(36, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(48, (5,5), padding='valid', activation='relu', strides=(2,2)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Dropout(dropout_rate))
model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(X_train.shape)
print(y_train.shape)

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose=1)

model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()