# go-to imports
import csv
import cv2
import numpy as np

# Keras imports
from keras.models import Sequential
from keras.layers import Flatten, Dense

# read driving log to an array I can access
driving_logs = []
with open('/opt/behavioral-cloning-data/track-1/driving_log.csv') as driving_log_file:
    reader = csv.reader(driving_log_file)
    for line in reader:
        driving_logs.append(line)

# extract the front camera images and angle measurements
front_images = []
angle_measurements = []
for log in driving_logs:
    front_path = log[0]
    front_image = cv2.imread(front_path)
    front_images.append(front_image)
    angle_measurement = float(log[3])
    angle_measurements.append(angle_measurement)
    
# set up the training data and label arrays
X_train = np.array(front_images)
y_train = np.array(angle_measurements)

# create the model
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

# configure the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

# save the model
model.save('model.h5')