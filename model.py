import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_csv('fer2013.csv')

pixels = data['pixels'].values
emotions = data['emotion'].values

pixels = [np.fromstring(pixel, dtype='int', sep=' ') for pixel in pixels]
pixels = np.vstack(pixels) / 255.0

pixels = pixels.reshape(-1, 48, 48, 1)

emotions = to_categorical(emotions)

train_pixels, test_pixels, train_emotions, test_emotions = train_test_split(pixels, emotions, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_pixels, train_emotions, epochs=10, batch_size=32, validation_data=(test_pixels, test_emotions))

model.save('emotionModel.h5')
