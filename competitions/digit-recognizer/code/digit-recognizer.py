import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')

train_x = np.array(train_set.iloc[:, 1:]) / 255
train_x = train_x.reshape((-1, 28, 28, 1))

test_x = np.array(test_set.iloc[:,:]) / 255
test_x = test_x.reshape((-1, 28, 28, 1))


label_y = np.array(train_set['label'])

batch_size = 64
num_labels = 10

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding='valid', strides=(2,2), input_shape=(28, 28, 1), data_format='channels_last', activation='relu'),
    keras.layers.Conv2D(16, kernel_size=3, padding='valid', strides=(2,2), data_format='channels_last', activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, label_y, epochs=2)


predictions = model.predict(test_x)

result = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': np.argmax(predictions, axis=1)})

result.to_csv(path_or_buf = '../input/submission.csv', index=False)
















