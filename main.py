from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam
import os
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = 0
if not os.path.exists('mnist.keras'):
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_split=0.2, batch_size=100, epochs=10, verbose=2)
    model.save('mnist.keras')
else:
    model = load_model('mnist.keras')


predictions = model.predict(x=x_test, verbose=1)
predictions = np.round(predictions)

acc = 0
for i in range(len(predictions)):
    if predictions[i].all() == y_test[i].all():
        acc += 1

acc = acc / len(predictions) * 100
print(acc)