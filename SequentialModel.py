import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

train_lables = []
train_samples = []

for i in range(10000):
    young_age = randint(13,50)
    train_samples.append(young_age)
    train_lables.append(1)
    young_age = randint(13,50)
    train_samples.append(young_age)
    train_lables.append(1)
    
    old_age = randint(50,100)
    train_samples.append(old_age)
    train_lables.append(0)

lables = np.array(train_lables)
samples = np.array(train_samples)

model = Sequential()
model.add(Dense(10, activation = "relu", input_dim = 1))
model.add(Dense(8, activation = "sigmoid"))
model.add(Dense(4, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop",
             loss = "binary_crossentropy",
             metrics = ["accuracy"])
model.fit(samples, lables, epochs =20,batch_size = 10, validation_split = 0.1)
