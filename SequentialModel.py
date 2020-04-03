import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy

# creating two empty list for samples and their lables
train_lables = []
train_samples = []

# generating random data for tow category of the age(young people, old people) withen 13-100 years age
for i in range(10000):
    young_age = randint(13,50)
    train_samples.append(young_age)
    train_lables.append(1)
    
    old_age = randint(50,100)
    train_samples.append(old_age)
    train_lables.append(0)

#     keras model accept the trianing sample in the form of numpy array, here we convert python list to numpy array
lables = np.array(train_lables)
samples = np.array(train_samples)

# scale the sample data between the range of 0 and 1
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform((samples).reshape(-1,1))

model = Sequential()
model.add(Dense(10, activation = "relu", input_dim = 1))
model.add(Dense(8, activation = "sigmoid"))
model.add(Dense(4, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop",
             loss = "binary_crossentropy",
             metrics = ["accuracy"])
model.fit(scaled_train_samples, lables, epochs =20,batch_size = 10, validation_split = 0.1)
