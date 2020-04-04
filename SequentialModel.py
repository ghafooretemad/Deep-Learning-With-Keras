# training a simple sequential keras model to classify weather a medicin has negative impact on a person or not
# the medicin doesn't have negative impact on the people with age of 13-50 
# the medicin has negative impact on the people with age of 50-100

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
t_samples = []
# generating random data for tow category of the age(young people, old people) withen 13-100 years age
for i in range(10000):
    young_age = randint(13,50)
    train_samples.append(young_age)
    train_lables.append(1)
    
    old_age = randint(50,100)
    train_samples.append(old_age)
    train_lables.append(0)
#    for prediction
for row in range(300):
    young_age = randint(13,50)
    t_samples.append(young_age)
    
    old_age = randint(50,100)
    t_samples.append(old_age)

#     keras model accept the trianing sample in the form of numpy array, here we convert python list to numpy array
lables = np.array(train_lables)
samples = np.array(train_samples)
test_samples = np.array(t_samples)
# scale the sample data between the range of 0 and 1
scaler = MinMaxScaler(feature_range = (0,1))
scaled_train_samples = scaler.fit_transform((samples).reshape(-1,1))
# for prediction 
scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

# creating simple sequential model with 4 layers, each layer uses different activation functions like relu, sigmoid
model = Sequential()
model.add(Dense(10, activation = "relu", input_dim = 1))
model.add(Dense(8, activation = "sigmoid"))
model.add(Dense(4, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

# complie the model 
model.compile(optimizer = "rmsprop",
             loss = "binary_crossentropy",
             metrics = ["accuracy"])
# train the model with the sample data and validate the model with the 10% of the data, this specified by the validation_split=0.1
model.fit(scaled_train_samples, lables, epochs =20,batch_size = 10, validation_split = 0.1)

# Predict the new data using the trained model
prediction = model.predict_classes(scaled_test_samples, batch_size = 5)
# print the classes of the each sample 
for row in prediction:
    print(row)
