# training a simple sequential keras model to classify weather a medicin has negative impact on a person or not
# the medicin doesn't have negative impact on the people with age of 13-50 
# the medicin has negative impact on the people with age of 50-100
%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.metrics import categorical_crossentropy

# creating two empty list for samples and their lables
train_lables = []
train_samples = []
t_samples = []
t_lables = []
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
    t_lables.append(1)

    old_age = randint(50,100)
    t_samples.append(old_age)
    t_lables.append(0)

#     keras model accept the trianing sample in the form of numpy array, here we convert python list to numpy array
lables = np.array(train_lables)
samples = np.array(train_samples)
test_samples = np.array(t_samples)
test_lables = np.array(t_lables)

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

#     this function draws the confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# creating  confusion matrix    
cm  = confusion_matrix(test_lables, prediction)
# adding lables
lbls = ["Nigative impact", "not impact"]
# ploting the confusion matrix
plot_confusion_matrix(cm, lbls)

# save the architecture, weight, optimizer and configuration of the model
model.save("mTrained_model.h5")
# load the saved model to the new model
new_model = load_model("mTrained_model.h5")
# predict the same test data using the new loaded model
nprediction = new_model.predict_classes(scaled_test_samples, batch_size = 5)
# draw the confusion matrix for this new model also
cm  = confusion_matrix(test_lables, nprediction)
lbls = ["Nigative impact", "not impact"]
plot_confusion_matrix(cm, lbls)
# the confusion matrix is same for both of the model
