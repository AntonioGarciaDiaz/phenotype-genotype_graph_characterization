from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load training set
training_set = numpy.loadtxt('../neural_net/training_set.npy', delimiter=',')
#validation_set = numpy.loadtxt('../neural_net/validation_set.npy', delimiter=',')
#testing_set = numpy.loadtxt('../neural_net/testing_set.npy', delimiter=',')
# split into input (X) and output (Y) variables
X = [x[3] for x in training_set]
Y = [x[2] for x in training_set]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)