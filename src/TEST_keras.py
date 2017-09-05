from keras.models import Sequential
from keras.layers import Dense
import numpy

# (DEBUG) Set random seed.
seed = 7
numpy.random.seed(seed)

print 'Loading datasets...',
# Load training, validation, and testing datasets
training_set = numpy.load('../neural_net/training_set.npy')
validation_set = numpy.load('../neural_net/validation_set.npy')
testing_set = numpy.load('../neural_net/testing_set.npy')
print 'DONE!'

# Get input and output variables from training set
training_input = [x[3] for x in training_set]
training_output = [x[2] for x in training_set]
validation_input = [x[3] for x in validation_set]
validation_output = [x[2] for x in validation_set]
testing_input = [x[3] for x in testing_set]
testing_output = [x[2] for x in testing_set]

# Get the set of all possible path types, from the input of all sets
# This is used to set the number of inputs for the NN (1 imput = 1 path type)
print 'Getting all path types...',
path_types = set().union(*(dic.keys() for dic in training_input))
path_types.update(*(dic.keys() for dic in validation_input))
path_types.update(*(dic.keys() for dic in testing_input))
print 'DONE!'

print '----------------------------------------------------------------'
print path_types

"""
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
print rounded
"""