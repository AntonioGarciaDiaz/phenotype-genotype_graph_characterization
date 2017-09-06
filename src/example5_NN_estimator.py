"""
This file executes the NN based estimators (decision algorithms),
and tests their accuracy
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy

def get_input_vars_from_set(path_dicts, path_types):
    """
    Get the list of input variables for every item in a dataset.
    This function uses a dictionary of alternate paths, and a list of all
    possible path types, to turn the dictionary into a list of integers,
    representing the number of alternate paths for each possible type.
    
    Args:
        -path_dicts: A list of dictionaries of alternate path types,
            representing alternate path types, and the number of paths.
            +Type: list[dict{str,int}]
        -input_vars: A list of all possible alternate path types.
            +Type: list[str]
    Returns:
        -input_vars: The number of alternate paths of each type,
            in the same order as path_types.
            +Type: list[list[int]]
    """
    input_vars = []
    for pth_dic in path_dicts:
        next_in_vars = []
        for path in path_types:
            if path in pth_dic:
                next_in_vars.append(pth_dic[path])
            else:
                next_in_vars.append(0)
        input_vars.append(next_in_vars)
    return input_vars


# (DEBUG) Set random seed.
seed = 7
numpy.random.seed(seed)

print 'Loading datasets...',
# Load training, validation, and testing datasets
training_set = numpy.load('../neural_net/training_set.npy')
validation_set = numpy.load('../neural_net/validation_set.npy')
testing_set = numpy.load('../neural_net/testing_set.npy')
print 'DONE!'

print 'Getting input and output variables...',
# Get raw input variables from training set (as lists of dictionaries)
training_raw_input = [x[3] for x in training_set]
validation_raw_input = [x[3] for x in validation_set]
testing_raw_input = [x[3] for x in testing_set]

# Get the set of all possible path types, from the input of all sets
# This is used to set the number of inputs for the NN (1 imput = 1 path type)
path_types = set().union(*(dic.keys() for dic in training_raw_input))
path_types.update(*(dic.keys() for dic in validation_raw_input))
path_types.update(*(dic.keys() for dic in testing_raw_input))

# Get input variables from training set
training_input = get_input_vars_from_set(training_raw_input, path_types)
validation_input = get_input_vars_from_set(validation_raw_input, path_types)
testing_input = get_input_vars_from_set(testing_raw_input, path_types)

# Get output variables from training set
training_output = [x[2] for x in training_set]
validation_output = [x[2] for x in validation_set]
testing_output = [x[2] for x in testing_set]
print 'DONE!'

# Build the NN model (4 layers, 32 and 64 neurons)
print '----------------------------------------------------------------'
print 'BUILDING AND FITTING NEURAL NETWORK'
model = Sequential()

model.add(Dense(32, activation="relu", kernel_initializer="uniform", input_dim=len(path_types)))
model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform",))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the NN with the training set's input and output
model.fit(training_input, training_output, epochs=150, batch_size=10,  verbose=2)

print 'CALCULATING VALIDATION SET PREDICTIONS'
# Calculate predictions and round them
predictions = model.predict(validation_input)
rounded_predict = []
for i in range(len(predictions)):
    new_prediction = (round(predictions[i][0]),validation_output[i])
    rounded_predict.append(new_prediction)
    print new_prediction