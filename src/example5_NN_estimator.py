"""
This file executes the NN based estimators (decision algorithms),
and tests their accuracy.
"""

from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
import numpy
import cPickle as pickle # For Python 2.7
# import _pickle as pickle # For Python 3

def f_beta_score(precision, recall, beta=1):
    """
    Calculates the f beta score of the estimation, given its precision and
    recall, and a value for beta (default is f1).
    Args:
        -precision: the precision value for the estimation (TP/TP+FP)
            +Type: float
        -recall: the recall (true positive ratio) value (TP/P = TP/TP+FN)
            +Type: float
        -beta: the beta value to be used (default is 1)
            +Type: int
    Returns:
        -f_beta: the corresponding f beta score
            +Type: float
    """
    beta_sq = beta**2
    f_beta = (1+beta_sq)*(precision*recall)/((beta_sq*precision)+recall)
    return f_beta

def build_NN_model(n_inputs, n_layers, n_neurons, act_funct, opt_algorithm):
    """
    Create a Keras sequential neural network model, with sigmoid output
    and parametrised characteristics.
    
    Args:
        -n_inputs: The number of inputs for the neural network. 
            +Type: int
        -n_layers: The number of neuron layers in the neural network. 
            +Type: int
        -n_neurons: The number of neurons in each of the network's layers. 
            +Type: int
        -act_funct: The activation function used by the neural network. 
            +Type: str
        -opt_algorithm: The optimization algorithm used by the neural network. 
            +Type: str
    Returns:
        -nn_model: The required neural network model.
            +Type: keras.models.Sequential
    """
    nn_model = Sequential()
    nn_model.add(Dense(n_neurons, activation=act_funct, kernel_initializer="uniform", input_dim=n_inputs))
    
    for i in range(n_layers):
        nn_model.add(Dense(n_neurons, activation=act_funct, kernel_initializer="uniform"))
    
    nn_model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    nn_model.compile(loss="binary_crossentropy", optimizer=opt_algorithm, metrics=["accuracy"])
    
    return nn_model

def relevant_statistics(predictions):
    """
    Analyses predictions made by a (neural network) estimator on a testing
    set, and calculates some relevant statistics: the precision, recall,
    and f1-score relative to positives (conn) and negatives (disc).
    
    Args:
        -predictions: A list of predictions, and their expected results.
            +Type: list[(int, int)]
    Returns:
        -relevant_stats: Some relevant statistics extracted from the comparison
            of the predictions with the expected results.
            These are, for both expected positive (here connected) and negative
            (here disconnected) cases, the precision, recall (TPR), and f1-score.
            +Type: list[(float, float, float), (float, float, float)]
    """
    shouldbe_pos = [x for x in predictions if x[1] == 1]
    shouldbe_neg = [x for x in predictions if x[1] == 0]
    shouldbe_pos_num = len(shouldbe_pos)
    shouldbe_neg_num = len(shouldbe_neg)
    
    # Get true and false positives (connected) and negatives (disconnected)
    true_pos = [x for x in shouldbe_pos if x[0] == 1]
    false_pos = [x for x in shouldbe_neg if x[0] == 1]
    true_pos_num = len(true_pos)
    false_pos_num = len(false_pos)
    false_neg_num = shouldbe_pos_num - true_pos_num
    true_neg_num = shouldbe_neg_num - false_pos_num
    
    # Relevant statistics: precision, recall, f1-score
    precision_conn = true_pos_num/(true_pos_num+false_pos_num)
    recall_conn = true_pos_num/shouldbe_pos_num
    f1_score_conn = f_beta_score(precision_conn, recall_conn)
    stats_conn = (precision_conn, recall_conn, f1_score_conn)
    
    precision_disc = true_neg_num/(true_neg_num+false_neg_num)
    recall_disc = true_neg_num/shouldbe_neg_num
    f1_score_disc = f_beta_score(precision_disc, recall_disc)
    stats_disc = (precision_disc, recall_disc, f1_score_disc)

    print "---------------------------------------"
    print "True positives =\t", true_pos_num
    print "False positives =\t", false_pos_num
    print "---------------------------------------"
    print "Precision (conn) =\t", precision_conn
    print "Recall (conn) =\t\t", recall_conn
    print "F1-Score (conn) =\t", f1_score_conn
    print "---------------------------------------"
    print "Precision (disc) =\t", precision_disc
    print "Recall (disc) =\t\t", recall_disc
    print "F1-Score (disc) =\t", f1_score_disc
    
    relevant_stats = [stats_conn, stats_disc]
    return relevant_stats

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

# (DEBUG) Set random seed.
seed = 7
numpy.random.seed(seed)

print 'Loading datasets...',
# Load training, validation, and testing datasets
training_set_conn = numpy.load('../neural_net/training_set/total_results_conn.npy')
validation_set_conn = numpy.load('../neural_net/validation_set/total_results_conn.npy')
testing_set_conn = numpy.load('../neural_net/testing_set/total_results_conn.npy')
training_set_disc = numpy.load('../neural_net/training_set/total_results_disc.npy')
validation_set_disc = numpy.load('../neural_net/validation_set/total_results_disc.npy')
testing_set_disc = numpy.load('../neural_net/testing_set/total_results_disc.npy')
print 'DONE!'

print 'Getting input and output variables...',
# Add the correct output for every input, join the two parts and shuffle
for pair in training_set_conn: numpy.append(pair,1)
for pair in validation_set_conn: numpy.append(pair,1)
for pair in testing_set_conn: numpy.append(pair,1)
for pair in training_set_disc: numpy.append(pair,0)
for pair in validation_set_disc: numpy.append(pair,0)
for pair in testing_set_disc: numpy.append(pair,0)

training_set = numpy.concatenate((training_set_conn,training_set_disc), axis=0)
validation_set = numpy.concatenate((training_set_conn,training_set_disc), axis=0)
testing_set = numpy.concatenate((training_set_conn,training_set_disc), axis=0)
numpy.random.shuffle(training_set)
numpy.random.shuffle(validation_set)
numpy.random.shuffle(testing_set)

print training_set_conn
print training_set_disc
print training_set

# Get input variables from training set (path counts, all but last element)
training_input = [x[:-1] for x in training_set]
validation_input = [x[:-1] for x in validation_set]
testing_input = [x[:-1] for x in testing_set]

# Get output variables from training set (last element, 1 or 0)
training_output = [x[-1] for x in training_set]
validation_output = [x[-1] for x in validation_set]
testing_output = [x[-1] for x in testing_set]
print 'DONE!'

print len(training_input), len(validation_input), len(testing_input)
print len(training_output), len(validation_output), len(testing_output)

print '---------------------------------------------------------------'
print '-------------------NEURAL NETWORK COMPARISON-------------------'
print '---------------------------------------------------------------'
nn_comparison_dict = dict()
# For each defined NN model
for a_fun in ["tanh","relu"]:
    for opt_al in ["adam", "sgd", "rmsprop"]:
        for n_lay in [2,3,4]:
            for n_neu in [16,32,64]:
                print "\nBuilding model with", n_lay, "layers,",
                print n_neu, "neurons, activation", a_fun,
                print "and optimization", opt_al
                nn_key = a_fun+", "+opt_al+", "+str(n_lay)+", "+str(n_neu)
                # Build the NN model and fit it with the training input and output
                neural_model = build_NN_model(len(path_types), 3, 16, "relu", "adam")
                neural_model.fit(training_input, training_output, validation_data=(validation_input,validation_output), epochs=150, batch_size=10, verbose=1)
                # Calculate predictions and round them
                print "\nPredictions for", n_lay, "layers,",
                print n_neu, "neurons, activation", a_fun,
                print "and optimization", opt_al
                predictions = neural_model.predict(testing_input)
                rnd_predictions = []
                for i in range(len(predictions)):
                    new_prediction = (int(round(predictions[i][0])),testing_output[i])
                    rnd_predictions.append(new_prediction)
                nn_comparison_dict[nn_key]=relevant_statistics(rnd_predictions)
                pickle.dump(nn_comparison_dict, open("../results/nn_comparison_dict.pkl", "wb"))