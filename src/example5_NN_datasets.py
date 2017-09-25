"""
This file shows an example on how to build the datasets for training and
testing the NN based estimators.
"""

from ontology_parser import load_data
from extract_paths import get_all_datasets_for_NN
# from extract_paths import alternative_get_all_datasets_for_NN

# Load the ontology data into data structures
print "Loading data into structures..."
phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links = load_data('../data_files/hp.obo', '../data_files/go.obo', '../data_files/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', '../data_files/goa_human.gaf')
print "DONE!\n"

# Build training, validation, and testing sets for neural network
print "Building pair datasets for NN..."
get_all_datasets_for_NN(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links, tr_size=100)
print "DONE!\n"

# Alternative formalism for the sets
#print "Building pair datasets for NN..."
#alternative_get_all_datasets_for_NN(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links, tr_size=100)
#print "DONE!\n"
