"""
This file tests the accuracy of the NN based probability estimator.
"""

from ontology_parser import load_data
from extract_paths import get_all_pair_datasets_for_NN

# Load the ontology data into data structures
print "Loading data into structures..."
phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links = load_data('../data_files/hp.obo', '../data_files/go.obo', '../data_files/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', '../data_files/goa_human.gaf')
print "DONE!\n"

# Build training, validation, and testing sets for neural network
print "Building pair datasets for NN..."
get_all_pair_datasets_for_NN(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links, tr_size=10)
print "DONE!\n"