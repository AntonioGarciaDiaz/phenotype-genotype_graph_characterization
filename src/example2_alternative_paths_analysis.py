"""
This file shows an example on how to process the alternative paths
previously computed using "get_connected_phenotype_genotype_alternative_paths"
"""

from ontology_parser import *
from graph_methods import *
from extract_paths import *

#This code assumes the alternative paths have already been computed (at least partially)
#using the following method (see example1_loader.py for details)
#get_connected_phenotype_genotype_alternative_paths(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links)


#OPTION 1: File paths obtained from arguments
#import sys
#read_and_analyze_alternative_paths(sys.argv[1],sys.argv[2],sys.argv[3])

#OPTION 2: File paths hardcoded into constants
read_and_analyze_alternative_paths('../results/total_results.pkl', '../results/type_index.pkl', '../results/list_elems.pkl')
