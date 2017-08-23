"""
This file tests the accuracy of the probability estimator by:
-Loading the ontology data into a graph (as in example 1)
-Loading statistics concerning different path types linking (dis)connected pairs
-Generating 1000 (dis)connected pairs that were not part of the training set
 used to generate the statistics
-Using the estimator on each pair, and record the estimated probability that it
 is connected, disconnected, and the difference between the two
-Saving the list of each pair and its estimated probabilities using pickle
"""

from ontology_parser import load_data
from graph_methods import build_graph
from probability_estimator import * # Both methods from the estimator are needed.
import random
import signal
import sys
import cPickle as pickle # For Python 2.7
# import _pickle as pickle # For Python 3

class GracefulKiller:
    """
    Safety measure to ensure that, even if the system kills the program,
    the generated data is not lost.
    Source: https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
    """
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self,signum, frame):
        self.kill_now = True

def test_probability_estimator_connected(phenotypes_ids, genotypes_ids, genes_ids,
                                         phenotypes_links, genotypes_links,
                                         phenotypes_genes_links, genotypes_genes_links,
                                         total_results_path_conn='../results/total_results.pkl',
                                         list_elems_path_conn='../results/list_elems.pkl',
                                         type_index_path_conn='../results/type_index.pkl',
                                         total_results_path_disc='../results/total_results_disc.pkl',
                                         list_elems_path_disc='../results/list_elems_disc.pkl',
                                         type_index_path_disc='../results/type_index_disc.pkl',
                                         number_of_pairs=1000):
    """
    Tests the accuracy of the probability estimator, using phenotype-genotype
    pairs that are certified to be connected, but that are not in the training set.

    Args:
        -phenotypes_ids: List of phenotypes
            +Type: list[str]
        -genotypes_ids: List of genotypes
            +Type: list[str]
        -genes_ids: List of genes
            +Type: list[str]
        -phenotypes_links: List of phenotype-phenotype links
            +Type: list[(str,str)]
        -genotypes_links: List of genotype-genotype links
            +Type: list[(str,str)]
        -phenotypes_genes_links: List of phenotype-genes links
            +Type: list[(str,str)]
        -genotypes_genes_links: List of genotype-genes links
            +Type: list[(str,str)]
        -total_results_path_conn: path to partial total_results for connected pairs
            +Type: str
        -list_elems_path_conn: path to partial list_elems for connected pairs
            +Type: str
        -type_index_path_conn: path to partial type_index for connected pairs
            +Type: str
        -total_results_path_disc: path to partial total_results for disconnected pairs
            +Type: str
        -list_elems_path_disc: path to partial list_elems for disconnected pairs
            +Type: str
        -type_index_path_disc: path to partial type_index for disconnected pairs
            +Type: str
        -number_of_pairs: number of genotype-phenotype pairs to test
            +Type: int
    Returns:
        None. Persists data.
    """
    list_elems = pickle.load(open(list_elems_path_conn, 'rb'))
    graph = build_graph(phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_links, genotypes_genes_links)
    statistics_conn = get_stats_from_alternative_paths(total_results_path_conn, type_index_path_conn, list_elems_path_conn)
    statistics_disc = get_stats_from_alternative_paths(total_results_path_disc, type_index_path_disc, list_elems_path_disc)
    probabilities_table = []
    prob_conn = prob_disc = prob_diff = 0
    itteration = 0
    loop_killer = GracefulKiller()

    try:
        # Loop until sufficient number of pairs have been tested
        while itteration < number_of_pairs:
            itteration += 1
            # Pick random phenotype
            p_id = random.choice(phenotypes_ids)
            # Get the list of linked genes
            p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0] == p_id]))
            # And the list of linked genotypes
            p_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
            # If the list of genotypes is empty, we start this itteration over
            if len(p_genotypes) == 0:
                itteration -= 1; continue

            # Pick a random genotype that is connected to the phenotype
            g_id = random.choice(p_genotypes)
            # If the selected pair was already in list_elems, we start this itteration over
            if (p_id, g_id) in list_elems:
                itteration -= 1; continue

            #Find the common genes of this pair
            common_genes = [gene for gene in genes_ids if (p_id,gene) in phenotypes_genes_links \
                            and (g_id,gene) in genotypes_genes_links]
            #Remove the links directly linking the source phenotype and the target genotype
            phenotypes_genes_pruned_links = [i for i in phenotypes_genes_links \
                                             if i[0]!= p_id or i[1] not in common_genes]
            graph = build_graph(phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_pruned_links, genotypes_genes_links)

            # Estimate probabilities the phenotype-genotype pair
            prob_conn, prob_disc = estimate_probabilities(graph, common_genes, (p_id, g_id), statistics_conn, statistics_disc)
            prob_diff = prob_conn - prob_disc
            print (p_id, g_id), prob_conn, prob_disc, prob_diff
            # Append the results to the probabilities_table
            probabilities_table.append(((p_id, g_id), prob_conn, prob_disc, prob_diff))

            # Exit the loop if the program is killed
            if loop_killer.kill_now:
                break
        raise KeyboardInterrupt

    # Code executed at the end of the function, or if the program is interrupted/killed
    except KeyboardInterrupt:
        # Sort probabilities_table by probability difference
        probabilities_table = sorted(probabilities_table, key=lambda x: x[3])
        pickle.dump(probabilities_table, open("../results/estimated_probs_conn.pkl", "wb"))


def test_probability_estimator_disconnected(phenotypes_ids, genotypes_ids, genes_ids,
                                            phenotypes_links, genotypes_links,
                                            phenotypes_genes_links, genotypes_genes_links,
                                            total_results_path_conn='../results/total_results.pkl',
                                            list_elems_path_conn='../results/list_elems.pkl',
                                            type_index_path_conn='../results/type_index.pkl',
                                            total_results_path_disc='../results/total_results_disc.pkl',
                                            list_elems_path_disc='../results/list_elems_disc.pkl',
                                            type_index_path_disc='../results/type_index_disc.pkl',
                                            number_of_pairs=1000):
    """
    Tests the accuracy of the probability estimator, using phenotype-genotype
    pairs that are certified to be disconnected, but that are not in the training set.

    Args:
        -phenotypes_ids: List of phenotypes
            +Type: list[str]
        -genotypes_ids: List of genotypes
            +Type: list[str]
        -genes_ids: List of genes
            +Type: list[str]
        -phenotypes_links: List of phenotype-phenotype links
            +Type: list[(str,str)]
        -genotypes_links: List of genotype-genotype links
            +Type: list[(str,str)]
        -phenotypes_genes_links: List of phenotype-genes links
            +Type: list[(str,str)]
        -genotypes_genes_links: List of genotype-genes links
            +Type: list[(str,str)]
        -total_results_path_conn: path to partial total_results for connected pairs
            +Type: str
        -list_elems_path_conn: path to partial list_elems for connected pairs
            +Type: str
        -type_index_path_conn: path to partial type_index for connected pairs
            +Type: str
        -total_results_path_disc: path to partial total_results for disconnected pairs
            +Type: str
        -list_elems_path_disc: path to partial list_elems for disconnected pairs
            +Type: str
        -type_index_path_disc: path to partial type_index for disconnected pairs
            +Type: str
        -number_of_pairs: number of genotype-phenotype pairs to test
            +Type: int
    Returns:
        None. Persists data.
    """
    list_elems = pickle.load(open(list_elems_path_disc, 'rb'))
    graph = build_graph(phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_links, genotypes_genes_links)
    statistics_conn = get_stats_from_alternative_paths(total_results_path_conn, type_index_path_conn, list_elems_path_conn)
    statistics_disc = get_stats_from_alternative_paths(total_results_path_disc, type_index_path_disc, list_elems_path_disc)
    probabilities_table = []
    prob_conn = prob_disc = prob_diff = 0
    itteration = 0
    loop_killer = GracefulKiller()

    try:
        # Loop until sufficient number of pairs have been tested
        while itteration < number_of_pairs:
            itteration += 1
            # Pick random phenotype
            p_id = random.choice(phenotypes_ids)
            # Get the list of linked genes
            p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0] == p_id]))
            # And the list of linked genotypes
            p_linked_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
            # From all genotypes, take away the linked genotypes to get the disconnected genotypes
            p_genotypes = list(set(genotypes_ids).difference(p_linked_genotypes))
            # If the list of genotypes is empty, we start this itteration over
            if len(p_genotypes) == 0:
                itteration -= 1; continue

            # Pick a random genotype that is disconnected to the phenotype
            g_id = random.choice(p_genotypes)
            # If the selected pair was already in list_elems, we start this itteration over
            if (p_id, g_id) in list_elems:
                itteration -= 1; continue

            # Estimate probabilities the phenotype-genotype pair
            prob_conn, prob_disc = estimate_probabilities(graph, [], (p_id, g_id), statistics_conn, statistics_disc)
            prob_diff = prob_conn - prob_disc
            print (p_id, g_id), prob_conn, prob_disc, prob_diff
            # Append the results to the probabilities_table
            probabilities_table.append(((p_id, g_id), prob_conn, prob_disc, prob_diff))

            # Exit the loop if the program is killed
            if loop_killer.kill_now:
                break
        raise KeyboardInterrupt

    # Code executed at the end of the function, or if the program is interrupted/killed
    except KeyboardInterrupt:
        # Sort probabilities_table by probability difference
        probabilities_table = sorted(probabilities_table, key=lambda x: x[3])
        pickle.dump(probabilities_table, open("../results/estimated_probs_disc.pkl", "wb"))

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# Load the ontology data into data structures
print "Loading data into structures..."
phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links = load_data('../data_files/hp.obo', '../data_files/go.obo', '../data_files/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', '../data_files/goa_human.gaf')
print "DONE!\n"

# Test probability estimator for random connected pairs
print "Testing estimator for connected pairs..."
test_probability_estimator_connected(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links)
print "DONE!\n"

# Test probability estimator for random disconnected pairs
print "Testing estimator for disconnected pairs..."
test_probability_estimator_disconnected(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links)
print "DONE!\n"
