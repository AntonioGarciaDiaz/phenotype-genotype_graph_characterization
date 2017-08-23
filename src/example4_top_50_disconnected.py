"""
This file is similar to example 3, it shows an example on how to:
-Load the ontology data into a graph (as in example 1)
-Load statistics concerning different path types linking (dis)connected pairs
-Generate various disconnected pairs that were not part of the training set
 used to generate the statistics
-Use the estimator on each pair, and keep a record of the top 50 pairs with
 highest estimated probability of being connected
-Save the top 50 list (each pair and its estimated probability) using pickle
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


class LinkedList_Node:
    """
    A node for the top 50 sorted chained list.
    Source: https://gist.github.com/ptigas/2820165
    """
    def __init__(self, pair):
        self.pair = pair
        self.next = None


class LinkedList:
    """
    A linked list object of maximum length, sorted on a priority value.
    Used for representing the list of the top 50 disconnected pairs with a
    higher estimated probability of being connected (with max length 50).
    Source: https://gist.github.com/ptigas/2820165
    """
    def __init__(self, max_len):
        """
        Initialize an empty linked list.
        """
        self.head = None
        self.max_len = max_len

    def add(self, newPair):
        """
        Add a new element to the linked list in the appropriate position.
        Returns true if the element was indeed added, false otherwise.
        """
        new_node = LinkedList_Node(newPair)
        curr_node = self.head # Current node starts at the head.
        curr_pos = 1 # Head is at position 1, last possible position is max_len
        pair_was_added = False # Wether the pair was added or not
        
        # If the list is empty, or the candidate is better than the head,
        # the candidate becomes the head.
        if (self.head is None) or (self.head.pair[1] < new_node.pair[1]):
            new_node.next = self.head
            self.head = new_node
            curr_node = self.head
            pair_was_added = True
            
        # Else check if the candidate should be added on a later position.
        else:
            # While there is a next node, and it is better than candidate.
            while (curr_node.next is not None) and (curr_node.next.pair[1] >= new_node.pair[1]):
                curr_node = curr_node.next
                curr_pos += 1
            # At this point, next node is either worse than candidate, or None.
            # If we haven't yet reached the max length, add the candidate node.
            if curr_pos < self.max_len:
                new_node.next = curr_node.next
                curr_node.next = new_node
                pair_was_added = True
                
        # In any case, if added, find the last node (limited by max length)
        if pair_was_added:
            while (curr_pos < self.max_len) and (curr_node.next is not None):
                curr_node = curr_node.next
                curr_pos += 1
            # Unreference any nodes that may follow (next node becomes None)
            if curr_pos == self.max_len: curr_node.next = None
        
        return pair_was_added


def get_list(top_50):
    """
    Outputs all the elements of a linked list object as a regular python list.

    Args:
        -top_50: A linked list object representing the top 50 disconnected pairs.
            +Type: LinkedList
    Returns:
        -output_list: The (ordered) items of top_50, in the form of a python list.
            +Type: list[((p_id, g_id), prob_conn, prob_disc)]
    """
    output_list = []
    curr_node = top_50.head
    while curr_node is not None:
        output_list.append(curr_node.pair)
        curr_node = curr_node.next
    return output_list

def find_top_50_pairs(phenotypes_ids, genotypes_ids, genes_ids,
                      phenotypes_links, genotypes_links,
                      phenotypes_genes_links, genotypes_genes_links,
                      total_results_path_conn='../results/total_results.pkl',
                      list_elems_path_conn='../results/list_elems.pkl',
                      type_index_path_conn='../results/type_index.pkl',
                      total_results_path_disc='../results/total_results_disc.pkl',
                      list_elems_path_disc='../results/list_elems_disc.pkl',
                      type_index_path_disc='../results/type_index_disc.pkl'):
    """
    Selects random phenotype-genotype pairs that are disconnected, until the
    program is killed or stopped with Ctrl+C, and stores in a list the top
    50 pairs with a highest estimated probability of being connected.
    This is useful to detect pairs which may be in fact connected, but are
    reported as disconnected because the gene that directly links them has
    not yet been found.

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
    Returns:
        None. Persists data.
    """
    #List of disconnected pairs used to compute the statistics
    list_elems = pickle.load(open(list_elems_path_disc, 'rb'))
    #Build graph with all the data
    graph = build_graph(phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_links, genotypes_genes_links)
    #Compute statistics for connected pairs
    statistics_conn = get_stats_from_alternative_paths(total_results_path_conn, type_index_path_conn, list_elems_path_conn)
    #Compute statistics for disconnected pairs
    statistics_disc = get_stats_from_alternative_paths(total_results_path_disc, type_index_path_disc, list_elems_path_disc)
    #Create an empty linked list to store results
    top_50 = LinkedList(50)
    top_50_list = get_list(top_50)
    prob_conn = prob_disc = 0
    loop_killer = GracefulKiller()

    try:
        # Loop until death
        while True:
            # Pick random phenotype
            p_id = random.choice(phenotypes_ids)
            # Get the list of linked genes
            p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0] == p_id]))
            # And then the list of linked genotypes
            p_linked_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
            # From all genotypes, take away the linked genotypes to get the disconnected genotypes
            p_genotypes = list(set(genotypes_ids).difference(p_linked_genotypes))
            # However unlikely, a case where no disconnected genotypes are found must be treated
            if len(p_genotypes) == 0: continue

            # Pick a random genotype that is disconnected to the phenotype
            g_id = random.choice(p_genotypes)
            # If the selected pair was used to compute the statistics, skip it
            if (p_id, g_id) in list_elems: continue
            # If the pair is already in the list of top pairs, skip it
            if (p_id, g_id) in [x[0] for x in top_50_list]: continue

            # Estimate probabilities the phenotype-genotype pair
            prob_conn, prob_disc = estimate_probabilities(graph, [], (p_id, g_id), statistics_conn, statistics_disc)
            # Try to add the new pair to the list of top pairs
            pair_was_added = top_50.add(((p_id, g_id), prob_conn, prob_disc))
            
            # Exit the loop if the program is killed
            if loop_killer.kill_now: break
            
            # If the element was added, print and save the list to disk
            if pair_was_added:
                top_50_list = get_list(top_50)
                pickle.dump(top_50_list, open("../results/top_50_disc_pairs_list.pkl", "wb"))
                print "----------------------------------------------------------------------"
                for i in top_50_list: print i
                print "----------------------------------------------------------------------"

        raise KeyboardInterrupt

    # Code executed at the end of the function, or if the program is interrupted/killed
    except KeyboardInterrupt:
        pickle.dump(get_list(top_50), open("../results/top_50_disc_pairs_list.pkl", "wb"))

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


# Load the ontology data into data structures
print "Loading data into structures..."
phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links = load_data('../data_files/hp.obo', '../data_files/go.obo', '../data_files/ALL_SOURCES_ALL_FREQUENCIES_genes_to_phenotype.txt', '../data_files/goa_human.gaf')
print "DONE!\n"

# Find the top 50 most likely disconnected pairs to be in fact connected
print "Finding the top 50 disconnected pairs..."
find_top_50_pairs(phenotypes, genotypes, genes, ph_ph_links, go_go_links, ph_gn_links, go_gn_links)
print "DONE!\n"
