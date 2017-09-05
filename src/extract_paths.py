from multiprocessing import Pool, cpu_count
import itertools
from graph_methods import * # All of the methods are used
import numpy as np
import cPickle as pickle # For Python 2.7
# import _pickle as pickle # For Python 3
import operator
import os
from os.path import exists
import random


def read_and_analyze_alternative_paths(total_results_path, type_index_path, list_elems_path):
    """
    Loads alternative paths stored in files and prints some statistics.

    Args:
        -total_results_path: Location of the file where paths are stored
            +Type: str
            +Pickle type: 2D numpy.array. Rows are phenotype-genotype pairs.
                Columns are path types
        -list_elems: Location of the file where list of computed pairs are stored.
                    Serves as row index.
            +Type: str
            +Pickle type: list[(phenotype_id ,genotype_id)]
        -type_index: Location of the file where the dictionary of path type
                    and column index is stored.
            +Type: str
            +Pickle type: dict{path_type, index}

    Returns:
        None. Prints data.

    """
    #Load the three structures
    #Type: 2D numpy.array. Rows are phenotype-genotype pairs.
    total_results = pickle.load(open(total_results_path,'rb'))
    #Type: dict{path_type, index}
    type_index = pickle.load(open(type_index_path,'rb'))
    #Type: list[(phenotype_id ,genotype_id)]
    list_elems = pickle.load(open(list_elems_path,'rb'))

    #Compute mean and stddev for each path type
    means = np.mean(total_results, axis=0)
    stddevs = np.std(total_results, axis=0)
    #Sort path types by index (i.e., dictonary value)
    sorted_types = sorted(type_index.items(), key=operator.itemgetter(1))
    #Print data
    print '-----------------------------'
    print 'Statistics for',len(list_elems),'phenotype-genotype pairs'
    print 'corresponding to',len(set([x[0] for x in list_elems])), 'unique phenotypes'
    print '-----------------------------'
    print "Path_type \t Mean \t StdDev"
    for elem in zip(sorted_types,means,stddevs):
        print elem[0][0],'\t ',elem[1],'\t ',elem[2]
    print '-----------------------------'


def persist_alternative_paths(total_results, list_elems, type_index,
                              total_results_path, list_elems_path,
                              type_index_path, continuing=False):
    """
    Persists partial results on disk using numpy

    Args:
        -total_results: paths stored
            +Type: 2D numpy.array. Rows are phenotype-genotype pairs.
                Columns are path types
        -list_elems: list of computed pairs. Serves as row index.
            +Type: list[(phenotype_id ,genotype_id)]
        -type_index: dictionary of path type and column index
            +Type: dict{path_type, index}
        -total_results_path: path for storing total_results
            +Type: str
        -list_elems_path: path for storing list_elems
            +Type: str
        -type_index_path: path for storing type_index
            +Type: str
        -continuing: Is this process a continuation of a partial execution?
            +Type: bool

    Returns:
        None. Persists data.
    """
    #TODO: files are currently overwritten. Consider when this may not be desirable
    #If we are continuing a previous partial execution, the previous pickle dump becomes a backup
    if continuing:
        backup_total_results_path = total_results_path + ".bak"
        backup_list_elems_path = list_elems_path + ".bak"
        backup_type_index_path = type_index_path + ".bak"
        #If there are already old backups, they are overwritten
        if exists(backup_total_results_path): os.remove(backup_total_results_path)
        if exists(backup_list_elems_path): os.remove(backup_list_elems_path)
        if exists(backup_type_index_path): os.remove(backup_type_index_path)
        #Turn the preexisting dump into the new backup
        os.rename(total_results_path, backup_total_results_path)
        os.rename(list_elems_path, backup_list_elems_path)
        os.rename(type_index_path, backup_type_index_path)
    #We then pickle dump the new results
    pickle.dump(total_results, open(total_results_path, "wb"))
    pickle.dump(list_elems, open(list_elems_path, "wb"))
    pickle.dump(type_index, open(type_index_path, "wb"))


def merge_alternative_paths(total_results, list_elems, type_index, partial_list):
    """
    Given a set of paths already processed, and a new set, merge both.

    Args:
        -total_results: previous paths stored
            +Type: 2D numpy.array. Rows are phenotype-genotype pairs.
                Columns are path types
        -list_elems: list of computed pairs. Serves as row index.
            +Type: list[(phenotype_id ,genotype_id)]
        -type_index: dictionary of path type and column index
            +Type: dict{path_type, index}
        -partial_list: new pairs data to be added
            +Type: (phenotype_id, genotype_id, dict{path_type, frequency})

    Returns:
        -total_results: merged paths. added rows (one per pair)
            and maybe some columns (if new path type found)
            +Type: 2D numpy.array. Rows are phenotype-genotype pairs.
                Columns are path types
        - list_elems: updated list of computed pairs (one new per pair)
            +Type: list[(phenotype_id ,genotype_id)]
        -type_index: dictionary of path type and column index.
            one new entry per new type of path
            +Type: dict{path_type, index}
    """
    #For each pair processed
    for current_paths in partial_list:
        #Get the pair in question
        p_id = current_paths[0]
        g_id = current_paths[1]
        #Append to the list
        list_elems.append((p_id, g_id))
        #Initialize values to zero
        current_np = np.zeros([len(type_index.keys())])
        #For each path
        paths = current_paths[2]
        for k,v in paths.iteritems():
            #If the type of path already exists
            if k in type_index.keys():
                #Set the value
                current_np[type_index[k]] = v
            #New type of path
            else:
                #Assign an index
                type_index[k] = total_results.shape[1]
                #Add a column of zeros to the total
                empty_col = np.zeros([total_results.shape[0],1])
                total_results = np.hstack((total_results,empty_col))
                #Add the value to current case
                current_np = np.append(current_np,v)
        total_results = np.vstack((total_results,current_np))
    return total_results, list_elems, type_index


def get_connected_phenotype_genotype_alternative_paths(phenotypes_ids,\
        genotypes_ids, genes_ids, phenotypes_links, genotypes_links,\
        phenotypes_genes_links, genotypes_genes_links, continuing=False,
        total_results_path = '../results/total_results.pkl',
        list_elems_path = '../results/list_elems.pkl',
        type_index_path = '../results/type_index.pkl'):
    """
    Given a list of genotypes and phenotypes, which may be linked through genes,
    for every linked genotype-phenotype pair find all the alternative paths
    when removing the linking gene/s.
    WARNING: This method is parallelized and will use all available CPUs.
    WARNING: This method takes a while to compute (i.e., probably weeks).
    For this reason, results are stored periodically on disc, and these are not returned.

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
        -continuing: Is this process a continuation of a partial execution?
            +Type: bool
        -total_results_path: path to partial and previous total_results
            +Type: str
        -list_elems_path: path to partial and previous list_elems
            +Type: str
        -type_index_path: path to partial and previous type_index
            +Type: str

    Returns:
        None (see previous Warning)
    """
    #Initialize structures
    if not continuing:
        list_elems = []
        type_index = {}
        total_results = np.empty([0,0])
    #If we are continuing a previous partial execution
    #load the precomputed values
    else:
        list_elems = pickle.load(open(list_elems_path,'rb'))
        type_index = pickle.load(open(type_index_path,'rb'))
        total_results = pickle.load(open(total_results_path,'rb'))
        print 'Continuing from a previous computation'
        print 'Total elements pre-computed:',len(list_elems)
        print 'Total num. of different paths pre-found:',len(type_index)
        print 'Data matrix shape:',total_results.shape
    #For each phenotype
    for p_id in phenotypes_ids:
        #Get the list of linked genes
        p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0]==p_id]))
        #And the list of linked genotypes
        if not continuing:
            p_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
        #If we are continuing a previous partial exeucution avoid doing the pre-computed pairs
        else:
            p_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes
                and (p_id,i[0]) not in list_elems]))
        #However unlikely, there may be no connected genotypes with the current phenotype
        if len(p_genotypes)==0: continue
        #Launch the computation for each linked genotype
        #pool = Pool(cpu_count())
        pool = Pool(2)
        print 'Going to compute',len(p_genotypes),'connected genotypes'
        partial_list = pool.map(find_phenotype_genotype_alternative_paths,\
                itertools.izip(itertools.repeat(p_id), p_genotypes,\
                itertools.repeat(phenotypes_ids), itertools.repeat(genotypes_ids),\
                itertools.repeat(genes_ids), itertools.repeat(phenotypes_links),\
                itertools.repeat(genotypes_links), itertools.repeat(phenotypes_genes_links),\
                itertools.repeat(genotypes_genes_links)))
        pool.close()
        pool.join()
        #Merge list with previous results
        total_results, list_elems, type_index = merge_alternative_paths(total_results, list_elems, type_index, partial_list)
        #Persist partial results
        persist_alternative_paths(total_results, list_elems, type_index,
                                  total_results_path, list_elems_path,
                                  type_index_path, continuing)
    return


def get_disconnected_phenotype_genotype_paths(phenotypes_ids,\
        genotypes_ids, genes_ids, phenotypes_links, genotypes_links,\
        phenotypes_genes_links, genotypes_genes_links, continuing=False, 
        total_results_path = '../results/total_results_disc.pkl',
        list_elems_path = '../results/list_elems_disc.pkl',
        type_index_path = '../results/type_index_disc.pkl'):
    """
    Given a list of genotypes and phenotypes, find all the paths
    for every genotype-phenotype pair not linked through a gene.
    WARNING: This method is parallelized and will use all available CPUs.
    WARNING: This method takes a while to compute (i.e., probably weeks).
    For this reason, results are stored periodically on disc, and these are not returned.

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
        -continuing: Is this process a continuation of a partial execution?
            +Type: bool
        -total_results_path: path to partial and previous total_results
            +Type: str
        -list_elems_path: path to partial and previous list_elems
            +Type: str
        -type_index_path: path to partial and previous type_index
            +Type: str

    Returns:
        None (see previous Warning)
    """
    #Initialize structures
    if not continuing:
        list_elems = []
        type_index = {}
        total_results = np.empty([0,0])
    #If we are continuing a previous partial execution
    #load the precomputed values
    else:
        list_elems = pickle.load(open(list_elems_path,'rb'))
        type_index = pickle.load(open(type_index_path,'rb'))
        total_results = pickle.load(open(total_results_path,'rb'))
        print 'Continuing from a previous computation'
        print 'Total elements pre-computed:',len(list_elems)
        print 'Total num. of different paths pre-found:',len(type_index)
        print 'Data matrix shape:',total_results.shape
    #For each phenotype
    for p_id in phenotypes_ids:
        #Get the list of linked genes
        p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0]==p_id]))
        #And the list of unlinked genotypes
        #...but first, find the linked genotypes
        p_linked_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
        #keep the rest
        p_genotypes = list(set(genotypes_ids).difference(p_linked_genotypes))
        #If we are continuing a previous partial execution avoid the already computed pairs
        if continuing:
            #Remove the already computed ones
            p_genotypes = [x for x in p_genotypes if (p_id,x) not in list_elems]
        #However unlikely, there may be no disconnected genotypes with the current phenotype
        if len(p_genotypes)==0: continue
        #Launch the computation for each linked genotype
        print 'Going to compute',len(p_genotypes),'disconnected genotypes'
        #pool = Pool(cpu_count())
        pool = Pool(2)
        partial_list = pool.map(find_phenotype_genotype_alternative_paths,\
                itertools.izip(itertools.repeat(p_id), p_genotypes,\
                itertools.repeat(phenotypes_ids), itertools.repeat(genotypes_ids),\
                itertools.repeat(genes_ids), itertools.repeat(phenotypes_links),\
                itertools.repeat(genotypes_links), itertools.repeat(phenotypes_genes_links),\
                itertools.repeat(genotypes_genes_links)))
        pool.close()
        pool.join()
        #Merge list with previous results
        total_results, list_elems, type_index = merge_alternative_paths(total_results, list_elems, type_index, partial_list)
        #Persist partial results
        persist_alternative_paths(total_results, list_elems, type_index,
                                  total_results_path, list_elems_path,
                                  type_index_path, continuing)
    return


def get_pair_dataset(phenotypes_ids, genotypes_ids,
        genes_ids, phenotypes_links, genotypes_links,
        phenotypes_genes_links, genotypes_genes_links, set_size,
        excluded_pairs=set()):
    """
    Given a list of genotypes and phenotypes, builds a random set of
    genotype-phenotype pairs, with a specific size and an equal number of
    connected and disconnected pairs. An additional set of excluded pairs can
    be added to prevent certain pairs from appearing in the new set.
    
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
        -set_size: The new set's size
            +Type: int
        -excluded_pairs: A set of pairs to be excluded
            +Type: set{(str,str)}
    Returns:
        -new_numpy_set: A numpy array representing a set of random pairs,
            contains information on wether they are linked, and their alternate paths
            +Type: np.array[[str,str,bool,dict{str,int}]]
    """
    new_set_as_list = [] # The new set currently being built, as a list
    new_set_exclude = set() # The new set as an 'excluded_pairs' set (just the pairs)
    
    # Step 1: get all the pairs that will be used in the set.
    print '(1) Building',set_size,'random pairs.'
    pair_count = 0
    new_element = []
    #We count the number of pairs until achieving the fixed (training set) size
    while pair_count < set_size:
        #Get a random phenotype
        p_id = random.choice(phenotypes_ids)
        #Get the list of linked genes
        p_genes = list(set([i[1] for i in phenotypes_genes_links if i[0]==p_id]))
        #Get the list of the linked genotypes
        p_genotypes = list(set([i[0] for i in genotypes_genes_links if i[1] in p_genes]))
        
        #If pair_count is even, the new pair will be connected, else it is disconnected.
        if pair_count%2 == 0:
            #However unlikely, there may be no disconnected genotypes with the current phenotype
            if len(p_genotypes)==0: continue
            g_id = random.choice(p_genotypes)
            #Avoid the pair if it was already added, or must be excluded
            if (p_id, g_id) in new_set_exclude|excluded_pairs: continue
            new_element = [p_id, g_id, True]
        else:
            #Deduce the list of unlinked genotypes
            p_unlinked_genotypes = list(set(genotypes_ids).difference(p_genotypes))
            #However unlikely, there may be no disconnected genotypes with the current phenotype
            if len(p_genotypes)==0: continue
            g_id = random.choice(p_genotypes)
            #Avoid the pair if it was already added, or must be excluded
            if (p_id, g_id) in new_set_exclude|excluded_pairs: continue
            new_element = [p_id, g_id, False]
        
        # Append new element to set
        new_set_as_list.append(new_element)
        new_set_exclude.add((p_id, g_id))
        pair_count += 1
    
    # Step 2: find the alternate paths for these pairs
    #pool = Pool(cpu_count())
    pool = Pool(2)
    print '(2) Computing paths for',set_size,'pairs.'
    paths_list = pool.map(find_phenotype_genotype_alternative_paths,\
                itertools.izip([x[0] for x in new_set_as_list],\
                [x[1] for x in new_set_as_list],\
                itertools.repeat(phenotypes_ids), itertools.repeat(genotypes_ids),\
                itertools.repeat(genes_ids), itertools.repeat(phenotypes_links),\
                itertools.repeat(genotypes_links), itertools.repeat(phenotypes_genes_links),\
                itertools.repeat(genotypes_genes_links)))
    pool.close()
    pool.join()
    # Append each path dictionnary to the corresponding pair in the set.
    for i in range(set_size):
        new_set_as_list[i].append(paths_list[i][2])
        # new_set_as_list[i] = tuple(new_set_as_list[i])
    
    # Step 3: turn the list of pairs into an actual set, print out some pairs
    print '(3) Sample pairs from newly built set:'
    new_numpy_set = np.array(new_set_as_list)
    sample_rows = np.random.choice(range(set_size), size=min(set_size, 10))
    for row in sample_rows:
        print new_numpy_set[row]
    return new_numpy_set


def get_all_pair_datasets_for_NN(phenotypes_ids, genotypes_ids,
        genes_ids, phenotypes_links, genotypes_links,
        phenotypes_genes_links, genotypes_genes_links, tr_size=1000,
        training_set_path = '../neural_net/training_set.npy',
        validation_set_path = '../neural_net/validation_set.npy',
        testing_set_path = '../neural_net/testing_set.npy'):
    """
    Given a list of genotypes and phenotypes, select three sets of
    genotype-phenotype pairs, each with equal number of connected and
    disconnected pairs. These sets are stored as numpy binary files (.npy),
    and meant to be used for training a classifier based on a neural network.
    They are treated respectively as the network's training (size = tr_size),
    validating (size = 0.25*tr_size), and testing sets (size = 0.15*tr_size).
    
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
        -tr_size: Training set size (other sets are proportional to it)
            +Type: int
        -training_set_path: Location of the file where the training set is stored
            +Type: str
        -validation_set_path: Location of the file where the validation set is stored
            +Type: str
        -testing_set_path: Location of the file where the testing set is stored
            +Type: str
    Returns:
        None. Persists data.
    """
    # Built from smallest to largest, as pairs included in previous sets
    # must be excluded from newly built sets.
    print "Building testing set..."
    testing_set = get_pair_dataset(phenotypes_ids, genotypes_ids,
            genes_ids, phenotypes_links, genotypes_links,
            phenotypes_genes_links, genotypes_genes_links, int(round(tr_size*0.15)))
    np.save(testing_set_path, testing_set)
    print "----------------------------------------------------------------"
    print "Building validation set..."
    excluded = set([(x[0], x[1]) for x in testing_set])
    validation_set = get_pair_dataset(phenotypes_ids, genotypes_ids,
            genes_ids, phenotypes_links, genotypes_links,
            phenotypes_genes_links, genotypes_genes_links, int(round(tr_size*0.25)),
            excluded_pairs=excluded)
    np.save(validation_set_path, validation_set)
    print "----------------------------------------------------------------"
    print "Building training set..."
    excluded = excluded|set([(x[0], x[1]) for x in validation_set])
    training_set = get_pair_dataset(phenotypes_ids, genotypes_ids,
            genes_ids, phenotypes_links, genotypes_links,
            phenotypes_genes_links, genotypes_genes_links, tr_size,
            excluded_pairs=excluded)


def find_phenotype_genotype_alternative_paths(argv):
#def find_paths(p_id, g_id, phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_links, genotypes_genes_links):
    #TODO: Due to multiprocessing, parameters are passed within a list
    #TODO: and unpacked here. This can be probably fixed.
    """
    Find all paths between a phenotype and a genotype which share a gene.
    Return the number and type of paths without using the phenotype-shared_gene link.

    Args:
        -p_id: Phenotype source of the path
            +Type: str
        -g_id: Genotype target of the path
            +Type: str
        -phenotypes_ids: List of phenotypes to be used as vertices
            +Type: list[str]
        -genotypes_ids: List of genotypes to be used as vertices
            +Type: list[str]
        -genes_ids: List of genes to be used as vertices
            +Type: list[str]
        -phenotypes_links: List of phenotype-phenotype links to be used as edges
            +Type: list[(str,str)]
        -genotypes_links: List of genotype-genotype links to be used as edges
            +Type: list[(str,str)]
        -phenotypes_genes_links: List of phenotype-genes links to be used as edges
            +Type: list[(str,str)]
        -genotypes_genes_links: List of genotype-genes links to be used as edges
            +Type: list[(str,str)]

    Returns:
        -paths_codes: Dictionary containing all path types and their frequency
            +Type: dict{(str,int)}
    """
    p_id = argv[0]
    g_id = argv[1]
    phenotypes_ids = argv[2]
    genotypes_ids = argv[3]
    genes_ids = argv[4]
    phenotypes_links = argv[5]
    genotypes_links = argv[6]
    phenotypes_genes_links = argv[7]
    genotypes_genes_links = argv[8]
    #Find the common genes of this pair
    common_genes = [gene for gene in genes_ids if (p_id,gene) in phenotypes_genes_links \
            and (g_id,gene) in genotypes_genes_links]
    #Remove the links directly linking the source phenotype and the target genotype
    phenotypes_genes_pruned_links = [i for i in phenotypes_genes_links \
            if i[0]!= p_id or i[1] not in common_genes]
    #Create the graph
    #TODO: add directionality of graph as parameter
    graph = build_graph(phenotypes_ids, genotypes_ids, genes_ids, phenotypes_links, genotypes_links, phenotypes_genes_pruned_links, genotypes_genes_links, undirected=True)
    #Find all paths
    paths = find_all_paths(graph, graph.vs.find(p_id).index, graph.vs.find(g_id).index,maxlen=4)
    #Compute the type and frequency of all paths
    paths_codes = {}
    for current_path in paths:
        current_code  = get_phenotype_genotype_path_code(graph,current_path,p_id,common_genes,g_id)
        if current_code in paths_codes:
            paths_codes[current_code]+=1
        else:
            paths_codes[current_code]=1
    #Return the pair and their alternative paths
    return p_id,g_id,paths_codes