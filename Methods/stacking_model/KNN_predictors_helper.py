# Helper functions for the KNN predictors
# Author: Lucy Van Kleunen

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import distance
from itertools import combinations
import random

''' Implementation of getting the out neighbor set of a node

Parameters:
G - network on which to calculate predictor values
node - node for which to get the out neighbor set

Returns:
the out neighbor set for the node
'''
def get_out_neighbor_set(G,node):
    out_neighbor_it = G.successors(node)
    out_neighbor_set = []
    for x in out_neighbor_it:
        out_neighbor_set.append(x)
    return set(out_neighbor_set)

''' Implementation of getting the in neighbor set of a node

Parameters:
G - network on which to calculate predictor values
node - node for which to get the in neighbor set

Returns:
the in neighbor set for the node
'''
def get_in_neighbor_set(G,node):
    in_neighbor_it = G.predecessors(node)
    in_neighbor_set = []
    for x in in_neighbor_it:
        in_neighbor_set.append(x)
    return set(in_neighbor_set)


''' Implementation of calculating KNN predictors based on node attribute distances

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values
norm_attr_dict - attribute vecors (already min-max normalized)
K - number of nearest neighbors to look at 
norm - Minkowski p-norm used for distance calculation, 1=Manhattan, 2=Euclidean
details - printing details or not

Returns:
KNN_in - list of KNN predictors in the in direction corresponding to the edge set
KNN_out - list of KNN predictors in the out direction corresponding to the edge set

'''
def KNN_predictor(G, edges, norm_attr_dict, K, norm, details):
            
    nodes = list(G.nodes())
    
    n = len(nodes) # number of data points == number of nodes 
    m = len(norm_attr_dict[nodes[0]]) # number of attributes per node
    
    index_id_map = {}
    d = np.zeros((n,m))
    
    for i in range(0, n):
        nodei = nodes[i]
        d[i][:] = norm_attr_dict[nodei]
        index_id_map[i] = nodei
            
    if details:
        print(index_id_map)
        print(d)
    
    tree = cKDTree(d)
    
    KNN_in = []
    KNN_out = []
    
    for edge in edges:
        
        edge_i = edge[0]
        edge_j = edge[1]
        
        if details:
            print(f"edge_i : {edge_i}")
            print("norm attr dict:")
            print(norm_attr_dict[edge_i])
        
        # Out direction i to j
        
        # Get the K nearest neighbors to node i (add 1 to exclude the node itself)
        _, ii = zip(*sorted(zip(*tree.query(norm_attr_dict[edge_i], k=list(range(1,K+2)), p=norm)), key=lambda x: x[0]))
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_i: # ignore self
                out_neighbors = get_out_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("out neighbors:")
                    print(out_neighbors)

                if edge_j in out_neighbors:
                    ct += 1
        KNN_out_curr = ct/K
        KNN_out.append(KNN_out_curr)
        
        # In direction j to i
        
        if details:
            print(f"edge_j : {edge_j}")
            print("norm attr dict:")
            print(norm_attr_dict[edge_j])
        
        # Get the K nearest neighbors to node j (add 1 to exclude the node itself)
        _, ii = zip(*sorted(zip(*tree.query(norm_attr_dict[edge_j], k=list(range(1,K+2)), p=norm)), key=lambda x: x[0]))
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_j: # ignore self
                in_neighbors = get_in_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("in neighbors:")
                    print(in_neighbors)

                if edge_i in in_neighbors:
                    ct += 1
        KNN_in_curr = ct/K
        KNN_in.append(KNN_in_curr)
    
    return KNN_in, KNN_out

''' Implementation of Jaccard similarity calculation, in the in direction

Parameters:
G - network on which to calculate predictor values
node_i, node_j - nodes for which to calculate the similarity score

Returns:
The Jaccard similarity score for the two nodes based on in neighbor sets
'''
def jaccard_similarity_in(G, node_i, node_j):
        
    in_neighbor_set_i = get_in_neighbor_set(G, node_i)
    in_neighbor_set_j = get_in_neighbor_set(G, node_j)
    
    intersection_in = in_neighbor_set_i & in_neighbor_set_j
    union_in = in_neighbor_set_i | in_neighbor_set_j
    
    if len(union_in) == 0: 
        # Edge case - neither node has in neighbors. Function is not defined in this case.
        # For our application, setting this to be 1 (as in the two sets are equivalent), as this will lead to a distance of 0
        # We would consider two nodes both with empty in neighbor sets to be similar in our application (e.g. basal nodes)
        return 1
    else:
        jaccard_sim = len(intersection_in)/len(union_in)
        return jaccard_sim

''' Implementation of Jaccard similarity calculation, in the out direction

Parameters:
G - network on which to calculate predictor values
node_i, node_j - nodes for which to calculate the similarity score

Returns:
The Jaccard similarity score for the two nodes based on out neighbor sets
'''
def jaccard_similarity_out(G, node_i, node_j):
        
    out_neighbor_set_i = get_out_neighbor_set(G, node_i)
    out_neighbor_set_j = get_out_neighbor_set(G, node_j)
    
    intersection_out = out_neighbor_set_i & out_neighbor_set_j
    union_out = out_neighbor_set_i | out_neighbor_set_j
    
    if len(union_out) == 0: 
        # Edge case - neither node has out neighbors. Function is not defined in this case.
        # For our application, setting this to be 1 (as in the two sets are equivalent), as this will lead to a distance of 0
        # We would consider two nodes both with empty in neighbor sets to be similar in our application (e.g. top predators)
        return 1
    else:
        jaccard_sim = len(intersection_out)/len(union_out)
        return jaccard_sim

''' Helper function to create a distance matrix (Jaccard in neighbors)

Parameters:
G - network on which to calculate predictor values
n - size (number of nodes)
index_id_map - map from matrix index to node id in network

Returns:
matrix of Jaccard distances in the in direction
'''
def create_distance_matrix_in(G, n, index_id_map):
    
    in_dist = np.full((n, n), -1, dtype = float)
    unique_pairs = list(combinations(range(0, n), 2))
        
    for i in range(0,n):
        in_dist[i,i] = 0 # diagonal set to 0
    
    for pair in unique_pairs:
        i = pair[0]
        j = pair[1]
        node_i = index_id_map[i]
        node_j = index_id_map[j]
        if i > j:
            jaccard = jaccard_similarity_in(G, node_i, node_j)
            in_dist[i,j] = 1 - jaccard
        if j > i:
            jaccard = jaccard_similarity_in(G, node_j, node_i)
            in_dist[j,i] = 1 - jaccard
    
    return in_dist

''' Helper function to create a distance matrix (Jaccard out neighbors)

Parameters:
G - network on which to calculate predictor values
n - size (number of nodes)
index_id_map - map from matrix index to node id in network

Returns:
matrix of Jaccard distances in the out direction
'''
def create_distance_matrix_out(G, n, index_id_map):
    
    out_dist = np.full((n, n), -1, dtype = float)
    unique_pairs = list(combinations(range(0, n), 2))
        
    for i in range(0,n):
        out_dist[i,i] = 0 # diagonal set to 0
    
    for pair in unique_pairs:
        i = pair[0]
        j = pair[1]
        node_i = index_id_map[i]
        node_j = index_id_map[j]
        if i > j:
            jaccard = jaccard_similarity_out(G, node_i, node_j)
            out_dist[i,j] = 1 - jaccard
        if j > i:
            jaccard = jaccard_similarity_out(G, node_j, node_i)
            out_dist[j,i] = 1 - jaccard
    
    return out_dist
    
''' Helper function to create a distance matrix based on binary attributes

Parameters:
G - network on which to calculate predictor values
n - size (number of nodes)
index_id_map - map from matrix index to node id in network

Returns:
matrix of Jaccard distances between binary attributes
'''
def create_distance_matrix_bin_attr(G, n, index_id_map, attr_dict):
    
    bin_dist = np.full((n, n), -1, dtype = float)
    unique_pairs = list(combinations(range(0, n), 2))
        
    for i in range(0,n):
        bin_dist[i,i] = 0 # diagonal set to 0
    
    for pair in unique_pairs:
        i = pair[0]
        j = pair[1]
        node_i = index_id_map[i]
        node_j = index_id_map[j]
        
        attr_i = attr_dict[node_i]
        attr_j = attr_dict[node_j]
        
        if i > j:
            bin_dist[i,j] = distance.jaccard(attr_i,attr_j)
        if j > i:
            bin_dist[j,i] = distance.jaccard(attr_i,attr_j)
    
    return bin_dist

''' Helper function to find a nearest neighbor for a node using the distance matrix

Parameters:
distance_matrix - current distance matrix
index - the index of the node that we want to find nearest neighbors for
skip - force skipping certain nodes

Returns:
the nearest neighbor for a node
'''
def find_NN_from_matrix(distance_matrix, index, skip):
        
    nn = None
    nn_dist = 10 # assuming all distance values between 0 and 1 (start high)

    # Scan through matrix to find lowest distance value
    row_i = distance_matrix[index, :]
    for col in range(0,len(row_i)):
        curr_dist = distance_matrix[index, col]
        # in this implementation, ignore the self as a nn option; ignore -1 values in upper triangle
        if col != index and curr_dist >= 0:
            if curr_dist < nn_dist and col not in skip:
                nn = col
                nn_dist = curr_dist

    col_i = distance_matrix[:,index]
    for row in range(0,len(col_i)):
        curr_dist = distance_matrix[row, index]
        if row != index and curr_dist >= 0:
            if curr_dist < nn_dist and row not in skip:
                nn = row
                nn_dist = curr_dist

    return nn, nn_dist

''' Helper function to get all the nodes at a certain distance from a node

Parameters:
distance_matrix - current distance matrix
index - the index of the node that we want to find nearest neighbors for
distance - the distance we want to find all the nodes at

Returns:
all the nodes at that distance from the starting node
'''
def get_all_at_distance(distance_matrix, index, distance):
    
    el = []

    # Scan through matrix to find elements at that distance
    row_i = distance_matrix[index, :]
    for col in range(0,len(row_i)):
        curr_dist = distance_matrix[index, col]
        if curr_dist == distance:
            el.append(col)

    col_i = distance_matrix[:,index]
    for row in range(0,len(col_i)):
        curr_dist = distance_matrix[row, index]
        if curr_dist == distance:
            el.append(row)

    return el

''' Helper function to find KNN from the distance matrix

Parameters:
distance_matrix - current distance matrix
index - the index of the node that we want to find nearest neighbors for
K - the number of nearest neighbors to find
sample - whether or not to randomly sample K from all options (otherwise return all)

Returns:
The K nearest neighbors
'''
def find_KNN_from_matrix(distance_matrix, index, K, sample):
           
    nn_indices = []
    nn_dists = []
    for x in range(0,K):
        next_nn, next_nn_dist = find_NN_from_matrix(distance_matrix, index, nn_indices)
        nn_indices.append(next_nn)
        nn_dists.append(next_nn_dist)
                
    # The above gives K, but there could be a lot more
    max_dist = np.max(nn_dists)
    other_options = get_all_at_distance(distance_matrix, index, max_dist)
    for other in other_options:
        if other not in nn_indices and other != index:
            nn_indices.append(other)
                
    # If there are more than K options, randomly sample K of the options to return
    if sample:
        random_indices = random.sample(range(0, len(nn_indices)),  K)
        final_K = [nn_indices[i] for i in random_indices]
    else:
        final_K = nn_indices
    return(final_K)

''' Implementation of KNN predictor where distance between nodes defined by their neighbor set overlap (Jaccard index)

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values
K - number of nearest neighbors to look at 
mode - calculate distances based on in or out neighbor sets
details - printing details or not

Returns:
KNN_in - list of KNN predictors in the in direction corresponding to the edge set
KNN_out - list of KNN predictors in the out direction corresponding to the edge set
'''
def KNN_predictor_neighborset(G, edges, K, mode, details):  
            
    nodes = list(G.nodes())
    n = len(nodes)
    
    # node id to index in distance matrix + back
    index_id_map = {}
    index_id_map_rev = {}
    for i in range(0, n):
        index_id_map[i] = nodes[i]
        index_id_map_rev[nodes[i]] = i
            
    if details:
        print(index_id_map)
    
    if mode == "in":
        distance_matrix = create_distance_matrix_in(G, n, index_id_map)
    elif mode == "out":
        distance_matrix = create_distance_matrix_out(G, n, index_id_map)
    else:
        print("incorrect mode provided to KNN neighbor set predictor")
        return -1
    
    if details:
        print(distance_matrix)
        
    KNN_out = []
    KNN_in = []

    for edge in edges:
        
        edge_i_id = edge[0]
        edge_j_id = edge[1]
        
        edge_i = index_id_map_rev[edge_i_id]
        edge_j = index_id_map_rev[edge_j_id]
        
        if details:
            print(f"edge_i_id : {edge_i_id}")
        
        # Out direction i to j
        
        # Get the K nearest neighbors to node i (note can be more if ties @ maximum KNN distance - sample=False)
        ii = find_KNN_from_matrix(distance_matrix, edge_i, K, False)
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_i_id: # ignore self
                out_neighbors = get_out_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("out neighbors:")
                    print(out_neighbors)

                if edge_j_id in out_neighbors:
                    ct += 1
        KNN_out_curr = ct/len(ii)
        KNN_out.append(KNN_out_curr)
                
        # In direction j to i
        
        if details:
            print(f"edge_j_id : {edge_j_id}")
        
        # Get the K nearest neighbors to node j (note can be more if ties @ maximum KNN distance - sample=False)
        ii = find_KNN_from_matrix(distance_matrix, edge_j, K, False)
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_j_id: # ignore self
                in_neighbors = get_in_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("in neighbors:")
                    print(in_neighbors)

                if edge_i_id in in_neighbors:
                    ct += 1
        KNN_in_curr = ct/len(ii)
        KNN_in.append(KNN_in_curr)
        
    return KNN_in, KNN_out

''' Implementation of KNN predictor where distance between nodes defined by the Jaccard distances between binary attributes

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values
attr_dict - binary attribute dictionary 
K - number of nearest neighbors to look at 
details - printing details or not

Returns:
KNN_in - list of KNN predictors in the in direction corresponding to the edge set
KNN_out - list of KNN predictors in the out direction corresponding to the edge set
'''
def KNN_predictor_jaccard_attr(G, edges, attr_dict, K, details):
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    # node id to index in distance matrix + back
    index_id_map = {}
    index_id_map_rev = {}
    for i in range(0, n):
        index_id_map[i] = nodes[i]
        index_id_map_rev[nodes[i]] = i
            
    if details:
        print(index_id_map)
    
    # Create distance matrix based on Jaccard distance between binary attribute vectors
    distance_matrix = create_distance_matrix_bin_attr(G, n, index_id_map, attr_dict)
    
    if details:
        print(distance_matrix)
        
    KNN_out = []
    KNN_in = []

    for edge in edges:
        
        edge_i_id = edge[0]
        edge_j_id = edge[1]
        
        edge_i = index_id_map_rev[edge_i_id]
        edge_j = index_id_map_rev[edge_j_id]
        
        if details:
            print(f"edge_i_id : {edge_i_id}")
        
        # Out direction i to j
        
        # Get the K nearest neighbors to node i (can be more than K if ties for max distance)
        ii = find_KNN_from_matrix(distance_matrix, edge_i, K, False)
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_i_id: # ignore self
                out_neighbors = get_out_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("out neighbors:")
                    print(out_neighbors)

                if edge_j_id in out_neighbors:
                    ct += 1
        KNN_out_curr = ct/len(ii)
        KNN_out.append(KNN_out_curr)
                
        # In direction j to i
        
        if details:
            print(f"edge_j_id : {edge_j_id}")
        
        # Get the K nearest neighbors to node j (can be more than K if ties for max distance)
        ii = find_KNN_from_matrix(distance_matrix, edge_j, K, False)
        
        if details:
            print(ii)
        
        ct = 0
        for index in ii:
            nn_id = index_id_map[index]
            if nn_id != edge_j_id: # ignore self
                in_neighbors = get_in_neighbor_set(G, nn_id)

                if details:
                    print(f"node: {nn_id}")
                    print("in neighbors:")
                    print(in_neighbors)

                if edge_i_id in in_neighbors:
                    ct += 1
        KNN_in_curr = ct/len(ii)
        KNN_in.append(KNN_in_curr)
        
    return KNN_in, KNN_out