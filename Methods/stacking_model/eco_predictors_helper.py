# Custom predictors implemented for food web link prediction
# Author: Lucy Van Kleunen

import numpy as np
import math

''' Implementation of ecologically adjusted Leight-Holme-Newman index

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values

Returns:
A list of predictor values for the given edges

'''
def eco_LHN(G, edges):
    LHN_list = []
    for (i,j) in edges:
        common_neigh_num = len(eco_common_neighbors(G,i,j))
        curr_pref_attach = eco_preferential_attachment(G,[(i,j)])[0][2]
        if curr_pref_attach == 0:
            curr_LHN = 0
        else:
            curr_LHN = round(common_neigh_num/curr_pref_attach,3)
        LHN_list.append(curr_LHN)
    return LHN_list

''' Implementation of ecologically adjusted Adamic Adar index

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values

Returns:
A list of predictor values for the given edges (with corresponding edge tuples)

'''
def eco_adamic_adar(G, edges):
    eco_adam_adar_obj = []
    for (i,j) in edges:
        common_neigh = eco_common_neighbors(G,i,j)
        curr_aa_score = 0
        for neigh in common_neigh:
            curr_aa_score += 1/math.log(G.degree(neigh))
        curr_aa_score = round(curr_aa_score,3)
        eco_adam_adar_obj.append((i,j,curr_aa_score))
    return eco_adam_adar_obj

''' Implementation of ecologically adjusted Resource Allocation index

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values

Returns:
A list of predictor values for the given edges (with corresponding edge tuples)

'''
def eco_resource_allocation(G, edges):
    eco_resource_allocation_obj = []
    for (i,j) in edges:
        common_neigh = eco_common_neighbors(G,i,j)
        curr_ra_score = 0
        for neigh in common_neigh:
            curr_ra_score += 1/G.degree(neigh)
        eco_resource_allocation_obj.append((i,j,curr_ra_score))
    return eco_resource_allocation_obj

''' Implementation of ecologically adjusted Jaccard coefficient

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values

Returns:
A list of predictor values for the given edges (with corresponding edge tuples)
'''
def eco_jaccard(G, edges):
    eco_jaccard_obj = []
    for (i,j) in edges:
        out_neighbor_set_i = get_out_neighbor_set(G,i)
        in_neighbor_set_j = get_in_neighbor_set(G,j)
        eco_common_neighbors = len(out_neighbor_set_i & in_neighbor_set_j)
        eco_total_neighbors = len(set(list(out_neighbor_set_i) + list(in_neighbor_set_j)))
        if eco_total_neighbors == 0:
            curr_jaccard = 0
        else:
            curr_jaccard = round(eco_common_neighbors/eco_total_neighbors,3)
        eco_jaccard_obj.append((i,j,curr_jaccard))
    return eco_jaccard_obj

''' Implementation of ecologically adjusted preferrential attachment 

Parameters:
G - network on which to calculate predictor values
edges - edge set over which to calculate predictor values

Returns:
A list of predictor values for the given edges (with corresponding edge tuples)

'''
def eco_preferential_attachment(G, edges):
    eco_pref_attach_obj = []
    for (i,j) in edges:
        out_neighbor_set_i = get_out_neighbor_set(G,i)
        in_neighbor_set_j = get_in_neighbor_set(G,j)
        curr_pref_attach = len(out_neighbor_set_i)*len(in_neighbor_set_j)
        eco_pref_attach_obj.append((i,j,curr_pref_attach))
    return eco_pref_attach_obj
   
''' Implementation of ecological common neighbor score calculation

Parameters:
G - network on which to calculate predictor values
edge_i, edge_j - edge for which to calculate count

Returns:
Count score for this edge

'''
def eco_common_neighbors_score(G,edge_i,edge_j):
    out_neighbor_set_i = get_out_neighbor_set(G,edge_i)
    in_neighbor_set_i = get_in_neighbor_set(G,edge_i)
    out_neighbor_set_j = get_out_neighbor_set(G,edge_j)
    in_neighbor_set_j = get_in_neighbor_set(G,edge_j)
    
    count_1 = len(out_neighbor_set_i & in_neighbor_set_j)
    count_2 = len(in_neighbor_set_i & out_neighbor_set_j)
    count_3 = len(out_neighbor_set_i & out_neighbor_set_j)
    count_4 = len(in_neighbor_set_i & in_neighbor_set_j)
    
    return count_1 - (count_2 + count_3 + count_4)

''' Implementation of ecologically adjusted common neighbors 

Parameters:
G - network on which to calculate predictor values
edge_i, edge_j - edge for which to find common neighbors

Returns:
Common neighbor set for given edge
'''
def eco_common_neighbors(G,edge_i,edge_j):
    out_neighbor_set_i = get_out_neighbor_set(G,edge_i)
    in_neighbor_set_j = get_in_neighbor_set(G,edge_j)
    return(list(out_neighbor_set_i & in_neighbor_set_j))


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

''' Helper function to perform min-max scaling on attribute dictionary

Parameters:
pos - attribute dictionary

Returns:
attribute dictionary after all values min-max scaled

'''
def min_max_norm(pos):
    pos_keys = list(pos.keys())
    if 'attr_names' in pos_keys:
        pos_keys.remove('attr_names')
    if 'attr_types' in pos_keys:
        pos_keys.remove('attr_types')
    num_nodes = len(pos_keys)
    num_attr = len(pos[pos_keys[0]])
    pos_array = np.zeros((num_nodes,num_attr))
    for v in range(0,num_nodes):
        pos_array[v,:] = pos[pos_keys[v]]
    mins = np.zeros(num_attr)
    maxs = np.zeros(num_attr)
    for a in range(0,num_attr):
        mins[a] = np.min(pos_array[:,a])
        maxs[a] = np.max(pos_array[:,a])
    pos_array_norm = np.zeros((num_nodes,num_attr))
    for v in range(0,num_nodes):
        new_row = []
        for a in range(0,num_attr):
            if mins[a] != maxs[a]: 
                normed = round((pos_array[v,a] - mins[a])/(maxs[a] - mins[a]),3)
                new_row.append(normed)
            else: # attribute all has the same value
                new_row.append(pos_array[v,a]) # just keep it the same
        pos_array_norm[v,:] = new_row
    pos_norm = {}
    for v in range(0,num_nodes):
        pos_norm[pos_keys[v]] = list(pos_array_norm[v,:])
    return pos_norm