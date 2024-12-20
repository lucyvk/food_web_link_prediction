# Helper functions for generating and saving the synthetic networks as well as calculating theoretical maximum ROC-AUC performance

import os
import networkx as nx
import random
from scipy.spatial import distance
import itertools 
import numpy as np
import pandas as pd
import sys
import eco_predictors_helper

# nodes - node list
# num_bin - number of random binary attributes to generate per node
# num_num - number of random numeric attributes to generate per node
# range_num - the respective range limits for each numeric attribute
def generate_random_attributes(nodes,num_bin,num_num,range_num):
    assert len(range_num) == num_num, "ranges for numeric attributes not the correct length"
    pos = {}
    for v in nodes:
        rand_pos = []
        if num_bin > 0:
            for i in range(0,num_bin):
                draw = random.random()
                if draw > 0.5:
                    rand_pos.append(1)
                else:
                    rand_pos.append(0)
        if num_num > 0:
            for i in range(0,num_num):
                draw = random.random()
                rand_num = range_num[i][0] + (range_num[i][1] - range_num[i][0])*draw
                rand_pos.append(rand_num)
        pos[v] = rand_pos
    return pos
    

# nodes - list of node ids
# num_num - number of numeric attributes
# num_bin - number of binary attributes
# range_num - ranges of numeric attributes
# SBM network properties - sbm_blocks, sbm_probs
# RGG network properties - rgg_kernel
# rec_prop - probability of a generated edge being pointed in both directions 
# alpha - level of mixing
# pos - used if you want to force the position vector to be certain values
def mixed_network_directed(nodes, num_bin, num_num, range_num, sbm_blocks, sbm_probs, rgg_kernel, rec_prop, alpha, pos):
        
    mixed_net = nx.DiGraph()
    mixed_net.add_nodes_from(nodes)
    
    if not pos:
        pos = generate_random_attributes(nodes,num_bin,num_num,range_num)
                             
    # scale positions before calculating distances
    normed_pos = eco_predictors_helper.min_max_norm(pos)
    nx.set_node_attributes(mixed_net, normed_pos, "normed_pos")
                             
    # assign the nodes into the sbm blocks
    # adapted from NetworkX generators.community.stochastic_block_model source code 
    size_cumsum = [sum(sbm_blocks[0:x]) for x in range(len(sbm_blocks) + 1)]
    partition = [
        set(nodes[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(len(size_cumsum) - 1)
    ]
                             
    # re-format into a dictionary keyed by node 
    partition_dict = {}
    for i in range(0,len(partition)):
        for j in partition[i]:
            partition_dict[j] = i
    
    # ITERATE THROUGH ALL POSSIBLE NODE COMBINATIONS:
                             
    # adapated from NetworkX source code -- https://networkx.org/documentation/stable/_modules/networkx/generators/geometric.html#random_geometric_graph
    edges = []
    for (u, pu), (v, pv) in itertools.combinations(mixed_net.nodes(data="normed_pos"), 2):
        
        # get the rgg probabilities of connection based on the distance between the normalized attribute vectors
        dist = distance.euclidean(pu,pv)
        rgg_p_connected_d1 = rgg_kernel(dist)
        rgg_p_connected_d2 = rgg_kernel(dist) # reciprocal
                             
        # get the sbm probabilities of connection based on partition assignment
        sbm_p_connected_d1 = sbm_probs[partition_dict[u]][partition_dict[v]]
        sbm_p_connected_d2 = sbm_probs[partition_dict[v]][partition_dict[u]]   

        # adjust sbm probabilities to force hierarchical direction (with some small % pointed the other way)
        if partition_dict[u] < partition_dict[v]:
            prob_mult_d1 = 1
        else:
            prob_mult_d1 = rec_prop
        sbm_p_connected_d1 = sbm_p_connected_d1 * prob_mult_d1
        
        if partition_dict[v] < partition_dict[u]:
            prob_mult_d2 = 1
        else:
            prob_mult_d2 = rec_prop
        sbm_p_connected_d2 = sbm_p_connected_d2 * prob_mult_d2
        
        # randomly choose between the two potential rgg directions
        # the other direction is adjusted to happen a small percent of the time
        draw = random.random()
        if draw < 0.5:
            prob_mult_d1 = 1
            prob_mult_d2 = rec_prop
        else:
            prob_mult_d1 = rec_prop
            prob_mult_d2 = 1
            
        rgg_p_connected_d1 = rgg_p_connected_d1*prob_mult_d1
        rgg_p_connected_d2 = rgg_p_connected_d2*prob_mult_d2
            
        # decide whether they are connected based on JOINT probability
        p_connected_d1 = sbm_p_connected_d1*(1-alpha) + rgg_p_connected_d1*(alpha)
        p_connected_d2 = sbm_p_connected_d2*(1-alpha) + rgg_p_connected_d2*(alpha)
                             
        #random draw and decide whether connected based on the joint probability
        draw = random.random()
        if draw < p_connected_d1:
            edges.append(((u, v)))
        
        draw = random.random()
        if draw < p_connected_d2:
            edges.append(((v, u)))
                             
    mixed_net.add_edges_from(edges)
    nx.set_node_attributes(mixed_net, pos, "pos") # the saved attributes for viz / prediction should not be the normalized versions
    nx.set_node_attributes(mixed_net, partition_dict, "block")
    
    return mixed_net

# Function to write graph P to file. Graph P has a pos attribute per node
# name - what to name the output network
# If group == True, also write out a column with the group parameters for each node 
def write_to_file(P,name,group):
    
    edge_list = open(f'{name}_edge_list.txt','w',newline='',encoding='utf-8')
    node_attr_list = open(f'{name}_node_attr_list.txt','w',newline='',encoding='utf-8')
    
    nodes = P.nodes()
    
    if 'pos' in list(P.nodes(data=True))[0][1]:
        len_pos = len(list(P.nodes(data=True))[0][1]['pos'])
        pos = True
    else:
        pos = False
    
    # Header 
    header = "id"
    if pos:
        for p in range(0,len_pos):
            header = header + " pos" + str(p)
    if group:
        header = header + " block"
    node_attr_list.write(f'{header}\n')
    
    for node in nodes:
        row = str(node)
        if pos:
            curr_pos = nodes[node]['pos']
            for p in range(0,len_pos):
                row = row + " " + str(curr_pos[p])
        if group:
            row = row + " " + str(nodes[node]['block'])
        node_attr_list.write(f'{row}\n')

    edges = list(P.edges())
    for edge in edges:
        edge_list.write(f"{edge[0]} {edge[1]} \n")
        
    edge_list.close()
    node_attr_list.close()
    
# Function to write graph P to file, same as above
# edges - fixed, pre-specified edge order (e.g., randomized)
def write_to_file_fixed(P,edges,name,group):
    
    edge_list = open(f'{name}_edge_list.txt','w',newline='',encoding='utf-8')
    node_attr_list = open(f'{name}_node_attr_list.txt','w',newline='',encoding='utf-8')
    
    nodes = P.nodes()
    
    if 'pos' in list(P.nodes(data=True))[0][1]:
        len_pos = len(list(P.nodes(data=True))[0][1]['pos'])
        pos = True
    else:
        pos = False
    
    # Header 
    header = "id"
    if pos:
        for p in range(0,len_pos):
            header = header + " pos" + str(p)
    if group:
        header = header + " block"
    node_attr_list.write(f'{header}\n')
    
    for node in nodes:
        row = str(node)
        if pos:
            curr_pos = nodes[node]['pos']
            for p in range(0,len_pos):
                row = row + " " + str(curr_pos[p])
        if group:
            row = row + " " + str(nodes[node]['block'])
        node_attr_list.write(f'{row}\n')

    assert set(edges) == set(list(P.edges())), "edges are not the same set"
    for edge in edges:
        edge_list.write(f"{edge[0]} {edge[1]} \n")
        
    edge_list.close()
    node_attr_list.close()
    

# Calculate theoretical maximum ROC-AUC performance for a mixed network, via Monte-Carlo sampling
# Alpha = 0, completely SBM, Alpha = 1, completely RGG
# networks - a list of synthetic networks generated at this value of alpha
# sbm_probs - sbm probabilities of connecting for the anchor SBM network
# rgg_kernel - rgg connection kernel used to generated the anchor RGG network
# alpha - current value of alpha
# rec_prop - proportion of edges reciprocated in mixed network generation when choosing direction
def get_theoretical_auc_max(networks, sbm_probs, rgg_kernel, alpha, rec_prop):
    
    # Normalize the attributes for all the nodes for distance calculations
    if alpha != 0:
        for ni in range(0,len(networks)):
            net = networks[ni]
            pos = nx.get_node_attributes(net,'pos')
            normed_pos = eco_predictors_helper.min_max_norm(pos)
            nx.set_node_attributes(net, normed_pos, "normed_pos")
    
    # Note that this is written for small networks, for larger networks you'd need to sample in 
    # a way that doesn't include explicitly listing all of the edges and non-edges 
    
    # Grab the set of edges across ALL of these synthetic networks
    all_edges = {} # {edge_index : (net_index, (edge_i, edge_j))}
    ei = 0
    for ni in range(0,len(networks)):
        for ee in networks[ni].edges():
            all_edges[ei] = (ni,(ee[0],ee[1]))
            ei += 1

    # Grab the set of non-edges across ALL of these synthetic networks
    all_non_edges = {} # {non_edge_index : (net_index, (non_edge_i, non_edge_j))}
    ei = 0
    for ni in range(0,len(networks)):
        non_edge_it = nx.non_edges(networks[ni])
        for ee in non_edge_it:
            all_non_edges[ei] = (ni,(ee[0],ee[1]))
            ei += 1
    
    # Create n randomly sampled pairs of true negatives (from non-edges), true positives (from edges)
    n = 100000
    pairs = []
    for i in range(0,n):
        #randomly sample a true negative (non-edge)
        rand_ind = np.random.randint(len(all_non_edges.keys()))
        # tn is  (net_index, (edge_i, edge_j))
        tn = all_non_edges[rand_ind]
        
        #randomly sample a true positive (edge)
        rand_ind = np.random.randint(len(all_edges.keys()))
        # tp is  (net_index, (edge_i, edge_j))
        tp = all_edges[rand_ind]
        
        pairs.append([tn,tp])
    
    # For each of the pairs, look at the actual probability of connecting for each from 
    # the generative model to get true positive score and true negative score 
   
    count = 0
    count_sbm = 0
    count_rgg = 0
    for pair in pairs:

        true_neg_net = networks[pair[0][0]]
        true_neg_edge = pair[0][1]
        true_pos_net = networks[pair[1][0]]
        true_pos_edge = pair[1][1]
        
        ## sbm probabilities of connecting ##
                
        if alpha != 1:
        
            tn_group1 = int(true_neg_net.nodes[true_neg_edge[0]]['block'])
            tn_group2 = int(true_neg_net.nodes[true_neg_edge[1]]['block'])

            tn_sbm_prob = sbm_probs[tn_group1][tn_group2]

            tp_group1 = int(true_pos_net.nodes[true_pos_edge[0]]['block'])
            tp_group2 = int(true_pos_net.nodes[true_pos_edge[1]]['block'])

            tp_sbm_prob = sbm_probs[tp_group1][tp_group2]
            
            # adjust probabilities for choosing direction
            if tn_group1 < tn_group2:
                directed_multiplier_1 = 1
            else:
                directed_multiplier_1 = rec_prop
            tn_sbm_prob = tn_sbm_prob * directed_multiplier_1

            if tp_group1 < tp_group2:
                directed_multiplier_2 = 1
            else:
                directed_multiplier_2 = rec_prop
            tp_sbm_prob = tp_sbm_prob * directed_multiplier_2
        
        else:
            
            # doesn't matter they go to 0 in the equation
            tn_sbm_prob = 0
            tp_sbm_prob = 0  
        
        ## rgg probabilities of connecting ###
        
        if alpha != 0:
                                    
            tn_pos1 = true_neg_net.nodes[true_neg_edge[0]]['normed_pos']
            tn_pos2 = true_neg_net.nodes[true_neg_edge[1]]['normed_pos']
            
            tn_distance = distance.euclidean(tn_pos1,tn_pos2)
            tn_rgg_prob = rgg_kernel(tn_distance)
            
            tp_pos1 = true_pos_net.nodes[true_pos_edge[0]]['normed_pos']
            tp_pos2 = true_pos_net.nodes[true_pos_edge[1]]['normed_pos']
            
            tp_distance = distance.euclidean(tp_pos1,tp_pos2)
            tp_rgg_prob = rgg_kernel(tp_distance)
                        
            # reciprocity adjustment
            adjustment = 0.5+(rec_prop/2)
            tn_rgg_prob = tn_rgg_prob*adjustment
            tp_rgg_prob = tp_rgg_prob*adjustment

        else:
            
            # doesn't matter they go to 0 in the equation
            tn_rgg_prob = 0
            tp_rgg_prob = 0
                
        #true negative score is the probabilty that the model generating process assigns for connecting the true negative
        tns = tn_sbm_prob*(1-alpha) + tn_rgg_prob*(alpha)
        #true positive score is the probability that the model generating process assigns for connecting the true positive 
        tps = tp_sbm_prob*(1-alpha) + tp_rgg_prob*(alpha)
              
        # If the true positive score is higher than the true negative score, add one to a running count
        if (tps > tns):
            count += 1
            
        # Break ties randomly
        if (tps == tns):
            rand_draw = np.random.random()
            if rand_draw > 0.5:
                count += 1
                
        # Also keep track of the SBM and RGG component curves for contextualizing performance      
        if (tp_sbm_prob > tn_sbm_prob):
            count_sbm += 1
            
        if (tn_sbm_prob == tp_sbm_prob):
            rand_draw = np.random.random()
            if rand_draw > 0.5:
                count_sbm += 1
        
        if (tp_rgg_prob > tn_rgg_prob):
            count_rgg += 1
            
        if (tn_rgg_prob == tp_rgg_prob):
            rand_draw = np.random.random()
            if rand_draw > 0.5:
                count_rgg += 1
    
    # The theoretical aux max is the proportion of pairs where true positive score was higher than true negative score) 
    auc_max = count / n
    auc_max_sbm = count_sbm /n
    auc_max_rgg = count_rgg / n
    return auc_max, auc_max_sbm, auc_max_rgg

                
# Calculate a numerical probability matrix representing the probability of two nodes connecting for a given mixed network
# This is another way to visualize mixing and can be used for an alternative calculation of ROC-AUC theoretical maximum
# net_num - number of mixed networks to generate at this alpha
# Arguments for net generation: nodes, num_bin, num_num, range_num, sbm_blocks, sbm_probs, rgg_kernel, rec_prop, alpha, pos
# net_return - the number of these example networks to return along with the probability matrix
def numerical_probabilities(net_num, nodes, num_bin, num_num, range_num, sbm_blocks, sbm_probs, rgg_kernel, rec_prop, alpha, pos, net_return):
    
    # Generate 10,000 mixed networks at a certain level of alpha (in the same way as for those tested)
    networks = []
    for i in range(0,net_num):
        net_curr = mixed_network_directed(nodes, num_bin, num_num, range_num, sbm_blocks, sbm_probs,\
                                          rgg_kernel, rec_prop, alpha, pos)
        networks.append(net_curr)
    
    # Assuming nodes indexed 0-n consecutively
    n_nodes = len(nodes)
    prob_matrix = np.zeros((n_nodes,n_nodes)) # entry i,j is the numerical probability for nodes conecting (directed)
    for network in networks:
        for edge in network.edges():
            prob_matrix[edge[0]][edge[1]] +=1
            
    # prob_matrix now represents the number of networks this edge showed up in, divide to get a numerical probability
    prob_matrix = prob_matrix/net_num
    return(prob_matrix, networks[0:net_return])   
             
# Alternative calculation for the ROC-AUC theoretical maximum at given alpha value based on a numerical probability matrix
# networks - list of synthetic networks at that alpha (node ids consecutive 0-N)
# alpha_prob - matrix of numerical probabilities of edge connections at that alpha (nodes indexed consecutive 0-N)
def get_theoretical_auc_max_numerical(networks, alpha_prob):
    
    # Note that this is written for small networks, for larger networks you'd need to sample in 
    # a way that doesn't include explicitly listing all of the non-edges 
    
    # Grab the set of edges across ALL of these synthetic networks
    all_edges = []
    for ni in range(0,len(networks)):
        for ee in networks[ni].edges():
            all_edges.append((ee[0],ee[1]))

    # Grab the set of non-edges across ALL of these synthetic networks
    all_non_edges = []
    for ni in range(0,len(networks)):
        non_edge_it = nx.non_edges(networks[ni])
        for ee in non_edge_it:
            all_non_edges.append((ee[0],ee[1]))
    
    # Create n randomly sampled pairs of true negatives (from non-edges), true positives (from edges)
    n = 100000
    pairs = []
    for i in range(0,n):
        #randomly sample a true negative (non-edge)
        rand_ind = np.random.randint(len(all_non_edges))
        tn = all_non_edges[rand_ind]
        
        #randomly sample a true positive (edge)
        rand_ind = np.random.randint(len(all_edges))
        tp = all_edges[rand_ind]
        
        pairs.append([tn,tp])
    
    # For each of the pairs, look at the numerical probability of connecting for each from 
    # the generative model to get true positive score and true negative score 
   
    count = 0
    for pair in pairs:

        true_neg_edge = pair[0]
        true_pos_edge = pair[1]
        
        ## Numeric probabilities of connecting
        tns = alpha_prob[true_neg_edge[0]][true_neg_edge[1]]
        tps = alpha_prob[true_pos_edge[0]][true_pos_edge[1]]

        # If the true positive score is higher than the true negative score, add one to a running count
        if (tps > tns):
            count += 1
            
        # Break ties randomly
        if (tps == tns):
            rand_draw = np.random.random()
            if rand_draw > 0.5:
                count += 1
    
    # The theoretical aux max is the proportion of pairs where true positive score was higher than true negative score) 
    auc_max = count / n
    return auc_max