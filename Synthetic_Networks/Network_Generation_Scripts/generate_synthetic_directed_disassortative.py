# This file generates synthetic networks across varying rho (the RGG networks are disassortative)
# An output folder should be made in the same directory
#
# Note that this was run on CentOS Linux 7 with the following package versions:
# python: 3.6.3, imbalanced-learn 0.8.1, networkx 2.5.1, numpy 1.19.5, pandas 1.1.5, scikit-learn 0.24.2, scipy 1.5.4
# Author: Lucy Van Kleunen

import numpy as np
import random
import csv
import pickle
import os

# If this file is not moved to the same directory
# import sys
# sys.path.append('..')
# sys.path.append(os.path.join('..','..','Methods','stacking_model'))

import generate_helper

# Set random seed throughout so it's reproducible
SEED = 19
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_FOLDER = 'Synth_Nets_Directed_Disassortative'

# Uncomment if output folder not in same location
# OUTPUT_FOLDER = os.path.join('..','Synth_Nets_Directed_Assortative')
# if not os.path.exists(OUTPUT_FOLDER):
#    os.mkdir(OUTPUT_FOLDER)

# Parameters shared by RGG and SBM networks
NUM_NODES = 45
NODES = range(0,NUM_NODES)

# SBM parameters
BLOCKZ = [15,15,15] # group sizes
PROBZ = [[0,0.544,0],[0.544,0,0.544],[0,0.544,0]] # edge probabilities
REC_PROP = 0.02 # proportion of edges that are reciprocated

# RGG parameters
NUM_BIN = 2 # number of binary attributes
NUM_NUM = 2 # number of numeric attributes 
NUM_RANGES = [[0,1],[0,1]] # ranges of numeric attributes
def step_fn_kernel_disassortative(distance):
    if distance > 1.425:
        prob = 1
    else:
        prob = 0
    return prob

# Parameters for how detailed the sweep is 
N = 30 # number of stops along rho to test at
M = 20 # network iterations per rho

# Rhos to generate networks at -- set up such that the results are staggered for the 3 models tested
full_model_stops = list(np.linspace(0,1,num=N,endpoint=True))
struc_model_stops = full_model_stops - (full_model_stops[1]/2) # stagger
attr_model_stops = full_model_stops + (full_model_stops[1]/2) # stagger
# adjust so they all still start and stop at 0 and 1
struc_model_stops[0] = 0
struc_model_stops[-1] = 1
attr_model_stops[0] = 0
attr_model_stops[-1] = 1

# Generate and save the networks 
# Lists of lists - One list for each of the N rhos, each list of size M
nets_struc = []
nets_full = []
nets_attr = []
auc_maxes = [] # Only save one set of theoeretical AUC maximums (not at all 3 staggered values) [from full because centered]
for s in range(0,N):
    rho_struc = struc_model_stops[s]
    rho_full = full_model_stops[s]
    rho_attr = attr_model_stops[s]
    # Lists of size M at this rho
    nets_struc_curr = []
    nets_full_curr = []
    nets_attr_curr = []
    for i in range(0,M):
        net_struc = generate_helper.mixed_network_directed(NODES, NUM_BIN, NUM_NUM, NUM_RANGES, BLOCKZ, PROBZ, step_fn_kernel_disassortative, REC_PROP, rho_struc, None)
        net_full = generate_helper.mixed_network_directed(NODES, NUM_BIN, NUM_NUM, NUM_RANGES, BLOCKZ, PROBZ, step_fn_kernel_disassortative, REC_PROP, rho_full, None)
        net_attr = generate_helper.mixed_network_directed(NODES, NUM_BIN, NUM_NUM, NUM_RANGES, BLOCKZ, PROBZ, step_fn_kernel_disassortative, REC_PROP, rho_attr, None)
        nets_struc_curr.append(net_struc)
        nets_full_curr.append(net_full)
        nets_attr_curr.append(net_attr)
    auc_max_curr = generate_helper.get_theoretical_auc_max(nets_full_curr,PROBZ,step_fn_kernel_disassortative,rho_full,REC_PROP)
    auc_maxes.append(auc_max_curr)
    nets_struc.append(nets_struc_curr)
    nets_full.append(nets_full_curr)
    nets_attr.append(nets_attr_curr)
        
# Write AUC max values to a file
with open(os.path.join(f'{OUTPUT_FOLDER}','theoretical_auc_max.csv'),'w') as max_file:
    max_writer = csv.writer(max_file)
    max_writer.writerow(["rho","auc_max","auc_max_sbm","auc_max_rgg"])
    for i in range(0,N):
        max_writer.writerow([full_model_stops[i], auc_maxes[i][0], auc_maxes[i][1], auc_maxes[i][2]])
        
# Randomize edge list order (for testing prediction) and write networks to file
def write_it(curr_net,name):
    curr_edges = list(curr_net.edges())
    num_edges = len(curr_edges)
    randomized_edges = random.sample(curr_edges, num_edges)
    generate_helper.write_to_file_fixed(curr_net,randomized_edges,name,False)

for s in range(0,N):
    for i in range(0,M):
        write_it(nets_struc[s][i],os.path.join(f'{OUTPUT_FOLDER}',f'n{s}_m{i}_struc'))
        write_it(nets_full[s][i],os.path.join(f'{OUTPUT_FOLDER}',f'n{s}_m{i}_full'))
        write_it(nets_attr[s][i],os.path.join(f'{OUTPUT_FOLDER}',f'n{s}_m{i}_attr'))

# Save networks in networkx Graph form too in case we want to load these in later to check / visualize
with open(os.path.join(f'{OUTPUT_FOLDER}','nets_struc.pickle'), 'wb') as handle:
    pickle.dump(nets_struc, handle)
with open(os.path.join(f'{OUTPUT_FOLDER}','nets_full.pickle'), 'wb') as handle:
    pickle.dump(nets_full, handle)
with open(os.path.join(f'{OUTPUT_FOLDER}','nets_attr.pickle'), 'wb') as handle:
    pickle.dump(nets_attr, handle)
    
# Record the stops along rho that were used too 
with open(os.path.join(f'{OUTPUT_FOLDER}','struc_model_stops.pickle'), 'wb') as handle:
    pickle.dump(struc_model_stops, handle)
with open(os.path.join(f'{OUTPUT_FOLDER}','full_model_stops.pickle'), 'wb') as handle:
    pickle.dump(full_model_stops, handle)
with open(os.path.join(f'{OUTPUT_FOLDER}','attr_model_stops.pickle'), 'wb') as handle:
    pickle.dump(attr_model_stops, handle)