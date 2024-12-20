# This file performs missing link prediction on the synthetic networks across varying rho (the RGG networks are disassortative)
# The synthetic networks and methods files should be moved to the same location where this script is run
# A results folder should be made in the same location
#
# Note that this was run on CentOS Linux 7 with the following package versions:
# python: 3.6.3, imbalanced-learn 0.8.1, networkx 2.5.1, numpy 1.19.5, pandas 1.1.5, scikit-learn 0.24.2, scipy 1.5.4
# Author: Lucy Van Kleunen

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # for multiprocessing
import multiprocessing
from multiprocessing import Pool
import time

# Uncomment these lines if the methods files are not copied originally to the same location as this script
# import sys
# sys.path.append(os.path.join('..','..','Methods','stacking_model'))

import link_prediction_helper

# Link prediction comparison arguments
details = False
attr_types = [0,0,1,1] # two binary then two numeric 
extra_links = False
seed = 223 # for reproducing the same results
K = 3
opt = "PR-AUC"
 
N = 30 # number of stops along rho to test at
M = 20 # network iterations per rho

lp_args = []
net_folder =  'Synth_Nets_Directed_Disassortative'
res_folder = 'Results_Synthetic_Directed_Disassortative'

# if the networks are not moved to the same folder use the following paths:
# net_folder = os.path.join('..','Synth_Nets_Directed_Disassortative')
# res_folder = os.path.join('..','Results_Synthetic_Directed_Disassortative')
# if not os.path.exists(res_folder):
#    os.mkdir(res_folder)

# If running this on Windows, the parallelization will not work unless included in a main block so uncomment this line and indent all following lines#
#if __name__ == '__main__':  

for s in range(0,N):
    for i in range(0,M):
        
        curr_web = f'n{s}_m{i}_struc'
        in_stem = os.path.join(f'{net_folder}',f'{curr_web}')
        temp_stem = curr_web
        out_stem = curr_web
        curr_arg = [in_stem,temp_stem,out_stem,details,attr_types,extra_links,res_folder,seed,K,opt]
        lp_args.append(curr_arg)
        
        curr_web = f'n{s}_m{i}_full'
        in_stem = os.path.join(f'{net_folder}',f'{curr_web}')
        temp_stem = curr_web
        out_stem = curr_web
        curr_arg = [in_stem,temp_stem,out_stem,details,attr_types,extra_links,res_folder,seed,K,opt]
        lp_args.append(curr_arg)
        
        curr_web = f'n{s}_m{i}_attr'
        in_stem = os.path.join(f'{net_folder}',f'{curr_web}')
        temp_stem = curr_web
        out_stem = curr_web
        curr_arg = [in_stem,temp_stem,out_stem,details,attr_types,extra_links,res_folder,seed,K,opt]
        lp_args.append(curr_arg)

# Send the link prediction tests running in parallel across synthetic networks
t0 = time.time()
p = Pool(multiprocessing.cpu_count()-1)
p.map(link_prediction_helper.run_stacking_prediction_comparison, lp_args)
p.close()
p.join()
t1 = time.time()
print(f"time - {str(t1-t0)}")
