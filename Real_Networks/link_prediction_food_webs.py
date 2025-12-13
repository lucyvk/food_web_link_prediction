# This file is used to submit missing link prediction tests across the entire food web database
# Methods and data files should first be copied to the location where this script is run and a results folder should be created 
#
# Note that this was run on CentOS Linux 7 with the following package versions:
# python: 3.6.3, imbalanced-learn 0.8.1, networkx 2.5.1, numpy 1.19.5, pandas 1.1.5, scikit-learn 0.24.2, scipy 1.5.4
# Author: Lucy Van Kleunen

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # for running in parallel
import multiprocessing
from multiprocessing import Pool
import time
import pickle

# Uncomment these lines if the methods files are not copied originally to the same location as this script
# import sys
# sys.path.append("../Methods/stacking_model")

import link_prediction_helper

# Note that the version of the data this is run on can be change by changing this variable
Processed_Data_Version = 'Input_Data_Disaggregated_Lifestage'

# Get saved information on food web attribute types and ids
with open(os.path.join(Processed_Data_Version, 'fw_attr_types.pickle'), 'rb') as handle:
    attribute_types = pickle.load(handle)
with open(os.path.join(Processed_Data_Version, 'fw_ids.pickle'), 'rb') as handle:
    fw_ids = pickle.load(handle)

# The following seeds correspond to those used for each iteration in the paper
# Input_Data_Disaggregated_Lifestage - seed=55 --> Results_Food_Webs_0
# Input_Data_Disaggregated_Lifestage_1 - seed=222 --> Results_Food_Webs_1
# Input_Data_Disaggregated_Lifestage_2 - seed=333 --> Results_Food_Webs_2
# Input_Data_Disaggregated_Lifestage_3 - seed=555 --> Results_Food_Webs_3
# Input_Data_Disaggregated_Lifestage_4 - seed=666 --> Results_Food_Webs_4
# Input_Data_Aggregated_Lifestage - seed=55 --> Results_Food_Webs_Aggregated

# Link prediction comparison arguments 
details = False
extra_links = False
res_folder = 'Results_Food_Webs_0'
seed = 55 # set a seed for reproducing the same results as those in the paper (on Input_Data_Disaggregated_Lifestage)

# short (in processed data) to long name (in results files) dictionary
folder_shorter_names = {'Grand Caricaie Clmown1':'Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1',\
    'Grand Caricaie Clmown2': 'Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown2',\
    'Grand Caricaie ClControl1': 'Grand Caricaie  marsh dominated by Cladietum marisci, not mown  ClControl1',\
    'Grand Caricaie ClControl2': 'Grand Caricaie  marsh dominated by Cladietum marisci, not mown  ClControl2',\
    'Grand Caricaie Scmown1': 'Grand Caricaie  marsh dominated by Schoenus nigricans, mown  Scmown1 ',\
    'Grand Caricaie Scmown2': 'Grand Caricaie  marsh dominated by Schoenus nigricans, mown  Scmown2 ',\
    'Grand Caricaie ScControl1': 'Grand Caricaie  marsh dominated by Schoenus nigricans, not mown  ScControl1 ',\
    'Grand Caricaie ScControl2': 'Grand Caricaie  marsh dominated by Schoenus nigricans, not mown  ScControl2 '}

K = 3 # set K used in KNN predictors
metric = "PR-AUC" # set metric used for model selection

# Uncomment these lines if a results folder is not already created at the same location as this script
# if not os.path.exists(res_folder):
#    os.mkdir(res_folder)

# If running this on Windows, the parallelization will not work unless included in a main block so uncomment this line and indent all following lines
#if __name__ == '__main__':  

lp_args = [] # collect arguments to pass into the function to run the missing link prediction tests for a single food web
for x in os.listdir(Processed_Data_Version):
    if os.path.isdir(os.path.join(Processed_Data_Version,x)):
        fw = x.split('_')[0]
        if fw in folder_shorter_names:
            fw = folder_shorter_names[fw]
        fw_id = x.split('_')[1]
        assert int(fw_id) == fw_ids[fw], "food web ids don't match"
        in_stem = os.path.join(Processed_Data_Version,x,fw_id)
        temp_stem = fw_id
        out_stem = x
        attr_types = attribute_types[fw] 
        curr_arg = [in_stem,temp_stem,out_stem,details,attr_types,extra_links,res_folder,seed,K,metric]
        lp_args.append(curr_arg)

# Parallelize so that multiple food webs are run at once
t0 = time.time()
p = Pool(multiprocessing.cpu_count()-1)
p.map(link_prediction_helper.run_stacking_prediction_comparison, lp_args)
p.close()
p.join() #wait    
t1 = time.time()
print(f"time - {str(t1-t0)}")