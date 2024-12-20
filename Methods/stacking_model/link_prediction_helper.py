# Helper functions to run link prediction tests for a food web
# Author: Lucy Van Kleunen

import networkx as nx
import OLP as olp
import csv
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # for when running in parallel on the cluster

''' Helper function to read a network from file (undirected, with attributes)

Parameters:
name - file name

Returns:
a NetworkX graph version of the network

'''
def read_from_file(name):
    
    node_rdr = open(name + '_node_attr_list.txt','r')
    edge_rdr = open(name + '_edge_list.txt','r')
    
    attr_names = []
    
    first = True
    gg = nx.Graph()
    for a in node_rdr:
        row = a.split()
        if first:
            attr_names = row
        else:
            row = [int(x) if x.isnumeric() else x for x in row]
            gg.add_node(row[0])
            for attr in range(1,len(row)):
                gg.nodes[row[0]][attr_names[attr]] = row[attr]
        first = False
        
    for a in edge_rdr:
        row = a.split()
        row = [int(x) if x.isnumeric() else x for x in row]
        gg.add_edge(row[0],row[1])
        
    node_rdr.close()
    edge_rdr.close()
    
    return gg 

''' Helper function to read a network from file (directed, with attributes)

Parameters:
name - file name

Returns:
a NetworkX graph version of the network

'''
def read_from_file_directed(name):
    
    node_rdr = open(name + '_node_attr_list.txt','r')
    edge_rdr = open(name + '_edge_list.txt','r')
    
    attr_names = []
    
    first = True
    gg = nx.DiGraph()
    for a in node_rdr:
        row = a.split()
        if first:
            attr_names = row
        else:
            # convert to int and float appropriately
            for x in range(0,len(row)):
                # check if int
                if row[x].isnumeric():
                    row[x] = int(row[x])
                # check if float
                elif row[x].replace(".", "").replace("-", "").isnumeric():
                    row[x] = float(row[x])
                else:
                    continue 
            gg.add_node(row[0])
            for attr in range(1,len(row)):
                gg.nodes[row[0]][attr_names[attr]] = row[attr]
        first = False
        
    for a in edge_rdr:
        row = a.split()
        # Directed edges
        row = [int(x) if x.isnumeric() else x for x in row]
        gg.add_edge(row[0],row[1])
        
    node_rdr.close()
    edge_rdr.close()
    
    return gg 

''' Helper function to do the basic comparison between the three models for a given network + save results (across 5 folds)

Parameters:
args - An argument matrix with arguments as listed below

Returns:
No returns, but saves missing link prediction test results to file

'''
def run_stacking_prediction_comparison(args):

    in_stem =  args[0] # network file to run on
    temp_stem = args[1] # for the out_name
    out_stem = args[2] # where to write the ROC-AUC and PR-AUC results 

    # Parameters passed through to the link prediction script
    edges_orig_path = f'{in_stem}_edge_list.txt' # edge list file path
    node_attr_path = f'{in_stem}_node_attr_list.txt' # node list file path
    details = args[3] # whether to print details
    attr_types = args[4] # what the attribute types are
    extra_links = args[5] # whether to include extra links in the training network
    res_folder = args[6] # location of the results folder
    seed = args[7] # seed for reproducibility
    K = args[8] # number of KNN to look at in KNN predictors
    metric = args[9] # metric to use for hyperparameter selection
    
    # ROC-AUC results
    out_file_ROC_AUC = f'./{res_folder}/stacking_auc_{out_stem}.csv'
    sim_output_ROC_AUC = open(out_file_ROC_AUC,'w',newline='') 
    sim_writer_ROC_AUC = csv.writer(sim_output_ROC_AUC)
    
    # Average precision score results (PR-AUC estimate)
    out_file_AVP = f'./{res_folder}/stacking_avp_{out_stem}.csv'
    sim_output_AVP = open(out_file_AVP,'w',newline='') 
    sim_writer_AVP = csv.writer(sim_output_AVP)
    
    # Average precision score baselines
    out_file_AVP_baseline = f'./{res_folder}/avp_baseline_{out_stem}.csv'
    sim_output_AVP_baseline = open(out_file_AVP_baseline,'w',newline='') 
    sim_writer_AVP_baseline = csv.writer(sim_output_AVP_baseline)

    for fold in range(0,5):

        # topological only
        print(f"running for topological only {temp_stem} fold - {fold}")
        out_name = f'{temp_stem}_top_{fold}'  
        auc0, avp0, b0 = olp.topol_stacking_attr_food_web(edges_orig_path, node_attr_path, out_name=out_name, include_topo=True, include_attr=False, details=details, attr_types = None, extra_links=extra_links, res_folder=res_folder, seed=seed, lp_fold=fold, feature_importance=0, K=K, metric=metric)
        
        # attributes only
        print(f"running for attributes only {temp_stem} fold - {fold}")
        out_name = f'{temp_stem}_attribute_{fold}' 
        auc1, avp1, b1 =  olp.topol_stacking_attr_food_web(edges_orig_path, node_attr_path, out_name=out_name, include_topo=False, include_attr=True, details=details, attr_types=attr_types, extra_links=extra_links, res_folder=res_folder, seed=seed, lp_fold=fold, feature_importance=0, K=K, metric=metric)
        
        #both topological and attributes - save feature importance for these
        print(f"running for full model {temp_stem} fold - {fold}")
        out_name = f'{temp_stem}_full_{fold}'
        auc2, avp2, b2 = olp.topol_stacking_attr_food_web(edges_orig_path, node_attr_path, out_name=out_name, include_topo=True, include_attr=True, details=details, attr_types=attr_types, extra_links=extra_links, res_folder=res_folder, seed=seed, lp_fold=fold, feature_importance=3, K=K, metric=metric)

        # 0 - result with stacking model, topological
        # 1 - result with stacking model, attributes
        # 2 - result with stacking model, full
        sim_res_row_ROC_AUC = [auc0,auc1,auc2]
        sim_writer_ROC_AUC.writerow(sim_res_row_ROC_AUC)
        
        sim_res_row_AVP = [avp0,avp1,avp2]
        sim_writer_AVP.writerow(sim_res_row_AVP)
        
        sim_res_row_AVP_baseline = [b0,b1,b2]
        sim_writer_AVP_baseline.writerow(sim_res_row_AVP_baseline)

    sim_output_ROC_AUC.close()
    sim_output_AVP.close()
    sim_output_AVP_baseline.close()
