# Stacking model link prediction code
# Adapted for 5 fold cross validation over directed, attributed networks with a custom set of ecological predictors by Lucy Van Kleunen from https://github.com/Aghasemian/OptimalLinkPrediction

import os
import os.path
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from scipy import linalg
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from scipy.spatial import distance
from sklearn.inspection import permutation_importance
import pickle
import math
import random
import eco_predictors_helper
import directed_triangles
import shutil
import KNN_predictors_helper


def gen_tr_ho_networks(A_net, alpha_, edge_list, lp_fold, details):
    """ 
    This function constructs the holdout and training (directed) networks uniformly sampled from the original network

    Parameters:
    A_net : the original network    
    alpha_ : the % of edges left in the training network from the hold out
    edge_list: the full edge list (already randomized) for constructing the folds
    lp_fold: which fold we are using as the test (for 5-fold CV for creating hold out networks)
    details: Flag for whether or not to print details

    Returns:
    A_ho: the hold out network 
    A_tr: the training network
    -------

    """
    
    full_edges = edge_list
    assert set(edge_list) == set(list(A_net.edges())), "not same set of edges"
    full_nodes = list(A_net.nodes())
    kf = KFold(n_splits=5) 
    for i, (train_index, test_index) in enumerate(kf.split(full_edges)):
        if i == lp_fold:
            hold_out_edges = [full_edges[j] for j in train_index]
    training_edges = random.sample(hold_out_edges,int(len(hold_out_edges)*alpha_))
        
    if details:
        print(f"number of holdout edges |E'| : {len(hold_out_edges)}")
        print(f"number of training edges |E''| : {len(training_edges)}")
        
    A_ho = nx.DiGraph()
    A_tr = nx.DiGraph()
        
    A_ho.add_nodes_from(full_nodes)
    A_tr.add_nodes_from(full_nodes)
    for ee in hold_out_edges:
        A_ho.add_edge(ee[0],ee[1])
    for ee in training_edges:
        A_tr.add_edge(ee[0],ee[1])
        
    return A_ho, A_tr
    
def sample_true_false_edges_small_directed(A_net, A_tr, A_ho, out_name, details, extra_links, res_folder): 
    """ 
    This function creates the training and holdout sets for model training and testing from the training and hold out networks

    Parameters:
    A_net: The original network
    A_tr: The training network
    A_ho: The holdout network
    out_name: Text used to identify this network in the output files
    details: Flag for whether or not to print details
    extra_links: Flag for if all edges in the training network added as examples to the positive class
    res_folder: Results folder name for saving files

    Returns:
    Nothing is returned but the following files are written:
    - Positive class, training: "f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_t.txt"
    - Negative class, training: f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_f.txt"
    - Positive class, holdout / test: f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_t.txt"
    - Negative class, holdout / test: f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_f.txt"
    """
    
    node_list = list(A_net.nodes())
    full_edges = list(A_net.edges())
    hold_out_edges = list(A_ho.edges())
    training_edges = list(A_tr.edges())
    
    # --- Training set -----     
    
    # True candidates - those removed from hold out to create training 
    training_true = list(set(hold_out_edges) - set(training_edges))
        
    # Add in the true links to the true candidates
    if extra_links:
        training_true = training_true + training_edges
    
    # False candidates - unobserved edges in the hold out network
    training_false = []
    # possible directed edges, size |V|(|V|-1), minus existing
    for ni in node_list:
        for nj in node_list:
            if (ni != nj):
                if ((ni,nj) not in hold_out_edges):
                    training_false.append((ni,nj))
                    
    # Reduce to 10K random false examples for model training if there are more (speed up for networks with over ~100 nodes)
    if len(training_false) > 10000:
        random_indices = random.sample(range(0, len(training_false)), 10000)
        training_false = [training_false[i] for i in random_indices]
        assert len(training_false) == 10000, "length doesn't equal 10000"
    
    if details:
        print(f"Number unique true examples (those removed from hold out to create training): {len(training_true)}")
        print(f"Number unique false examples (unobserved in hold out): {len(training_false)}")
        print(f"Numebr unique examples total training (# non-links): {len(training_true) + len(training_false)}")
  
    # store for later use
    if not os.path.isdir(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/"):
        os.mkdir(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/")
    np.savetxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_t.txt",training_true,fmt='%u')
    np.savetxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_f.txt",training_false,fmt='%u')

    # ---- Hold out set --------
    
    # Trues - those removed to create hold out network 
    holdout_true = list(set(full_edges) - set(hold_out_edges))
    
    # Falses - true negative edges
    holdout_false = []
    # possible directed edges, size |V|(|V|-1), minus existing
    for ni in node_list:
        for nj in node_list:
            if (ni != nj):
                if ((ni,nj) not in full_edges):
                    holdout_false.append((ni,nj))
        
    if details:
        print(f"Number unique true examples (those removed from full to create hold out - MISSING LINKS): {len(holdout_true)}")
        print(f"Number unique false examples (unobserved in true - TRUE NEGATIVES): {len(holdout_false)}")
        print(f"Number unique examples total holdout (# non-links): {len(holdout_true) + len(holdout_false)}")
    
    # store for later use
    if not os.path.isdir(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/"):
        os.mkdir(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/")
    np.savetxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_t.txt",holdout_true,fmt='%u')
    np.savetxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_f.txt",holdout_false,fmt='%u')    
    

def gen_topol_feats_attr_food_web(G, edge_s, attr_dict, include_topo, K, details): 
    
    """ 
    This function generates features from the passed in directed network G

    Parameters:
    G: The directed network to generate features from
    edge_s: The set of edges to generate features for 
    attr_dict: A dictionary of node attributes, if Null, no node attribute features are generated
    include_topo: Whether or not to generate topological (network structure) featurs
    K: K to use for KNN predictors
    details: Flag for whether or not to print details

    Returns:
    -------
    df_feat: A data frame of features for the edge set
    """
    
    if details:
        print(f"size of edge set used to generate topological features: {len(edge_s)}")
    
    # samples chosen - features
    edge_pairs_f_i = edge_s[:,0]
    edge_pairs_f_j = edge_s[:,1]
    d = {'i':edge_pairs_f_i, 'j':edge_pairs_f_j}

    if attr_dict:    
           
        # collect node attributes
        num_attr = len(attr_dict[edge_s[0,0]])
        attr_names = attr_dict['attr_names']
        attr_types = attr_dict['attr_types']
        attrs_i = np.zeros((len(edge_s),num_attr))
        attrs_j = np.zeros((len(edge_s),num_attr))

        assert len(attr_types) == len(attr_names), "attribute types and names size don't match"
        # treat binary and numeric attributes differently
        num_numeric = np.sum(attr_types)
        num_bin = len(attr_types) - num_numeric

        # whether to include different distance predictors
        include_num_dist = True if num_numeric > 0 else False
        include_bin_dist = True if num_bin > 0 else False
        include_dist = True if num_numeric > 0 and num_bin > 0 else False

        # Prepare to save distances
        if include_dist:
            euclideans = np.zeros(len(edge_s))
            manhattans = np.zeros(len(edge_s))
            cosines = np.zeros(len(edge_s))
            dots = np.zeros(len(edge_s))

        if include_num_dist:
            euclideans_num = np.zeros(len(edge_s))
            manhattans_num = np.zeros(len(edge_s))
            cosines_num = np.zeros(len(edge_s))
            dots_num = np.zeros(len(edge_s))
            
            # min-max normalize the numeric attributes to range [0,1]
            norm_attr_dict = eco_predictors_helper.min_max_norm(attr_dict)
            norm_attrs_i = np.zeros((len(edge_s),num_attr))
            norm_attrs_j = np.zeros((len(edge_s),num_attr))
            attr_ratios1 = np.zeros((len(edge_s),num_numeric))

        if include_bin_dist:
            hamming = np.zeros(len(edge_s))
            jaccard_attr = np.zeros(len(edge_s))
           
            bin_attr_dict = create_binary_attr_dict(attr_dict)
       
        # Collect attributes and distance metrics 
        ctt = 0 
        for ee in range(len(edge_s)):    
            an = 0 
            for aa in range(num_attr):
                attrs_i[ee,aa] = attr_dict[edge_s[ee][0]][aa]
                attrs_j[ee,aa] = attr_dict[edge_s[ee][1]][aa]

                if attr_types[aa] == 1:
                    if attrs_j[ee,aa]: # prevent divide by zero - not common
                        attr_ratios1[ee,an] =  attrs_i[ee,aa]/attrs_j[ee,aa]
                    else:
                        attr_ratios1[ee,an] = 1000 # large number
                    an+=1

            if include_num_dist:
                # use normalized versions for distance metrics
                attr_vector_i = norm_attr_dict[edge_s[ee][0]]
                attr_vector_j = norm_attr_dict[edge_s[ee][1]]
                
            else:
                attr_vector_i = attr_dict[edge_s[ee][0]]
                attr_vector_j = attr_dict[edge_s[ee][1]]

            # distance metrics
            if include_dist:
                
                # Antcipate and prevent divide by 0 warning for cosine distance
                uu = np.average(np.square(attr_vector_i), weights=None)
                vv = np.average(np.square(attr_vector_j), weights=None)
                if np.sqrt(uu * vv) == 0:
                    do_cos = False
                else:
                    do_cos = True
                
                euc = distance.euclidean(attr_vector_i,attr_vector_j)
                man = distance.cityblock(attr_vector_i,attr_vector_j)
                if do_cos:
                    cos = distance.cosine(attr_vector_i,attr_vector_j)
                else:
                    cos = 0
                dot = np.dot(attr_vector_i,attr_vector_j)
                
                if not math.isfinite(euc):
                    euc = 0
                if not math.isfinite(man):
                    man = 0
                if not math.isfinite(cos):
                    cos = 0
                if not math.isfinite(dot):
                    dot = 0
                
                euclideans[ee] = euc
                manhattans[ee] = man
                cosines[ee] = cos
                dots[ee] = dot
                
            # distance metrics for binary and numeric features only 
            if include_num_dist:                
                
                attr_vector_num_i = [attr_vector_i[x] for x in range(0,len(attr_vector_i)) if attr_types[x] == 1]
                attr_vector_num_j = [attr_vector_j[x] for x in range(0,len(attr_vector_j)) if attr_types[x] == 1]
                
                # Antcipate and prevent divide by 0 warning for cosine distance
                uu = np.average(np.square(attr_vector_num_i), weights=None)
                vv = np.average(np.square(attr_vector_num_j), weights=None)
                if np.sqrt(uu * vv) == 0:
                    do_cos = False
                else:
                    do_cos = True
                
                euc_num = distance.euclidean(attr_vector_num_i,attr_vector_num_j)
                man_num = distance.cityblock(attr_vector_num_i,attr_vector_num_j)
                if do_cos:
                    cos_num = distance.cosine(attr_vector_num_i,attr_vector_num_j)
                else:
                    cos_num = 0
                dot_num = np.dot(attr_vector_num_i,attr_vector_num_j)
                
                if not math.isfinite(euc_num):
                    euc_num = 0
                if not math.isfinite(man_num):
                    man_num = 0
                if not math.isfinite(cos_num):
                    cos_num = 0
                if not math.isfinite(dot_num):
                    dot_num = 0
                
                euclideans_num[ee] = euc_num
                manhattans_num[ee] = man_num
                cosines_num[ee] = cos_num
                dots_num[ee] = dot_num
                    
            if include_bin_dist:
                
                attr_vector_bin_i = bin_attr_dict[edge_s[ee][0]]
                attr_vector_bin_j = bin_attr_dict[edge_s[ee][1]]
                
                ham = distance.hamming(attr_vector_bin_i,attr_vector_bin_j)
                jac = distance.jaccard(attr_vector_bin_i,attr_vector_bin_j)
                
                if not math.isfinite(ham):
                    ham = 0
                if not math.isfinite(jac):
                    jac = 0
                
                hamming[ee] = ham
                jaccard_attr[ee] = jac

            ctt +=1
             
        if include_topo: # Only for full model
                        
            if include_num_dist:
                attr_dict_curr = norm_attr_dict # pass min max normalized attributes if there are numeric attributes
            else:
                attr_dict_curr = {}
                for node_key in attr_dict:
                    if node_key != 'attr_types' and node_key != 'attr_names':
                        attr_dict_curr[node_key] = attr_dict[node_key]
            if details:
                print(f"K used for KNN predictors: {K}")
                
            # KNN predictor - D1 / based on Euclidean distance between full attribute vectors (normalized)
            KNN_in_D1, KNN_out_D1 = KNN_predictors_helper.KNN_predictor(G, edge_s, attr_dict_curr, K, 2, False)
            # KNN predictor - D2 / based on Manhattan distance between full attribute vectors (normalized)
            KNN_in_D2, KNN_out_D2 = KNN_predictors_helper.KNN_predictor(G, edge_s, attr_dict_curr, K, 1, False)
            
            if include_bin_dist:
                # KNN predictor - D3 / based on Manhattan distance between binary part of attribute vectors
                KNN_in_D3, KNN_out_D3 = KNN_predictors_helper.KNN_predictor(G, edge_s, bin_attr_dict, K, 1, False)
                
                # KNN predictor - D4 / based on Jaccard distance between binary part of attribute vectors
                KNN_in_D4, KNN_out_D4 = KNN_predictors_helper.KNN_predictor_jaccard_attr(G, edge_s, bin_attr_dict, K, False)
            
        #add in the attributes for i and j
        for aa in range(0,num_attr):
            d[attr_names[aa] + "_i"] = attrs_i[:,aa]
            d[attr_names[aa] + "_j"] = attrs_j[:,aa]


        if include_dist:
            d['euclidean'] = euclideans
            d['manhattan'] = manhattans
            d['cosine'] = cosines
            d['dot_product'] = dots

        if num_numeric > 0:
            an = 0
            for aa in range(0,num_attr):
                if attr_types[aa] == 1:
                    d[attr_names[aa] + "_ratio"] = attr_ratios1[:,an]
                    an+=1

        if include_num_dist:
            d['euclidean_num'] = euclideans_num
            d['manhattan_num'] = manhattans_num
            d['cosine_num'] = cosines_num
            d['dot_product_num'] = dots_num

        if include_bin_dist:
            d['hamming'] = hamming
            d['jaccard_attr'] = jaccard_attr
            
        if include_topo: # only for full model
            d['KNN-in-D1'] = KNN_in_D1
            d['KNN-out-D1'] = KNN_out_D1
            d['KNN-in-D2'] = KNN_in_D2
            d['KNN-out-D2'] = KNN_out_D2
            if include_bin_dist:
                d['KNN-in-D3'] = KNN_in_D3
                d['KNN-out-D3'] = KNN_out_D3 
                d['KNN-in-D4'] = KNN_in_D4
                d['KNN-out-D4'] = KNN_out_D4 
        
        number_attr = len(d.keys())
        if details:
            print(f"number attribute features: {number_attr}")
    else:
        number_attr = 0 
    if include_topo:
        
        nodes = list(G.nodes())
        edges = list(G.edges())
        A = nx.adjacency_matrix(G,nodelist=nodes).todense()
        N = len(nodes)
        
        assert nx.is_directed(G), "graph not directed"
        
        # local number of directed triangles for i and j (LNTD_i, LNTD_j)
        numtriang_nodes_obj = directed_triangles.triangles(G)
        numtriang_nodes = []
        for nn in range(len(nodes)):
            numtriang_nodes.append(numtriang_nodes_obj[nn])
        numtriang1_edges = []
        numtriang2_edges = []
        for ee in range(len(edge_s)):
            numtriang1_edges.append(numtriang_nodes[edge_s[ee][0]])
            numtriang2_edges.append(numtriang_nodes[edge_s[ee][1]])
            
        # Trophic levels i and j
        try:
            trophic_levels = nx.trophic_levels(G)
            trophic_levels_nodes = []
            for nn in range(len(nodes)):
                trophic_levels_nodes.append(trophic_levels[nn])
            trophic_level1_edges = []
            trophic_level2_edges = []
            for ee in range(len(edge_s)):
                trophic_level1_edges.append(trophic_levels_nodes[edge_s[ee][0]])
                trophic_level2_edges.append(trophic_levels_nodes[edge_s[ee][1]])
        except:
            print(f"trophic level exception")
            trophic_level1_edges = [0]*len(edge_s)
            trophic_level2_edges = [0]*len(edge_s)

        # Page rank values for i and j (PR_i, PR_j)
        page_rank_nodes_obj = nx.pagerank(G)
        page_rank_nodes = []
        for nn in range(len(nodes)):
            page_rank_nodes.append(page_rank_nodes_obj[nn])

        page_rank1_edges = []
        page_rank2_edges = []
        for ee in range(len(edge_s)):
            page_rank1_edges.append(page_rank_nodes[edge_s[ee][0]])
            page_rank2_edges.append(page_rank_nodes[edge_s[ee][1]])

        # j-th entry of the personalized page rank of node i (PPR)
        page_rank_pers_nodes = []
        hot_vec = {}
        for nn in range(len(nodes)):
            hot_vec[nn] = 0
        for nn in range(len(nodes)):
            hot_vec_copy = hot_vec.copy()
            hot_vec_copy[nn] = 1 
            page_rank_pers_nodes.append(nx.pagerank(G,personalization=hot_vec_copy))

        page_rank_pers_edges = []
        for ee in range(len(edge_s)):
            page_rank_pers_edges.append(page_rank_pers_nodes[edge_s[ee][0]][edge_s[ee][1]])

        # local clustering coefficients for i and j
        clust_nodes_obj = nx.clustering(G)
        clust_nodes = []
        for nn in range(len(nodes)):
            clust_nodes.append(clust_nodes_obj[nn])

        clust1_edges = []
        clust2_edges = []
        for ee in range(len(edge_s)):
            clust1_edges.append(clust_nodes[edge_s[ee][0]])
            clust2_edges.append(clust_nodes[edge_s[ee][1]])

        # average neighbor degrees for i and j (AND_i, AND_j)
        ave_neigh_deg_in_nodes_obj = nx.average_neighbor_degree(G, source="in")
        ave_neigh_deg_in_nodes = []
        for nn in range(len(nodes)):
            ave_neigh_deg_in_nodes.append(ave_neigh_deg_in_nodes_obj[nn])

        ave_neigh_deg_in1_edges = []
        ave_neigh_deg_in2_edges = []
        for ee in range(len(edge_s)):
            ave_neigh_deg_in1_edges.append(ave_neigh_deg_in_nodes[edge_s[ee][0]])
            ave_neigh_deg_in2_edges.append(ave_neigh_deg_in_nodes[edge_s[ee][1]])

        ave_neigh_deg_out_nodes_obj = nx.average_neighbor_degree(G, source="out")
        ave_neigh_deg_out_nodes = []
        for nn in range(len(nodes)):
            ave_neigh_deg_out_nodes.append(ave_neigh_deg_out_nodes_obj[nn])

        ave_neigh_deg_out1_edges = []
        ave_neigh_deg_out2_edges = []
        for ee in range(len(edge_s)):
            ave_neigh_deg_out1_edges.append(ave_neigh_deg_out_nodes[edge_s[ee][0]])
            ave_neigh_deg_out2_edges.append(ave_neigh_deg_out_nodes[edge_s[ee][1]])
            
        # degree centralities for i and j
        in_deg_cent_nodes_obj = nx.in_degree_centrality(G)
        in_deg_cent_nodes = []
        for nn in range(len(nodes)):
            in_deg_cent_nodes.append(in_deg_cent_nodes_obj[nn])

        in_deg_cent1_edges = []
        in_deg_cent2_edges = []
        for ee in range(len(edge_s)):
            in_deg_cent1_edges.append(in_deg_cent_nodes[edge_s[ee][0]])
            in_deg_cent2_edges.append(in_deg_cent_nodes[edge_s[ee][1]])

        out_deg_cent_nodes_obj = nx.out_degree_centrality(G)
        out_deg_cent_nodes = []
        for nn in range(len(nodes)):
            out_deg_cent_nodes.append(out_deg_cent_nodes_obj[nn])

        out_deg_cent1_edges = []
        out_deg_cent2_edges = []
        for ee in range(len(edge_s)):
            out_deg_cent1_edges.append(out_deg_cent_nodes[edge_s[ee][0]])
            out_deg_cent2_edges.append(out_deg_cent_nodes[edge_s[ee][1]])

        # eigenvector centralities for i and j (EC_i, EC_j)
        # in-direction / "left" eigenvector centrality
        tr = 1
        toler = 1e-6
        while tr == 1:
            try:
                eig_cent_in_nodes_obj = nx.eigenvector_centrality(G,tol = toler)
                tr = 0
            except:
                toler = toler*1e1

        eig_cent_in_nodes = []
        for nn in range(len(nodes)):
            eig_cent_in_nodes.append(eig_cent_in_nodes_obj[nn])

        eig_cent_in1_edges = []
        eig_cent_in2_edges = []
        for ee in range(len(edge_s)):
            eig_cent_in1_edges.append(eig_cent_in_nodes[edge_s[ee][0]])
            eig_cent_in2_edges.append(eig_cent_in_nodes[edge_s[ee][1]])   

        # out-direction / "right" eigenvector centrality
        G_rev = G.reverse()
        tr = 1
        toler = 1e-6
        while tr == 1:
            try:
                eig_cent_out_nodes_obj = nx.eigenvector_centrality(G_rev,tol = toler)
                tr = 0
            except:
                toler = toler*1e1

        eig_cent_out_nodes = []
        for nn in range(len(nodes)):
            eig_cent_out_nodes.append(eig_cent_out_nodes_obj[nn])

        eig_cent_out1_edges = []
        eig_cent_out2_edges = []
        for ee in range(len(edge_s)):
            eig_cent_out1_edges.append(eig_cent_out_nodes[edge_s[ee][0]])
            eig_cent_out2_edges.append(eig_cent_out_nodes[edge_s[ee][1]])
                
        # Katz centralities for i and j (KC_i, KC_j)
        # in-direction / "left" Katz centrality
        ktz_cent_in_nodes_obj = nx.katz_centrality_numpy(G)
        ktz_cent_in_nodes = []
        for nn in range(len(nodes)):
            ktz_cent_in_nodes.append(ktz_cent_in_nodes_obj[nn])

        ktz_cent_in1_edges = []
        ktz_cent_in2_edges = []
        for ee in range(len(edge_s)):
            ktz_cent_in1_edges.append(ktz_cent_in_nodes[edge_s[ee][0]])
            ktz_cent_in2_edges.append(ktz_cent_in_nodes[edge_s[ee][1]]) 

        # out-direction / "right" Katz centrality
        G_rev = G.reverse()
        ktz_cent_out_nodes_obj = nx.katz_centrality_numpy(G_rev)
        ktz_cent_out_nodes = []
        for nn in range(len(nodes)):
            ktz_cent_out_nodes.append(ktz_cent_out_nodes_obj[nn])

        ktz_cent_out1_edges = []
        ktz_cent_out2_edges = []
        for ee in range(len(edge_s)):
            ktz_cent_out1_edges.append(ktz_cent_out_nodes[edge_s[ee][0]])
            ktz_cent_out2_edges.append(ktz_cent_out_nodes[edge_s[ee][1]])    
            
        # Custom food web Jaccard coefficient score 
        eco_jacc_coeff_obj = eco_predictors_helper.eco_jaccard(G,edge_s)
        eco_jacc_coeff_edges = []
        for uu,vv,jj in eco_jacc_coeff_obj:
            eco_jacc_coeff_edges.append([uu,vv,jj])   
        df_eco_jacc_coeff = pd.DataFrame(eco_jacc_coeff_edges, columns=['i','j','EJC'])
        df_eco_jacc_coeff['ind'] = df_eco_jacc_coeff.index 

        # Custom food web resource allocation index of i, j    
        eco_res_alloc_ind_obj = eco_predictors_helper.eco_resource_allocation(G, edge_s)
        eco_res_alloc_ind_edges = []
        for uu,vv,jj in eco_res_alloc_ind_obj:
            eco_res_alloc_ind_edges.append([uu,vv,jj])
        df_eco_res_alloc_ind = pd.DataFrame(eco_res_alloc_ind_edges, columns=['i','j','ERA'])    
        df_eco_res_alloc_ind['ind'] = df_eco_res_alloc_ind.index

        # Custom food web Adamic/Adar index of i, j (AA)
        eco_adam_adar_obj =  eco_predictors_helper.eco_adamic_adar(G, edge_s)
        eco_adam_adar_edges = []
        for uu,vv,jj in eco_adam_adar_obj:
            eco_adam_adar_edges.append([uu,vv,jj])
        df_eco_adam_adar = pd.DataFrame(eco_adam_adar_edges, columns=['i','j','EAA'])
        df_eco_adam_adar['ind'] = df_eco_adam_adar.index

        # Custom food web preferential attachment (degree product) of i, j 
        eco_pref_attach_obj = eco_predictors_helper.eco_preferential_attachment(G, edge_s)
        eco_pref_attach_edges = []
        for uu,vv,jj in eco_pref_attach_obj:
            eco_pref_attach_edges.append([uu,vv,jj])
        df_eco_pref_attach = pd.DataFrame(eco_pref_attach_edges, columns=['i','j','EPA'])
        df_eco_pref_attach['ind'] = df_eco_pref_attach.index 
            
        df_merge = pd.merge(df_eco_jacc_coeff,df_eco_res_alloc_ind, on=['ind','i','j'], sort=False)
        df_merge = pd.merge(df_merge,df_eco_adam_adar, on=['ind','i','j'], sort=False)
        df_merge = pd.merge(df_merge,df_eco_pref_attach, on=['ind','i','j'], sort=False)

        # Food web commmon neighbors i, j 
        eco_com_ne = []
        for ee in range(len(edge_s)):
            eco_com_ne.append(len(eco_predictors_helper.eco_common_neighbors(G,edge_s[ee][0],edge_s[ee][1])))

        eco_com_ne_score = []
        for ee in range(len(edge_s)):
            eco_com_ne_score.append(eco_predictors_helper.eco_common_neighbors_score(G,edge_s[ee][0],edge_s[ee][1]))
            
        # closeness centralities for i and j
        # closeness centrality based on in paths 
        closn_cent_in_nodes_obj = nx.closeness_centrality(G)
        closn_cent_in_nodes = []
        for nn in range(len(nodes)):
            closn_cent_in_nodes.append(closn_cent_in_nodes_obj[nn])

        closn_cent_in1_edges = []
        closn_cent_in2_edges = []
        for ee in range(len(edge_s)):
            closn_cent_in1_edges.append(closn_cent_in_nodes[edge_s[ee][0]])
            closn_cent_in2_edges.append(closn_cent_in_nodes[edge_s[ee][1]])

        # closeness centrality based on out paths
        G_rev = G.reverse()
        closn_cent_out_nodes_obj = nx.closeness_centrality(G_rev)
        closn_cent_out_nodes = []
        for nn in range(len(nodes)):
            closn_cent_out_nodes.append(closn_cent_out_nodes_obj[nn])

        closn_cent_out1_edges = []
        closn_cent_out2_edges = []
        for ee in range(len(edge_s)):
            closn_cent_out1_edges.append(closn_cent_out_nodes[edge_s[ee][0]])
            closn_cent_out2_edges.append(closn_cent_out_nodes[edge_s[ee][1]])
            
        # shortest path between i, j (SP)        
        short_Mat_aux = nx.shortest_path_length(G)
        short_Mat={}
        for ss in range(N):
            value = next(short_Mat_aux)
            short_Mat[value[0]] = value[1]   
        short_path_edges = []
        for ee in range(len(edge_s)):
            if edge_s[ee][1] in short_Mat[edge_s[ee][0]].keys():
                short_path_edges.append(short_Mat[edge_s[ee][0]][edge_s[ee][1]])  
            else:
                #short_path_edges.append(np.inf)
                short_path_edges.append(10000) # some large number other than infinity (infinity throws error)

        # load centralities for i and j (LC_i, LC_j)
        load_cent_nodes_obj = nx.load_centrality(G,normalized=True)
        load_cent_nodes = []
        for nn in range(len(nodes)):
            load_cent_nodes.append(load_cent_nodes_obj[nn])

        load_cent1_edges = []
        load_cent2_edges = []
        for ee in range(len(edge_s)):
            load_cent1_edges.append(load_cent_nodes[edge_s[ee][0]])
            load_cent2_edges.append(load_cent_nodes[edge_s[ee][1]])

        # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
        betw_cent_nodes_obj = nx.betweenness_centrality(G,normalized=True)
        betw_cent_nodes = []
        for nn in range(len(nodes)):
            betw_cent_nodes.append(betw_cent_nodes_obj[nn])

        betw_cent1_edges = []
        betw_cent2_edges = []
        for ee in range(len(edge_s)):
            betw_cent1_edges.append(betw_cent_nodes[edge_s[ee][0]])
            betw_cent2_edges.append(betw_cent_nodes[edge_s[ee][1]])

        neigh_ = {}
        for nn in range(len(nodes)):
            neigh_[nn] = np.where(A[nn,:])[0]

        U, sig, V = np.linalg.svd(A, full_matrices=False)
        S = np.diag(sig)
        Atilda = np.dot(U, np.dot(S, V))
        Atilda = np.array(Atilda)

        f_mean = lambda x: np.mean(x) if len(x)>0 else 0
        # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
        svd_edges = []
        # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
        svd_edges_dot = []
        # average of entries i and j’s neighbors in low rank approximation
        svd_edges_mean = []
        for ee in range(len(edge_s)):
            svd_edges.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
            svd_edges_dot.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
            svd_edges_mean.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]]))        

        # Custom Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
        eco_LHN_edges = eco_predictors_helper.eco_LHN(G,edge_s) 

        U, sig, V = np.linalg.svd(A)
        S = linalg.diagsvd(sig, A.shape[0], A.shape[1])
        S_trunc = S.copy()
        S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(A.shape[0])))]] = 0
        Atilda = np.dot(np.dot(U, S_trunc), V)
        Atilda = np.array(Atilda)

        f_mean = lambda x: np.mean(x) if len(x)>0 else 0
        # an approximation of LRA (LRA-approx)
        svd_edges_approx = []
        # an approximation of dLRA (dLRA-approx)
        svd_edges_dot_approx = []
        # an approximation of mLRA (mLRA-approx)
        svd_edges_mean_approx = []
        for ee in range(len(edge_s)):
            svd_edges_approx.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
            svd_edges_dot_approx.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
            svd_edges_mean_approx.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]])) 
            
        # KNN predictor - D5 / based on Jaccard distance between prey sets
        KNN_in_D5, KNN_out_D5 = KNN_predictors_helper.KNN_predictor_neighborset(G, edge_s, K, "in", False)
        # KNN predictor - D6 / based on Jaccard distance between predator sets
        KNN_in_D6, KNN_out_D6 = KNN_predictors_helper.KNN_predictor_neighborset(G, edge_s, K, "out", False)
        
        curr_len = len(d.keys())
        d.update({'LNTDi': numtriang1_edges, 'LNTDj': numtriang2_edges,\
                  'trophic-i': trophic_level1_edges, 'trophic-j': trophic_level2_edges,\
                  'PPRD':page_rank_pers_edges,'PRDi':page_rank1_edges,'PRDj':page_rank2_edges,\
                  'DSPBCi':betw_cent1_edges,'DSPBCj':betw_cent2_edges,'LCi':load_cent1_edges,\
                  'LCj':load_cent2_edges,'LRA':svd_edges,'dLRA':svd_edges_dot,\
                  'mLRA':svd_edges_mean,'LRA-approx':svd_edges_approx,\
                  'dLRA-approx':svd_edges_dot_approx,'mLRA-approx':svd_edges_mean_approx,\
                  'DSP':short_path_edges,\
                  'ANDIi':ave_neigh_deg_in1_edges,'ANDIj':ave_neigh_deg_in2_edges,\
                  'ANDOi':ave_neigh_deg_out1_edges,'ANDOj':ave_neigh_deg_out2_edges,\
                  'ECIi':eig_cent_in1_edges, 'ECIj':eig_cent_in2_edges,\
                  'ECOi':eig_cent_out1_edges, 'ECOj':eig_cent_out2_edges,\
                  'DCIi':in_deg_cent1_edges,'DCIj': in_deg_cent2_edges, \
                  'DCOi':out_deg_cent1_edges,'DCOj': out_deg_cent2_edges, \
                  'CCIi':closn_cent_in1_edges,'CCIj':closn_cent_in2_edges, \
                  'CCOi':closn_cent_out1_edges,'CCOj':closn_cent_out2_edges, \
                  'KCIi':ktz_cent_in1_edges,'KCIj':ktz_cent_in2_edges, \
                  'KCOi':ktz_cent_out1_edges,'KCOj':ktz_cent_out2_edges,\
                  'LCCDi':clust1_edges,'LCCDj':clust2_edges,\
                  'ECN': eco_com_ne,'ECN-score': eco_com_ne_score,'ELHN':eco_LHN_edges,\
                  'KNN-in-D5':KNN_in_D5,'KNN-out-D5':KNN_out_D5,\
                  'KNN-in-D6':KNN_in_D6,'KNN-out-D6':KNN_out_D6})
        
        number_topo = (len(d.keys())-curr_len)+4  # add the ones in df merge
        if details:
            print(f"number topological features: {number_topo}")
    else:
        number_topo = 0

    if details:
        print(f"number total features: {number_topo + number_attr}")

    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    
    if include_topo:
        df_feat = pd.merge(df_feat, df_merge, on=['ind','i','j'], sort=False)
    
    return df_feat

def creat_full_set(df_t,df_f):
    """ 
    This reads dataframes created for the positive and negative class and joins them with their associated labels.

    Parameters:
    df_t: dataframe of features for the positive class
    df_f: dataframe of features for the negative class

    Returns:
    df_all: dataframe of them joined together
    """

    # Drop any duplicates
    if not df_t.empty:
        df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    if not df_t.empty:
        df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]        
    df_all = pd.concat(df_all)
        
    return df_all

def creat_numpy_files_food_web(dir_results, df_ho, df_tr, attr_dict, include_topo, details, seed):
    
    """ 
    This function does some final clean up and outputs files used for model selection, training, and evaluation

    Parameters:
    dir_results: results folder location
    df_ho: dataframe of features for the hold out set
    df_tr: dataframe of features for the training set
    attr_dict: dictionary of node attributes, if False no attribute features included
    include_topo: whether topological (structural) features included
    details: Flag for whether or not to print details
    seed: Seed for reproducibility

    Returns:
    -------
    feature_set: the set of features include
    Also saves the following files:
    - dir_results+'/X_trainE_'+'cv'+str(nFold) for 5 folds for model selection
    - X_Eseen/X_Eunseen, y_Eseen/y_Eunseen for final model training and evaluation
    """
    
    if include_topo:
        feature_set = ['LNTDi','LNTDj','trophic-i','trophic-j', 'PPRD', 'PRDi', 'PRDj', 'DSPBCi', 'DSPBCj', 'LCi', 'LCj', 'LRA', 'dLRA','mLRA','LRA-approx','dLRA-approx','mLRA-approx','DSP', 'ANDIi', 'ANDIj', 'ANDOi', 'ANDOj', 'ECIi', 'ECIj','ECOi','ECOj','DCIi','DCIj','DCOi','DCOj','CCIi','CCIj','CCOi','CCOj','KCIi','KCIj','KCOi','KCOj','EPA','LCCDi','LCCDj','ECN','ECN-score','ELHN','EJC','ERA','EAA','KNN-in-D5','KNN-out-D5','KNN-in-D6','KNN-out-D6']                     
    else:
        feature_set = []
    if attr_dict:
        for name in attr_dict['attr_names']:
            feature_set.append(name +"_i")
            feature_set.append(name +"_j")
          
        # whether to include different distance predictors
        attr_types = attr_dict['attr_types']
        num_attr = len(attr_types)
        num_numeric = np.sum(attr_types)
        num_bin = len(attr_types) - num_numeric
        include_num_dist = True if num_numeric > 0 else False
        include_bin_dist = True if num_bin > 0 else False
        include_dist = True if num_numeric > 0 and num_bin > 0 else False
        

        if include_num_dist:
            for an in range(0,num_attr):
                if attr_types[an] == 1:
                    feature_set.append(attr_dict['attr_names'][an] +"_ratio")
        
        if include_dist:
            feature_set.append('euclidean')
            feature_set.append('manhattan')
            feature_set.append('cosine')
            feature_set.append('dot_product')
        
        if include_num_dist:
            feature_set.append('euclidean_num')
            feature_set.append('manhattan_num')
            feature_set.append('cosine_num')
            feature_set.append('dot_product_num')
            
        if include_bin_dist:
            feature_set.append('hamming')
            feature_set.append('jaccard_attr')
            
        if include_topo: # only for full model
            feature_set.append('KNN-in-D1')
            feature_set.append('KNN-out-D1')
            feature_set.append('KNN-in-D2')
            feature_set.append('KNN-out-D2')
            if include_bin_dist:
                feature_set.append('KNN-in-D3')
                feature_set.append('KNN-out-D3')
                feature_set.append('KNN-in-D4')
                feature_set.append('KNN-out-D4')
       
    if details:
        print("features used:")
        print(feature_set)
        print("size feature set: " + str(len(feature_set)))

    # replace any infinity values with a large number
    df_ho.replace([np.inf, -np.inf], 100000, inplace=True)
    df_tr.replace([np.inf, -np.inf], 100000, inplace=True)
        
    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    X_train_orig = df_tr    
    y_train_orig = np.array(df_tr.TP)
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        
        X_train = train.loc[:,feature_set]
        X_test = test.loc[:,feature_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=seed) 
        X_train, y_train = sm.fit_resample(X_train, y_train)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        if details:
            print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,feature_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  
    
    if details:
        print("before RandomOverSampler:")
        print("X for training set (features) - numpy matrix shape:")
        print(X_seen.shape)
        print("Y for training set (outcomes for training) - numpy matrix shape:")
        print(y_seen.shape)

    # balance train set with upsampling
    sm = RandomOverSampler(random_state=seed)
    X_seen, y_seen = sm.fit_resample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    
    if details:
        print("after RandomOverSampler:")
        print("created train set ...")
        print("X for training set (features) - numpy matrix shape:")
        print(X_seen.shape)
        print("Y for training set (outcomes for training) - numpy matrix shape:")
        print(y_seen.shape)

    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,feature_set] # subset to only features
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 

    if details:
        print("created holdout set ...")
        print("X for test set (features) - numpy matrix shape:")
        print(X_unseen.shape)
        print("Y for test set (outcomes for evaluation) - numpy matrix shape:")
        print(y_unseen.shape)
    
    return feature_set
     
def model_selection(path_to_data, n_depths, n_ests, details, seed, metric):
    
    """ 
    This function runs cross validation on train set and finds the random forest model parameters which give the best metric performance

    Parameters:
    path_to_data: path to files used for model selection
    n_depths: a list of max_depths to use in hyperparameter selection
    n_ests: a list of n_estimators to use in hyperparameter selection
    details: Flag for whether or not to print details
    seed: Seed for reproducibility
    metric: metric to use for hyperparameter selection

    Returns:
    n_depth: Best max_depth hyperparameter to use
    ne_est: Best n_estimator hyperparameter to use
    """
    
    fmeasure_matrix = np.zeros((len(n_depths),len(n_ests)))
    pr_measure_matrix = np.zeros((len(n_depths),len(n_ests)))
    
    # load train and validation set for each fold
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for nFold in range(1,6):
        
        exec("X_train["+ str(nFold) +"] = np.load( path_to_data + '/X_trainE_cv"+ str(nFold) +".npy')")
        exec("y_train["+ str(nFold) +"] = np.load( path_to_data + '/y_trainE_cv"+ str(nFold) +".npy')")
        exec("X_test["+ str(nFold) +"] = np.load( path_to_data + '/X_testE_cv"+ str(nFold) +".npy')")
        exec("y_test["+ str(nFold) +"] = np.load( path_to_data + '/y_testE_cv"+ str(nFold) +".npy')")
    
    # run a grid search for parameter tuning 
    if details:
        print("start grid search ... ")
    for n_ii, ii in enumerate(n_depths):
        for n_jj, jj in enumerate(n_ests):
        
            f_measure_total = np.zeros((5,2))
            pr_measure_total = np.zeros((5,2))
            
            for cv in range(1,6):
                
                Xtr = X_train[cv]
                ytr = y_train[cv]
                Xts = X_test[cv]
                yts = y_test[cv]
                
                # train the model
                dtree_model = RandomForestClassifier(max_depth=ii,n_estimators=jj,random_state=seed).fit(Xtr, ytr)
                    
                dtree_predictions = dtree_model.predict(Xts)
                dtree_proba = dtree_model.predict_proba(Xts)
                        
                # calculate performance metrics
                cm_dt4 = confusion_matrix(yts, dtree_predictions)
                                  
                _, _, f_measure_aux, _ = precision_recall_fscore_support(yts, dtree_predictions, average=None, zero_division=0)
                avg_precision_aux = average_precision_score(yts, dtree_proba[:,1]) # pr-auc

                f_measure_total[cv-1,:] = f_measure_aux
                pr_measure_total[cv-1,:] = avg_precision_aux
              
            # take average of performance metrics across folds
            mean_f_measure = np.mean(f_measure_total,axis=0)
            mean_pr_measure = np.mean(pr_measure_total,axis=0)
            
            # keep track of average fmeasure for each parameter set
            fmeasure_matrix[n_ii,n_jj] = mean_f_measure[0]
            pr_measure_matrix[n_ii,n_jj] = mean_pr_measure[0]
            
    # find the model parameters which gives the best average fmeasure on 5 fold validation sets    
    if metric == "F-score":
        i,j = np.unravel_index(fmeasure_matrix.argmax(), fmeasure_matrix.shape)
    elif metric == "PR-AUC":
        i,j = np.unravel_index(pr_measure_matrix.argmax(), pr_measure_matrix.shape)
    else:
        print("Valid metric not provided for hyperparameter selection")
        return(-1)
    n_depth = n_depths[i]
    ne_est = n_ests[j]
    if details:
        print(f"Hyper-parameters chosen via {metric}")
        print(f"best hyper-parameters for random forest are: n_depth: {n_depth} and n_estimators: {ne_est}")
    return n_depth, ne_est
        
def heldout_performance(path_to_data, path_to_results, feat_names, n_depth, n_est, details, seed, feature_importance):
    
    """ 
    This function trains a random forest model on the training dataset and performs prediction on the test dataset

    Parameters:
    path_to_data: path to feature matrices for training and test data
    path_to_results: path to save feature importance
    feat_names: used to save the feature names in the importance results
    n_depth: max_depth for random forest hyperparameter
    n_est: n_estimators for random forest hyperparameter
    details: whether to print details 
    seed: seed for reproducibility
    feature_importance: whether to save feature importance results (0- neither, 1 - Gini, 2 - permutation, 3 - both)

    Returns:
    auc_measure: ROC-AUC performance
    avg_precision: PR-AUC performance
    pr_baseline: Baseline for PR-AUC performance
    - Also saves files with feature importance results 
    """
    
    # read data
    X_train = np.load(path_to_data+'/X_Eseen.npy')
    y_train = np.load(path_to_data+'/y_Eseen.npy')
    X_test = np.load(path_to_data+'/X_Eunseen.npy')
    y_test = np.load(path_to_data+'/y_Eunseen.npy')
    
    if details:
        print("in heldout_performance:")
        print("X train shape:")
        print(X_train.shape)
        print("y train shape:")
        print(y_train.shape)
        print("X test shape:")
        print(X_test.shape)
        print("y test shape:")
        print(y_test.shape)
    
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    
    col_mean = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_mean, inds[1])
     
    # train the model
    dtree_model = RandomForestClassifier(n_estimators=n_est,max_depth=n_depth,random_state=seed).fit(X_train, y_train)
    
    # prediction on test set 
    dtree_predictions = dtree_model.predict(X_test)
    dtree_proba = dtree_model.predict_proba(X_test)
      
    # calculate performance metrics
    cm_dt4 = confusion_matrix(y_test, dtree_predictions)
    auc_measure = roc_auc_score(y_test, dtree_proba[:,1])
    avg_precision = average_precision_score(y_test, dtree_proba[:,1])
    
    # calculate the pr-auc baseline for comparison, this is equal to the fraction of positive examples (missing links)
    unique, counts = np.unique(y_test, return_counts=True)
    count_dict = dict(zip(unique, counts))
    total = len(y_test)
    pos_count = count_dict[1]
    pr_baseline = pos_count / total
    if details:
        print(f" size of y test: {total}")
        print(f" count of positive examples (missing links) in y test: {pos_count}")
        print(f" count of negative examples (non-links) in y test: {count_dict[0]}")
    
    precision_total, recall_total, f_measure_total, _ = precision_recall_fscore_support(y_test, dtree_predictions, average=None, zero_division=0)

    if feature_importance == 1 or feature_importance == 3:
        # Gini importance
        if not os.path.isdir(path_to_results):
            os.mkdir(path_to_results)
        feature_importances = pd.DataFrame(dtree_model.feature_importances_,index = feat_names, columns=['importance']).sort_values('importance', ascending=False)
        feature_importances.to_csv(path_to_results +'/gini_importances.csv')
    
    if feature_importance == 2 or feature_importance == 3:
        # Permutation importance
        if not os.path.isdir(path_to_results):
            os.mkdir(path_to_results)
        perm = permutation_importance(dtree_model, X_train, y_train)
        perm_importances = pd.DataFrame(perm.importances_mean,
                            index = feat_names,
                            columns=['importance']).sort_values('importance', ascending=False)
        perm_importances.to_csv(path_to_results +'/perm_importances.csv') 
    
    if details:
        print(f"ROC-AUC: {np.round(auc_measure,2)}")
        print(f"PR-AUC: {np.round(avg_precision,2)}")
        print(f"PR-AUC baseline: {np.round(pr_baseline,2)}")
        print(f"precision: {np.round(precision_total[0],2)}")
        print(f"recall: {np.round(recall_total[0],2)}")
        print(f"f measure: {f_measure_total}")
        
    return auc_measure, avg_precision, pr_baseline

def model_predictions(path_to_data, path_to_results, n_depth, n_est):
    """ 
    This function trains a random forest model on the passed in network + outputs predictions for unobserved links

    Parameters:
    path_to_data: path to held out feature matrices for training data
    path_to_results: path to save predictions
    n_depth: max_depth for random forest parameter
    n_est: n_estimators for random forest parameter

    Returns:
    Nothing is returned but outputs a file with model predictions:
    - path_to_results + '/model_predictions.txt'
    """
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    f = open(path_to_results + '/model_predictions.txt','w')
    
    # read data
    X_train = np.load(path_to_data+'/X_Eseen.npy')
    y_train = np.load(path_to_data+'/y_Eseen.npy')
    
    X_test = np.load(path_to_data+'/X_Eunseen.npy')
    
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    
    col_mean = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_mean, inds[1])

    # train the model
    dtree_model = RandomForestClassifier(n_estimators=n_est,max_depth=n_depth).fit(X_train, y_train)
    
    dtree_predictions = dtree_model.predict(X_test)
    dtree_proba = dtree_model.predict_proba(X_test)
         
    for pred in list(dtree_proba):
        f.write(str(pred[0]) + " " + str(pred[1]) + "\n")
    f.close()

def create_binary_attr_dict(attr_dict):
    """
    This function creates an attribute dictionary mapping nodes to their binary attributes

    Parameters:
    attr_dict: A dictionary mapping nodes to their attribute vectors

    Returns:
    bin_attr_dict: A dictionary mapping nodes to their binary attributes
    """
    
    bin_attr_dict = {}
    attr_types = attr_dict['attr_types']
    for node_key in attr_dict:
        if node_key != 'attr_types' and node_key != 'attr_names':
            attr_vector = attr_dict[node_key]
            attr_vector_bin = [attr_vector[x] for x in range(0,len(attr_vector)) if attr_types[x] == 0]
            bin_attr_dict[node_key] = attr_vector_bin
    return bin_attr_dict
    
def create_attr_dict(node_attr_path, new_id_map, details, attr_types):
    """
    This function creates an attribute dictionary mapping nodes to their attribute vectors

    Parameters:
    node_attr_path: Path to a file that has the node attributes saved
    new_id_map: Mapping from the id's in the node attributes file to the network node id's
    details: Whether or not to print details
    attr_types: The attribute types (0 - binary , 1 - numeric)

    Returns:
    attr_dict: A dictionary mapping nodes to their attributes
    (Note that the dictionary also includes attr_names, a list of names, and attr_types, a dict of attribute types)
    """
    
    assert attr_types, "No attribute types provided"
    
    attr_dict = {}
    attr_dict['attr_types'] = attr_types # map from attribute to type (1 is numeric, 0 is binary)
    attr_file = open(node_attr_path,'r')
    first = True
    num_attr = 0 
    for line in attr_file:
        if first:
            bi_ind = -1
            name_line = line.split(" ")
            name_line[len(name_line)-1] = name_line[len(name_line)-1].replace("\n","")
            attr_head = []
            for i in range(1,len(name_line)):
                attr_head.append(name_line[i])
            attr_dict['attr_names'] = attr_head
            num_attr = len(attr_dict['attr_names'])
            first = False
        else:
            a = line.split()
            curr_id = new_id_map[a[0]]
            curr_attrs = []
            for i in range(1,len(a)):
                if i != bi_ind:
                    curr_attrs.append(a[i])
            np.asarray(a[1:],dtype=np.float64)
            attr_dict[curr_id] = np.asarray(curr_attrs,dtype=np.float64)
            
    assert num_attr == len(attr_types), "Attribute types wrong size"
    
    if details:
        print(f"number of attributes: {num_attr}")
                        
    attr_file.close()
    
    assert num_attr > 0, "No attributes provided for nodes"
    
    return attr_dict

def topol_stacking_attr_food_web(edges_orig_path, node_attr_path, out_name="", include_topo=True, include_attr=False,  details=False, ground_truth=True, attr_types=None, extra_links=False, res_folder='Results', seed=None, lp_fold=0, feature_importance=0, K=3, metric="PR-AUC"):
    """
    This function runs link prediction for a single food web one time and returns ROC-AUC and PR-AUC results
    The default behavior is to use topological predictors only, and construct a holdout set for evaluation
    Results are put in a "Results" folder in the same working directory or in a results folder with the provided name
    
    Required parameters:
        edges_orig_path
            full file path of network edge list
            ** IMPORTANT -- Edges are assumed to be in a RANDOM order to be split into folds **
            ** They must be randomized BEFORE passing the edge list into this function **
            ** If comparing models use the same order for the edge list across models to have the same 5 folds**
            edges are assumed to be listed in from->to (resource, consumer) order in the edge list
                e.g.
                    1 2
                    5 7
                    2 3
        node_attr_path
            full file path of node list - each line is a node and attribute values are separated by spaces
            the first line is a header with the attribute names
                e.g.
                    id attr_1 attr_2
                    0  0.5    0
                    1  0.7    1
                    2  0.2    0        
    
    Optional parameters: 
        out_name 
            a short string used as a unique identifier for results files 
        include_topo 
            boolean - whether to include topological predictors, default is True
        include_attr
            boolean - whether to include attribute predictors, default is False
        details
            boolean - whether to print details of the prediction procedure, default is False
        ground_truth
            boolean - whether the full network is treated as ground truth, default is True
                in the case of True, a hold out network is constructed from the network to evaluate performance
                in the case of False, predictions are outputted for non-edges in the provided network and the performance is not evaluated
        attr_types
            list of 0/1 values representing whether each attribute is binary (0) or numeric (1), default is None
                e.g. [0,0,0,1] would indicate the first three attributes are binary and the final attribute is numeric
        extra_links
            whether to include all links as examples in the training set
        res_folder
            custom results folder name, or else results are put in a folder called 'Results' which is made if it doesn't exist
        seed
            optional seed to set for reproducing the same train / test split, same upsampling, model training and performance results
        lp_fold
            for 5-fold cross-validation, which fold is being treated as the test set
        feature_importance 
            0 - No feature importance saved
            1 - Save Gini importance
            2 - Save permutation importance
            3 - Save both Gini and permutation importance
        K - what value to use for KNN predictors, default is 3
        metric - what metric to use for hyperparameter selection on the train set.
            Currently supports:
            - F-score
            - PR-AUC
    """
                   
    if seed:
        if details:
            print(f"setting seed - {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    if not os.path.isdir(f"./{res_folder}"):
        os.mkdir(f"./{res_folder}")
    if not os.path.isdir(f"./{res_folder}/temp_" + out_name):
        os.mkdir(f"./{res_folder}/temp_" + out_name)
    
    with open(edges_orig_path,'r') as edge_file:
        edge_list = edge_file.readlines()
        for ee in range(0,len(edge_list)):
            edge_list[ee] = tuple(edge_list[ee].split())
        
    #### create a mapping from the node ids in the file to node ids 0 -> n for use in this file
    new_id = 0
    new_id_map = {}
    node_file = open(node_attr_path,'r')
    first = True
    node_list = []
    for line in node_file:
        if not first:
            a = line.split()
            if a[0] not in new_id_map:
                new_id_map[a[0]] = new_id
                node_list.append(new_id)
                new_id += 1
        first = False
    node_file.close()
    
    if details:
        # print mapping to file so results can be examined on a per-link basis 
        path_to_details = f"./{res_folder}/stacking_details_" + out_name 
        if not os.path.isdir(path_to_details):
            os.mkdir(path_to_details)        
        mp = open(path_to_details + '/node_mapping', 'wb')
        pickle.dump(new_id_map, mp)
        mp.close()

    # adjust edges_orig with new internal ids
    for i in range(0,len(edge_list)):
        edge_list[i] = (new_id_map[edge_list[i][0]],new_id_map[edge_list[i][1]])
    
    if details:
        print(f"number original nodes |V|  - {len(node_list)}")
        print(f"number original edges |E| - {len(edge_list)}") 
        print("new id map")
        print(new_id_map)
    
    #### load the node attributes from file 
    if include_attr:
        attrs = create_attr_dict(node_attr_path,new_id_map,details,attr_types)
    else:
        attrs = None
        
    #create initial version of the network with ALL the nodes
    A_net = nx.DiGraph() # create a directed network 
    A_net.add_nodes_from(node_list)
    A_net.add_edges_from(edge_list) # add edges, assumed listed i->j 
                
    if details:
        nx.write_gml(A_net, path_to_details + '/A_net.gml')

    #### construct the holdout and training matriced from the original matrix
    alpha_ = 0.8 # sampling rate for training network
    A_ho, A_tr = gen_tr_ho_networks(A_net, alpha_, edge_list, lp_fold, details)
    
    # in the case where there is no ground_truth network to compare final predictions (predicting from the full)
    if not ground_truth:
        A_tr = A_ho
        A_ho = A_net # in this case the entire passed in network is the observed / hold out 
    
    if details:
        nx.write_gml(A_tr, path_to_details + '/A_tr.gml')          
             
    sample_true_false_edges_small_directed(A_net, A_tr, A_ho, out_name, details, extra_links, res_folder)
                                         
    edge_t_tr = np.loadtxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_tr/edge_f.txt").astype('int')
    if details:
        print("generating topological features for F edges in training (unobserved in hold out), based on training net:")
    df_f_tr = gen_topol_feats_attr_food_web(A_tr, edge_f_tr,attrs, include_topo, K, details)
    if details:
        print("generating topological features for T edges in training (removed from hold out), based on training net:")
    df_t_tr = gen_topol_feats_attr_food_web(A_tr, edge_t_tr,attrs,include_topo, K, details)
    
    if ground_truth:
        edge_t_ho = np.loadtxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt(f"./{res_folder}/temp_" + out_name + "/edge_tf_ho/edge_f.txt").astype('int')
    if details:
        print("generating topological features for F edges in holdout (true non-links in original network), based on hold out network:")
        print("this set is equivalent to the non-links in the holdout network in the case of no ground truth network")
    df_f_ho = gen_topol_feats_attr_food_web(A_ho, edge_f_ho,attrs,include_topo, K, details)
    if details:
        print("generating topological features for T edges in holdout (missing links from original network), based on hold out network:")
        print("this set is empty if there is no ground truth network")
    if ground_truth:
        df_t_ho = gen_topol_feats_attr_food_web(A_ho, edge_t_ho, attrs, include_topo, K, details)
    else:
        df_t_ho = pd.DataFrame()
    
    feat_path = f"./{res_folder}/temp_" + out_name + "/ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = f"./{res_folder}/temp_" + out_name + "/ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')

    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    if details:
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 3)
        print("Training set:")
        display(df_tr)
        print("Test set:")
        display(df_ho)
        feat_path = f"./{res_folder}/temp_" + out_name + "/ho_tr_full_df/"
        if not os.path.isdir(feat_path):
            os.mkdir(feat_path)
        df_ho.to_pickle(feat_path + 'df_ho')
        df_tr.to_pickle(feat_path + 'df_tr')
                           
    #### create and save feature matrices #### 
    dir_output = f"./{res_folder}/temp_" + out_name + "/feature_metrices"  # output path
    if details:
        print("create final training and test objects:")  
    ft_names = creat_numpy_files_food_web(dir_output, df_ho, df_tr, attrs, include_topo, details, seed)
    
    #### perform model selection #### 
    path_to_data = f"./{res_folder}/temp_" + out_name + "/feature_metrices" 
    path_to_results = f"./{res_folder}/stacking_results_" + out_name
    n_depths = [3, 5, 7, 10, 12]
    n_ests = [25, 50, 90, 120] 
    n_depth, n_est = model_selection(path_to_data, n_depths, n_ests, details, seed, metric)
    
    #### evaluation #### 
    if ground_truth:
        rocauc, prauc, pr_baseline = heldout_performance(path_to_data, path_to_results, ft_names, n_depth, n_est, details, seed, feature_importance)
        
        if not details:
            # Delete the temporary files used during link prediction to save storage
            shutil.rmtree(f"./{res_folder}/temp_" + out_name)
        
        return rocauc, prauc, pr_baseline
    else:
        # simply report predictions on the full network, as it is not possible to calculate evaluation metrics
        model_predictions(path_to_data, path_to_results, n_depth, n_est)
        return -1
