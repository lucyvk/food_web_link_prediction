# This file provides helper functions used in the notebooks for visualizing the results on the food web database.
#
# Author: Lucy Van Kleunen
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import scipy.stats
import pandas as pd
from matplotlib.lines import Line2D

'''
Helper function to check to make sure we have results for all the food webs 

Parameters:
folder - data folder
res_folder - results folder

Returns:
ids of food webs that didn't have results 
'''
def check_all_results(folder, res_folder):

    ct1 = 0 # fully missing auc
    ct2 = 0 # missing some auc results
    ct3 = 0 # fuly missing avp
    ct4 = 0 # missing some avp results
    iterations = 5
    ids_to_run = []
    for x in os.listdir(folder):
        if os.path.isdir(f'{folder}{x}'):
            fw_name = x.split('_')[0]
            fw_id = x.split('_')[1]
            out_file = os.path.join(res_folder,f'stacking_auc_{fw_name}_{fw_id}.csv')
            if os.path.isfile(out_file): 
                with open(out_file,'r') as obj:
                    rd = csv.reader(obj)
                    rwct = 0
                    for row in rd:
                        for el in row:
                            if not el.replace(".","").isnumeric():
                                print(f'error: {fw_name} {fw_id} {el}')
                            if float(el) == 0:
                                print(f'error: {fw_name} {fw_id} {el}')
                        rwct += 1
                if rwct != iterations:
                    print(f"not all {iterations}  iterations auc - {out_file}")
                    ct2 += 1
                    ids_to_run.append(fw_id)
            else:
                print(f"missing auc - {out_file}")
                ct1 += 1
                ids_to_run.append(fw_id)
            out_file = os.path.join(res_folder,f'stacking_avp_{fw_name}_{fw_id}.csv')
            if os.path.isfile(out_file): 
                with open(out_file,'r') as obj:
                    rd = csv.reader(obj)
                    rwct = 0
                    for row in rd:
                        for el in row:
                            if not el.replace(".","").isnumeric():
                                print(f'error: {fw_name} {fw_id} {el}')
                            if float(el) == 0:
                                print(f'error: {fw_name} {fw_id} {el}')
                        rwct += 1
                if rwct != iterations:
                    print(f"not all {iterations}  iterations avp- {out_file}")
                    ct4 += 1
                    ids_to_run.append(fw_id)
            else:
                print(f"missing avp - {out_file}")
                ct3 += 1
                ids_to_run.append(fw_id)
    print(f"count fully missing auc {ct1}")
    print(f"count missing some auc results {ct2}")
    print(f"count fully missing avp {ct3}")
    print(f"count missing some avp results {ct4}")
    print(f"food web ids to still run: {ids_to_run}")
    return ids_to_run

'''
Helper function to write average results to file (for running the regressions with food web metadata)

Parameters:
folder - data folder
Results_Folder - results folder of where to put this summarized file
num_it - the number of iterations of results to average over for a single food web
ids_to_skip - list of food web ids with results that should be skipped
folder_shorter_names - dictionary mapping a subset of the food webs between shorter and longer versions of their names

Returns:
doesn't return anything but writes a CSV file with the summarized results per food web
'''
def food_web_result_to_file(folder, Results_Folder, num_it, ids_to_skip, folder_shorter_names):
        
    Full_Res_File = os.path.join(Results_Folder,f'food_web_lp_res_{Results_Folder}.csv')
    rows = []
    for x in os.listdir(folder):
        if os.path.isdir(f'{folder}/{x}'):
            fw_name = x.split('_')[0]
            fw_id = x.split('_')[1]
            if fw_name in folder_shorter_names:
                fw_name = folder_shorter_names[fw_name]
            roc_out_file = os.path.join(Results_Folder,f'stacking_auc_{fw_name}_{fw_id}.csv')
            pr_out_file = os.path.join(Results_Folder,f'stacking_avp_{fw_name}_{fw_id}.csv')
            if fw_id not in ids_to_skip:
                row_to_write = []
                with open(roc_out_file, 'r') as sim_output, open(pr_out_file, 'r') as sim_output2:
                    sim_reader = csv.reader(sim_output)
                    struc = []
                    attr = []
                    full = []
                    for row in sim_reader:
                        struc.append(float(row[0])) # structure
                        attr.append(float(row[1])) # attributes 
                        full.append(float(row[2])) # full
                    # save averages
                    assert len(struc) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    assert len(attr) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    assert len(full) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    struc_roc = np.mean(struc) # structure
                    attr_roc = np.mean(attr) # attributes 
                    full_roc = np.mean(full) # full
                    
                    sim_reader2 = csv.reader(sim_output2)
                    struc = []
                    attr = []
                    full = []
                    for row in sim_reader2:
                        struc.append(float(row[0])) # structure
                        attr.append(float(row[1])) # attributes 
                        full.append(float(row[2])) # full
                    assert len(full) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    assert len(struc) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    assert len(attr) == num_it, f"wrong size {Results_Folder} {fw_name}"
                    # save averages
                    struc_pr = np.mean(struc) # structure
                    attr_pr = np.mean(attr) # attributes 
                    full_pr = np.mean(full) # full
            new_row = [fw_id,struc_roc,attr_roc,full_roc,struc_pr,attr_pr,full_pr]
            rows.append(new_row)
    # sort by food web id
    df = pd.DataFrame(rows, columns=['net_id','struc_roc_mean','attr_roc_mean','full_roc_mean','struc_pr_mean','attr_pr_mean','full_pr_mean'])
    df = df.astype({'net_id': int})
    df.sort_values(by='net_id', inplace=True)
    # output to CSV
    df.to_csv(Full_Res_File, encoding='utf-8', index=False)

'''
Helper function to plot the predictive performance results

Parameters:
Data_Folder - data folder
Results_Folders - a list of results folders to summarize over
ids_to_skip - list of food web ids with results that should be skipped
out_stem - Used to title the plots
type1 - short substring used for finding resutls CSVs based on the metric summarized (auc==ROC-AUC or avp==PR-AUC)
type2 - short substring used for labelling the plots (ROC or PR)
num_it - the number of iterations of results to average over for a given food web
bl_check - if True print out results for checks of if individual folds or food web average results are under respective metric baselines
plot_means - if True plot the means on the plots
subplots - use this to indicate which subplots in a larger figure should be used to plot results
skip - use this to indicate if the best model results should be skipped or shown
num_webs - the number of overall webs plotted across
folder_shorter_names - dictionary mapping a subset of the food webs between shorter and longer versions of their names
FONT_SIZE - plot font size
full_color - color for the full model results
struc_color - color for the structure model results
attr_color - color for the attribute model results

Returns:
Doesn't return anything but plots summary plots on passed in subplots
'''
def food_web_result_plots(Data_Folder, Results_Folders, ids_to_skip, out_stem, type1, type2, num_it, bl_check,\
                          plot_means, subplots, skip, num_webs, folder_shorter_names, FONT_SIZE, full_color, struc_color, attr_color):
    
    best_model = [0,0,0] # counts of which is the best model full / structure / attributes 
    res_plot = np.zeros((num_webs-len(ids_to_skip),3)) # topo, traits, full
    i = 0
    struc_bl_ct = 0
    full_bl_ct = 0
    attr_bl_ct = 0
    struc_bl_avg = 0
    full_bl_avg = 0
    attr_bl_avg = 0
    for x in os.listdir(Data_Folder):
        if os.path.isdir(os.path.join(Data_Folder,x)):
            fw_name = x.split('_')[0]
            if fw_name in folder_shorter_names:
                fw_name = folder_shorter_names[fw_name]
            fw_id = x.split('_')[1]
            if fw_id not in ids_to_skip: 

                pr_bls = []
                if type2 == 'PR':
                    # get PR baseline from file (should be exactly or nearly the same per food web across)
                    for Results_Folder in Results_Folders:
                        with open(f'./{Results_Folder}/avp_baseline_{fw_name}_{fw_id}.csv','r') as blout:
                            blreader = csv.reader(blout)
                            for row in blreader:
                                pr_bls.append(float(row[0]))
    
                pr_bl = np.mean(pr_bls)
    
                ct = 0 
                for Results_Folder in Results_Folders:
                    out_file = f'./{Results_Folder}/stacking_{type1}_{fw_name}_{fw_id}.csv'
                    bl = 0.5 if type2 =='ROC' else pr_bl
                    with open(out_file, 'r') as sim_output:
                        sim_reader = csv.reader(sim_output)
                        for row in sim_reader:
                            res_plot[i,0] = res_plot[i,0] + float(row[0]) # structure
                            ct+=1
                            if float(row[0]) <= bl:
                                struc_bl_ct += 1
                            res_plot[i,1] = res_plot[i,1] + float(row[1]) # attributes
                            if float(row[1]) <= bl:
                                attr_bl_ct += 1
                            res_plot[i,2] = res_plot[i,2] + float(row[2]) # full
                            if float(row[2]) <= bl:
                                full_bl_ct += 1
    
                assert ct == num_it, "num it unexpected"
                # Save average across 5 x 5 folds
                res_plot[i,0] = res_plot[i,0]/num_it # structure
                if res_plot[i,0] <= bl:
                    struc_bl_avg += 1
                    if bl_check:
                        print(f"average structure under baseline: {fw_name}")
                res_plot[i,1] = res_plot[i,1]/num_it # attributes
                if res_plot[i,1] <= bl:
                    attr_bl_avg +=1
                    if bl_check:
                        print(f"average attribute under baseline: {fw_name}")
                res_plot[i,2] = res_plot[i,2]/num_it # full
                if res_plot[i,2] <= bl:
                    full_bl_avg +=1
                    if bl_check:
                        print(f"average full under baseline: {fw_name}")
                if res_plot[i,0] > res_plot[i,1] and res_plot[i,0] > res_plot[i,2]:
                    best_model[1] = best_model[1] + 1 # structure
                if res_plot[i,1] > res_plot[i,0] and res_plot[i,1] > res_plot[i,2]:
                    best_model[2] = best_model[2] + 1 # atributes
                if res_plot[i,2] > res_plot[i,0] and res_plot[i,2] > res_plot[i,1]:
                    best_model[0] = best_model[0] + 1 # full 
                i+=1
            

    if bl_check:
        print(f"individual t/t split structure under baseline: {struc_bl_ct}")
        print(f"individual t/t split attribute under baseline - {attr_bl_ct}")
        print(f"individual t/t split full under baseline - {full_bl_ct}") 
        print(f"avg for a web structure under baseline: {struc_bl_avg}")
        print(f"avg for a web attribute under baseline - {attr_bl_avg}")
        print(f"avg for a web full under baseline - {full_bl_avg}")
    
    struc_res = np.array(res_plot[:,0])
    attr_res = np.array(res_plot[:,1])
    full_res = np.array(res_plot[:,2])

    assert len(struc_res) == num_webs-len(ids_to_skip), "wrong size"
    
    aucs = [full_res, struc_res, attr_res]
    medians = [round(np.median(full_res),2), round(np.median(struc_res),2), round(np.median(attr_res),2)]
    means = [round(np.mean(full_res),2), round(np.mean(struc_res),2), round(np.mean(attr_res),2)]
    stddevs = [round(np.std(full_res),2), round(np.std(struc_res),2), round(np.std(attr_res),2)]

    print(f"full std - {stddevs[0]}")
    print(f"struc std - {stddevs[1]}")
    print(f"attr std - {stddevs[2]}")
    
    # Is there a difference in mean (via paired t-tests as each is for the same set of food webs)
    print("T test - difference in mean")
    res1 = scipy.stats.ttest_rel(struc_res,attr_res)
    s1 = res1.statistic
    p1 = res1.pvalue
    print(f"structure vs. attribute - {s1} , {p1}")
    res2 = scipy.stats.ttest_rel(full_res,attr_res)
    s2 = res2.statistic
    p2 = res2.pvalue
    print(f"full vs. attribute - {s2} , {p2}")
    res3 = scipy.stats.ttest_rel(full_res,struc_res)
    s3 = res3.statistic
    p3 = res3.pvalue
    print(f"full vs. structure - {s3} , {p3}")
    ps = [p1,p2,p3]
    
    # Peform a multiple testing comparison adjustment to the p-values
    aps = scipy.stats.false_discovery_control(ps, method='bh')

    sigs = []
    sig_stars = []
    for ap in aps:
        if ap < 0.05:
            sigs.append("SIGNIFICANT")
        else:
            sigs.append("NOT SIGNIFICANT")
        if ap < 0.0001:
            sig_stars.append("****")
        elif ap < 0.001:
            sig_stars.append("***")
        elif ap < 0.01:
            sig_stars.append("**")
        elif ap < 0.05:
            sig_stars.append("*")
        else:
            sig_stars.append("")
    
    print("T test - difference in mean, after multiple testing adjustment:")
    print(f"structure vs. attribute - {s1} , {aps[0]} {sigs[0]} {sig_stars[0]}")
    print(f"full vs. attribute - {s2} , {aps[1]} {sigs[1]} {sig_stars[1]}")
    print(f"full vs. structure - {s3} , {aps[2]} {sigs[2]} {sig_stars[2]}")

    plt.subplot(subplots[0][0], subplots[0][1], subplots[0][2])
    plt.title(f'{out_stem}',fontsize=FONT_SIZE)
    plt.xlim([0,4])
    if type2 == 'ROC':
        plt.hlines(0.5, 0, 5, linestyles='dashed', color = 'k', linewidth=0.8)
        plt.ylim([0.4,1.13])
        y_range = 0.73
    else:
        plt.ylim([0,1.2])
        y_range = 1.2
    bar_y_init = 1 + (y_range*0.033)
    level_adjust = (y_range * 0.04)
    text_y_adjust = -(0.03*y_range)
    plt.hlines(1, 0, 5, linestyles='dashed', color = 'k', linewidth=0.8)
    bar_tip_y_adjust = -(y_range * 0.02)
    y_bar_levels = [1,2,0]
    axes = plt.gca()
    bplot = axes.boxplot(aucs, patch_artist=True)
    colors = [full_color,struc_color,attr_color]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set_color('black')
    ct = 1
    if plot_means:
        for me in means:
            plt.text(ct+0.2,me,f"{me}",color='k',fontsize=FONT_SIZE)
            ct+=1
    # Significant differences:
    # Code adapted from - https://rowannicholls.github.io/python/graphs/ax_based/boxplots_significance.html
    # Structure vs. attribute
    level = y_bar_levels[0]
    bar_height = bar_y_init + level*level_adjust
    bar_tips = bar_height + bar_tip_y_adjust
    print(bar_height)
    plt.plot([2, 2, 3, 3],[bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    text_height = bar_height + text_y_adjust
    print(text_height)
    plt.text((2 + 3) * 0.5, text_height, sig_stars[0], ha='center', va='bottom', c='k', fontsize=FONT_SIZE)
    # Full vs. attribute
    level = y_bar_levels[1]
    bar_height = bar_y_init + level*level_adjust
    bar_tips = bar_height + bar_tip_y_adjust
    plt.plot([1, 1, 3, 3],[bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    text_height = bar_height + text_y_adjust
    plt.text((1 + 3) * 0.5, text_height, sig_stars[1], ha='center', va='bottom', c='k', fontsize=FONT_SIZE)
    # Full vs. structure
    level = y_bar_levels[2]
    bar_height = bar_y_init + level*level_adjust
    bar_tips = bar_height + bar_tip_y_adjust
    plt.plot([1, 1, 2, 2],[bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    text_height = bar_height + text_y_adjust
    plt.text((1 + 2) * 0.5, text_height, sig_stars[2], ha='center', va='bottom', c='k', fontsize=FONT_SIZE)
    plt.ylabel(f"{type2}-AUC",fontsize=FONT_SIZE)
    plt.setp(axes, xticks=[1,2,3], xticklabels=['Full', 'Structure', 'Attribute']) 
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    if not skip:
        plt.subplot(subplots[1][0], subplots[1][1], subplots[1][2])
        axes = plt.gca()
        bars = plt.bar(range(3), best_model, color ='black')
        bars[0].set_color(full_color)
        bars[1].set_color(struc_color)
        bars[2].set_color(attr_color)
        plt.ylabel("Count of webs",fontsize=FONT_SIZE)
        ct = 0
        for med in medians:
            plt.text(ct-0.1, best_model[ct] + 3, str(best_model[ct]),fontsize=FONT_SIZE)
            ct += 1
        plt.ylim([0,290])
        plt.title(f'Best Model {type2}-AUC',fontsize=FONT_SIZE)
        plt.setp(axes, xticks=[0,1,2], xticklabels=['Full', 'Structure', 'Attribute']) 
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)

'''
Helper function to list the aggregated feature importance results

Parameters:
imp_type - importance type
ids_to_skip - list of food web ids with results that should be skipped
eco_type - current ecosystem type (for naming output files)
res_folders - folders of results to summarize over

Returns:
Saves CSV files listing all of the importance results summarized per food web and overall for inspection
Returns the top 10 of the overall results
'''
def enumerate_importance(imp_type, ids_to_skip, eco_type, res_folders):
    imps_overall = {}
    for fw_id in range(0,290):
        if fw_id not in ids_to_skip:
            imps_curr = {}
            for fold in range(0,5):
                for res_folder in res_folders:
                    res = pd.read_csv(os.path.join(res_folder,f'stacking_results_{fw_id}_full_{fold}',f'{imp_type}_importances.csv'))
                    for _, row in res.iterrows():
                        curr_feat = row['Unnamed: 0'] # Note -- feature names are unnamed in current feature importance files
                        curr_imp = row['importance']
                        if curr_feat not in imps_curr:
                            imps_curr[curr_feat] = [curr_imp]
                        else:
                            imps_curr[curr_feat].append(curr_imp)
            # Average across the 5 x 5 folds for each feature / food web
            for feat in imps_curr:
                assert len(imps_curr[feat]) == 25
                mean_curr = np.mean(imps_curr[feat])
                imps_curr[feat] = mean_curr
                if feat not in imps_overall:
                    imps_overall[feat] = [mean_curr]
                else:
                    imps_overall[feat].append(mean_curr)
            # Save this to file to see the top within each food web
            df_list_curr = list(zip(imps_curr.keys(), imps_curr.values()))
            imps_df_curr = pd.DataFrame(df_list_curr, columns=['feature','importance'])
            imps_df_curr.sort_values('importance', ascending=False, inplace=True)
            imps_df_curr.to_csv(os.path.join('Summarized_Results',f'feature_importance_{imp_type}_aggregated_{eco_type}_{fw_id}.csv'), index=False)  
    for feat in imps_overall:
        imps_overall[feat] = np.mean(imps_overall[feat])
    df_list = list(zip(imps_overall.keys(), imps_overall.values()))
    imps_df = pd.DataFrame(df_list, columns=['feature','importance'])
    imps_df.sort_values('importance', ascending=False, inplace=True)
    imps_df.to_csv(os.path.join('Summarized_Results',f'feature_importance_{imp_type}_aggregated_{eco_type}_overall.csv'), index=False)  
    display(imps_df.head(20))
    return imps_df.head(10)['feature'].tolist()

'''
Helper function to visualize the aggregated feature importance results

Parameters:
imp_file - feature importance summary file to read results from 
feature_set - the set of features to visualize results for 
x_lab - x label
top_n - number of top features to show out of the feature set
y_ticks - if False don't show y ticks
ttl - plot title if provided
FONT_SIZE - font size for the plot
x_lab_adjusts - small adjustments for visualizing the feature type symbols
legend_y_adjust - small adjustment for placing the legend
to_scatter - whether to visualize feature type symbols
show_legend - whether to show a legend
mksiz1 - marker size for the feature type symbols
full_color - color for the full model results
struc_color - color for the structure model results
attr_color - color for the attribute model results
feature_types - corresponding types for the features
display_dict - maps display names for all the features

Returns:
No returns both plots the aggreagated feature importance plot as specified
'''
def visualize_importance(imp_file, feature_set, x_lab, top_n, y_ticks, ttl, FONT_SIZE, x_lab_adjusts, legend_y_adjust, to_scatter, show_legend, mksiz1,\
                         full_color,struc_color,attr_color, feature_types, display_dict):
    imp_df = pd.read_csv(imp_file)
    imp_mask = imp_df['feature'].isin(feature_set)
    imp_df = imp_df[imp_mask]
    imp_df.sort_values('importance', ascending=False, inplace=True)
    display(imp_df)
    bar_colors = []
    feature_labels = imp_df['feature'].tolist()
    importances = imp_df['importance'].tolist()
    if to_scatter:
        scatter_full = [[],[]]
        scatter_attr = [[],[]]
        scatter_struc = [[],[]]
    i = 0 
    for feat in feature_labels:
        if feat in feature_types and feature_types[feat] == 'structure':
            bar_colors.append(struc_color)
            if to_scatter:
                scatter_struc[0].append(importances[i]-x_lab_adjusts[0])
                scatter_struc[1].append(i)
        elif feat in feature_types and feature_types[feat] == 'full':
            bar_colors.append(full_color)
            if to_scatter:
                scatter_full[0].append(importances[i]-x_lab_adjusts[1])
                scatter_full[1].append(i)
        else:
            bar_colors.append(attr_color)
            if to_scatter:
                scatter_attr[0].append(importances[i]-x_lab_adjusts[2])
                scatter_attr[1].append(i)
        i+=1
    feature_labels = [display_dict[x] for x in feature_labels]
    # only show first n
    if top_n:
        feat_set_len = len(feature_set) 
        if feat_set_len > top_n:
            feature_set = feature_set[0:top_n]
            feature_labels = feature_labels[0:top_n]
            bar_colors = bar_colors[0:top_n]
            importances= importances[0:top_n]
            if to_scatter:
                scatter_full_new = [[],[]]
                scatter_attr_new = [[],[]]
                scatter_struc_new = [[],[]]
                for ii in range(0,len(scatter_struc[0])):
                    if scatter_struc[1][ii] < top_n:
                        scatter_struc_new[0].append(scatter_struc[0][ii])
                        scatter_struc_new[1].append(scatter_struc[1][ii])
                for ii in range(0,len(scatter_attr[0])):
                    if scatter_attr[1][ii] < top_n:
                        scatter_attr_new[0].append(scatter_attr[0][ii])
                        scatter_attr_new[1].append(scatter_attr[1][ii])
                for ii in range(0,len(scatter_full[0])):
                    if scatter_full[1][ii] < top_n:
                        scatter_full_new[0].append(scatter_full[0][ii])
                        scatter_full_new[1].append(scatter_full[1][ii])
        else:
            if to_scatter:
                scatter_full_new = scatter_full
                scatter_attr_new = scatter_attr
                scatter_struc_new = scatter_struc
    plt.barh(list(range(0,len(feature_set))), importances, tick_label=feature_labels, color=bar_colors)
    if to_scatter:
        plt.scatter(scatter_struc_new[0],scatter_struc_new[1],s=mksiz1,marker="1",color='k')
        plt.scatter(scatter_attr_new[0],scatter_attr_new[1],s=mksiz1,marker="2",color='k')
        plt.scatter(scatter_full_new[0],scatter_full_new[1],s=500,marker=".",color='k')
    if not y_ticks:
        plt.gca().set_yticklabels([])
    plt.xticks(rotation=90, fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel(x_lab, fontsize=FONT_SIZE)
    plt.gca().invert_yaxis() # make the features descending rather than ascending by importance vertically
    if show_legend:
        legend_elements = [Line2D([0], [0], linewidth=30, marker='.', markersize=22,color=full_color, mfc='k', mec='k', label='Full'),
               Line2D([0], [0], linewidth=30, marker='1', markersize=22, markeredgewidth=2, color=struc_color, mfc='k', mec='k',label='Structure'),
               Line2D([0], [0], linewidth=30, marker='2', markersize=22,markeredgewidth=2, color=attr_color, mfc='k', mec='k',label='Attribute')]
        plt.legend(handles=legend_elements, loc='lower center', ncol=1, bbox_to_anchor=(0.5,legend_y_adjust), fontsize=FONT_SIZE, borderpad=0.6)
    if ttl:
        plt.title(ttl,fontsize=FONT_SIZE)