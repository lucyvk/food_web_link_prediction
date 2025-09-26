# Helper functions used throughout the data processing notebooks
# Author - Lucy Van Kleunen

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import os
import seaborn as sns

### Functions to format strings consistently for metabolic type and movement type features ###

'''
Parameters:
met_list - list of metabolic types

Returns:
new_list - list of formatted metabolic types

'''
def format_mets(met_list):
    new_list = []
    for met in met_list:
        new_list.append(format_met(met))
    return new_list
    
'''
Parameters:
curr_met - metabolic type value

Returns:
formatted metabolic type value

'''
def format_met(curr_met):
    new_met = curr_met.replace(' ','_')
    return f'met_{new_met}'
    
'''
Parameters:
mov_list - list of movement types

Returns:
new_list - list of formatted movement types

'''
def format_movs(mov_list):
    new_list = []
    for mov in mov_list:
        new_list.append(format_mov(mov))
    return new_list
    
'''
Parameters:
curr_mov - movement type value

Returns:
formatted movement type value

'''
def format_mov(curr_mov):
    new_mov = curr_mov.replace(' ','_')
    return f'mov_{new_mov}'

''' Function to check mass values
Parameters:
mass - current mass value

Returns:
nothing is returned just alerts if there is something unexpected in a mass value

'''
def check_mass(mass):
    if (float(mass) < 0 and float(mass) != -999) or float(mass) == 0:
        print(f"weird mass - {mass}")

''' Function to get the genus from a full node name
Makes a very strong assumption that this is just the first part of a two part name

Parameters:
tax - current taxonomic label

Returns:
expected genus value from teh current taxonomic label

'''
def get_genus(tax):
    tax_split = tax.split()
    return tax_split[0]


'''Function to return the final taxonomic category, combining some taxonomic categories
Parameters:
tax - current taxonomic level value

Returns:
final taxonomic category

'''
def get_combined_tax(tax):
    tax = tax.lower() # lowercase for matching strings
    if tax in ['species', 'subspecies', 'variety', 'form']:
        return 'species'
    elif tax in ['genus']:
        return 'genus'
    elif tax in ['family', 'order', 'class', 'phylum', 'kingdom']:
        return 'family+'
    else:
        assert tax == 'na' or tax =='unranked', f'other tax -- {tax}'
        return 'unclassified'

'''Function to plot counts 

Parameters:
ct_dict - count dictionary to visualize
check_num - number to check that overall count aligns with expectations
x_adjust - small adjustment to axis limits
y_adjust - small adjustment to labels
count_of - short substring for the plot title
fig_name - short substring for saving the figure 
FONT_SIZE - figure font size

Returns:
doesn't return anything but plots a count plot

'''
def sorted_count_plot(ct_dict, check_num, x_adjust, y_adjust, count_of, fig_name, FONT_SIZE):
    sorted_dict = sorted(ct_dict.items(), key=lambda x:x[1])
    
    # To check that the overall number is expected
    # assert np.sum(list(ct_dict.values())) == check_num, f"incorrect number - {check_num}"

    labels = []
    heights = []
    for e in sorted_dict:
        labels.append(e[0])
        heights.append(e[1])

    plt.barh(np.arange(len(labels)), heights, tick_label=labels)
    for i in range(0,len(labels)):
        plt.text(heights[i]+y_adjust,i,f'{heights[i]}',fontsize=FONT_SIZE)
    plt.xlim(0,max(heights)+x_adjust)
    plt.xticks(fontsize = FONT_SIZE, rotation=90) 
    plt.yticks(fontsize = FONT_SIZE)
    plt.title(f'count of {count_of}\n out of {check_num}',fontsize=FONT_SIZE)
    plt.savefig(os.path.join('figures',f'{fig_name}.png'),dpi=1000,bbox_inches='tight')
    plt.show()
    

''' Function to create a histogram summarizing a network property across the database 

Parameters:
df - the dataframe over which to plot a histogram
prop_name - the column name over which to plot a histogram
et - ecosystem type (for the plot title, if provided)
folder - the folder in which to save the figure
FONT_SIZE - figure font size
display_names - display names for the network properties

Returns:
doesn't return anything but plots a histogram
'''
def property_histogram(df, prop_name, et, folder, FONT_SIZE, display_names):
    print(prop_name)
    if et:
        plt.title(f"Histogram of {display_names[prop_name]}; eco type: {et}", fontsize=FONT_SIZE)
    else:
        plt.title(f"Histogram of {display_names[prop_name]}", fontsize=FONT_SIZE)
    ax = sns.histplot(data=df[prop_name])
    _, y_max = ax.get_ylim()
    plt.plot([np.min(df[prop_name])]*2,[0,0],label=f'min ({round(np.min(df[prop_name]),4)})')
    plt.plot([np.quantile(df[prop_name],0.25)]*2,[0,y_max],label=f'0.25 quantile ({round(np.quantile(df[prop_name],0.25),4)})')
    plt.plot([np.quantile(df[prop_name],0.5)]*2,[0,y_max],label=f'median ({round(np.quantile(df[prop_name],0.5),4)})')
    plt.plot([np.quantile(df[prop_name],0.75)]*2,[0,y_max],label=f'0.75 quantile ({round(np.quantile(df[prop_name],0.75),4)})')
    plt.plot([np.max(df[prop_name])]*2,[0,0],label=f'max ({round(np.max(df[prop_name]),4)})')
    plt.legend(fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    ax.set_xlabel(display_names[prop_name], fontsize=FONT_SIZE)
    ax.set_ylabel('Count', fontsize=FONT_SIZE)
    plt.rc('axes', unicode_minus=False)
    plt.savefig(f'{folder}/{prop_name}_distribution_{et}.png',dpi=1000,bbox_inches='tight')
    plt.show()