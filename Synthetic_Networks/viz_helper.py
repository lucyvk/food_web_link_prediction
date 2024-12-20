# Helper function for visualizing the mixed synthetic networks

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Visualize the connection probabilities for a kernel function
def viz_kernel(kernel):
    ds = np.linspace(0,3)
    ps = []
    for di in ds:
        ps.append(kernel(di))

    s = np.full((len(ds),1),3)
    plt.scatter(ds,ps, c='r', marker=".",s=s)
    plt.xlabel("distance")
    plt.ylabel("edge probability")
    plt.title("attachment kernel")
    plt.xlim(min(ds),max(ds))
    plt.ylim(-0.1,1.1)
    plt.show()
    
# Visualize a mixed network G where there are three predefined blocks used to assign positions
# ax - can optionally pass in an existing axis to visualize this on
def draw_3groups_graph(G,ax,ttl,fs,ps):
    
    colors = []
    pos = []
    ct1 = 0
    ct2 = 0
    ct3 = 0
    for node in G.nodes():
        bl = G.nodes[node]['block']
        if bl == 0:
            colors.append("#000000")
            xpos = 0.33
            ct1+=1
            ct=ct1
        elif bl == 1:
            colors.append("#0000ff") 
            xpos = 0.66
            ct2+=1
            ct=ct2
        else:
            colors.append("#ff0000") 
            xpos = 0.99
            ct3+=1
            ct=ct3
        ypos = ct + 1/15
        ct +=1
        pos.append([xpos,ypos])
        
    if ax:
        ax.set_title(ttl, fontsize=fs,pad=ps)
        nx.draw_networkx(G,pos=pos,node_color=colors,node_size=20,with_labels=False,ax=ax)
    else:
        nx.draw_networkx(G,pos=pos,node_color=colors,node_size=20,with_labels=False)
    
# Visualize a mixed network G where there are position vectors used to assign positions
# ax - can optionally pass in an existing axis to visualize this on
def draw_pos_graph(G, pos, ax):
        
    colors = []
    for node in G.nodes():
        if 'block' in G.nodes[node]: # If block attributes, use these to assign node colors 
            bl = G.nodes[node]['block']
            if bl == 0:
                colors.append("#000000")
            elif bl == 1:
                colors.append("#0000ff") 
            else:
                colors.append("#ff0000")
        else:
            colors.append("#000000")
    if ax:
        nx.draw_networkx(G,node_color=colors,node_size=20,pos=pos,with_labels=False,ax=ax)
    else:
        nx.draw_networkx(G,node_color=colors,node_size=20,pos=pos,with_labels=False)