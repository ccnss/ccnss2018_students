# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 05:16:40 2018

@author: J-Y Moon
"""
import matplotlib.pyplot as plt    # import matplotlib
import numpy as np                 # import numpy
import scipy as sp                 # import scipy
from scipy import sparse           # import sparse module from scipy
import networkx as nx              # import networkx 



# code for generating connectivity matrix from a connectivity list, for small or near-full network

def net_mat1(net_list):
    len_net = np.amax(net_list)+1
    len_list = len(net_list)
    net_mat = np.zeros((len_net,len_net))
    for i in range(len_list):
        net_mat[ net_list[i,0] , net_list[i,1] ] = net_list[i,2]
   
    return net_mat   


# code for generating connectivity matrix from a connectivity list, for large yet sparse network

def net_mat2(net_list):
    len_net =  np.amax(net_list)+1
    net_mat = sp.sparse.coo_matrix((net_list[:,2], (net_list[:,0],net_list[:,1])), shape=(len_net,len_net) )
    # net_mat_csc = sp.sparse.csc_matrix(net_mat)
    
    return net_mat



# computes degree of input network,
# and also the cumulative probability distribution for the degree, and prints the resulting figures

def net_degree_plot(net_mat,net_name):

    net_degree = np.sum(net_mat,axis=0)
    net_degree_unique, net_degree_counts  = np.unique(net_degree, return_counts=True)
    net_degree_cumul = np.zeros(len(net_degree_unique))

    #print(net_degree_unique)
    #print(net_degree_counts)

    net_degree_cumul[-1]=net_degree_counts[-1]
    for i in range(len(net_degree_unique)-2,-1,-1):
        net_degree_cumul[i] = net_degree_cumul[i+1]+net_degree_counts[i]

  
    plt.figure(figsize=(15,5))
    
    plt.subplot( 1, 2, 1 )
    plt.plot(net_degree_unique, net_degree_cumul,'C0o')
    plt.xlabel('degree')
    plt.ylabel('cumulative dist.')
    plt.title(net_name)
    
    plt.subplot( 1, 2, 2 )
    plt.loglog(net_degree_unique, net_degree_cumul,'C0o')
    plt.xlabel('degree')
    plt.ylabel('cumulative dist.')
    plt.title(net_name)
    plt.show      
    
    
    
# calculates clustering coefficient of a given network and a node

def clustering_coef(net_mat, node_number):
  
    neighbors = np.nonzero(net_mat[node_number])[0]
    neighbors_N = neighbors.shape[0]

    if neighbors_N == 1: return 0
    links = 0
    for w in neighbors:
        for u in neighbors:
            neighbors_neighbors = np.nonzero(net_mat[w])[0]
            if u in neighbors_neighbors: links += 0.5
                
    return 2.0*links/(neighbors_N *(neighbors_N -1))  



# calculate distance matrix from a given connectivity matrix

def net_mat_distance(net_mat):
    net_mat = np.matrix(net_mat)
    net_mat_N = len(net_mat)
    net_mat_distance = np.zeros((net_mat_N,net_mat_N))
    net_mat_product = net_mat.copy()
    D = net_mat.copy()
    T = net_mat.copy()
    i=3

    for k in range(net_mat_N):
        net_mat_product = net_mat_product*net_mat
        net_where = np.where(net_mat_product > 0)  
        D[net_where]=1
        T = T+D 
        net_mat_distance = i*D - T
        i = i+1
        if len(np.where(net_mat_distance<=0)[0]) == 0:
            break
        
    return net_mat_distance


# calculate characteristic path length and efficiency from a given distance matrix

def net_L_E(net_mat_d):
    net_mat_d = np.matrix(net_mat_d)
    N = net_mat_d.shape[0]
    
    L = 1/N * 1/(N-1) * (np.sum(net_mat_d)-np.trace(net_mat_d))
    E = 1/N * 1/(N-1) * (np.sum(1/net_mat_d)-np.trace(1/net_mat_d))
    
    return L, E


# calculates betweenness centrality from a given connectivity matrix

def betweenness_central(net_mat,normalized=True):
    net_mat = np.matrix(net_mat)
    graph = nx.to_networkx_graph(net_mat)
    bc = nx.betweenness_centrality(graph,normalized=normalized) # dictionary where key = node
    bc = np.array([bc[i] for i in range(len(bc))])
    return bc