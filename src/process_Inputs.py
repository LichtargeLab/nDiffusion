'''@author: minhpham'''

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csgraph, identity
from collections import Counter

def getGraph (net_lst = '../data/networks/STRINGv10.txt'):
          G = nx.read_edgelist(open(net_lst, 'r'), data=(('weight',float),))
          graph_node = list(G.nodes())
          adjMatrix = nx.to_scipy_sparse_matrix(G)
          node_degree = dict(nx.degree(G))
          G_degree = node_degree.values()
          return G, graph_node, adjMatrix, node_degree, G_degree

def getDiffusionParam (adjMatrix):
          L = csgraph.laplacian(adjMatrix, normed=True)
          n = adjMatrix.shape[0]
          I = identity(n, dtype='int8', format='csr')
          axisSum = coo_matrix.sum(np.abs(L), axis=0)
          sumMax = np.max(axisSum)
          diffusionParameter = (1 / float(sumMax))
          ps = (I + (diffusionParameter * L))
          return ps

def readInput (fl):
    lst = []
    for line in open(fl).readlines():
        line = line.strip('\n').strip('\r').split('\t')
        lst.append(line[0])
    return lst

def getIndex (lst, graph_node):
    index = []
    for i in lst:
        ind = graph_node.index(i)
        index.append(ind)
    return index

def getDegree (pred_node, node_degree):
    pred_degree = []
    for i in pred_node:
        pred_degree.append(node_degree[i])
    pred_degree_count = dict(Counter(pred_degree))
    return pred_degree_count

def parseGeneInput (fl1, fl2, graph_node):
    ### Parsing input files
    group1 = readInput(fl1)
    group2 = readInput(fl2)
    overlap = list(set(group1).intersection(group2))
    ### Mapping genes into the network
    group1_node = list(set(group1).intersection(graph_node))
    group2_node = list(set(group2).intersection(graph_node))
    overlap_node = list(set(overlap).intersection(graph_node))
    other = list(set(graph_node) - set(group1_node) - set(group2_node))
    group1_only_node = list(set(group1_node)-set(overlap_node))
    group2_only_node = list(set(group2_node)-set(overlap_node))
    print("{} genes mapped (out of {}) in {}\n {} genes mapped (out of {}) in {}\n {} overlapped and mapped (out of {})\n".format(len(group1_node), len(group1), fl1, len(group2_node), len(group2) fl2, len(overlap_node), len(overlap)))
    ### Getting indexes of the genes in the network node list
    group1_only_index = getIndex(group1_only_node, graph_node)
    group2_only_index = getIndex(group2_only_node, graph_node)
    overlap_index = getIndex(overlap_node, graph_node)
    other_index = list(set(range(len(graph_node))) - set(group1_only_index) - set(group2_only_index)-set(overlap_index))
    ### Getting counter dictionaries for the connectivity degrees of the genes
    group1_only_degree_count = getDegree(test_only_node)
    group2_only_degree_count = getDegree(GS_only_node)
    overlap_degree_count = getDegree(overlap_node)
    ### Combining these features into dictionaries
    GP1_only_dict={'node':group1_only_node, 'index':group1_only_index, 'degree': group1_only_degree_count}
    GP2_only_dict={'node':group2_only_node, 'index':group2_only_index, 'degree': group2_only_degree_count}
    overlap_dict={'node':overlap_node, 'index':overlap_index, 'degree': overlap_degree_count}
    other_dict={'node':other, 'index':other_index} 
    
    return GP1_only_dict, GP2_only_dict, overlap_dict, other_dict

def getDegreeNode(G_degree, node_degree, other):
    degree_nodes = {}
    for i in set(G_degree):
        degree_nodes[i] = []
        for y in node_degree:
            if node_degree[y] == i and y in other:
                degree_nodes[i].append(y)
        degree_nodes[i] = list(set(degree_nodes[i]))
        random.shuffle(degree_nodes[i])
    return degree_nodes


'''
pickle.dump(G, open(network_path+'G.csv', 'wb'))
pickle.dump(graph_node, open(network_path+'graph_node.csv', 'wb'))
pickle.dump(adjMatrix, open(network_path+'adjMatrix.csv', 'wb'))
pickle.dump(node_degree, open(network_path+'node_degree.csv', 'wb'))
pickle.dump(G_degree, open(network_path+'G_degree.csv', 'wb'))

G: 49.306450843811035 seconds
graph_node: 0.0009710788726806641 seconds
adjMatrix: 26.21620202064514 seconds
node_degree: 0.016659975051879883 seconds
G_degree: 3.0994415283203125e-06 seconds
L 0.3128049373626709 seconds
n 5.0067901611328125e-06 seconds
I 0.0002460479736328125 seconds
axisSum 1.215691089630127 seconds
sumMax 0.00013208389282226562 seconds
diffusionParameter 5.0067901611328125e-06 seconds
ps 0.3286440372467041 seconds
'''