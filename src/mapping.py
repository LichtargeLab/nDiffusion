from IPython import embed
from process_Inputs import *

def mapping():
          network_fl = '../data/networks/STRINGv11.txt'
          G, graph_node, adjMatrix, node_degree, G_degree = getGraph(network_fl)
          mapping_dict = {}
          mapping_fl = open('../data/networks/gene_symbols.txt').readlines()
          for line in mapping_fl:
                    if 'Alias' not in line:
                              line = line.strip('\n').split('\t')
                              alias = line[1].split(',')
                              for i in alias:
                                        if i not in graph_node:
                                                  try:
                                                            mapping_dict[i].append(line[0])
                                                  except:
                                                            mapping_dict[i]=[line[0]]
          return mapping_dict

