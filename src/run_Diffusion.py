'''@author: minhpham'''
from process_Inputs import *

if __name__ == "__main__":
    ### Assigning network and gene inputs
    network_fl = '../data/networks/STRINGv10.txt'
    geneList1_fl = '../data/genes/List1.txt'
    geneList2_fl = '../data/genes/List2.txt'
    group1_name = geneList1_fl.split('/')[-1]
    group2_name = geneList2_fl.split('/')[-1]

    ### Directory of the result folder
    result_fl = '../results/'
    if not os.path.exists(result_fl):
        os.makedirs(result_fl)
    print('Running ...')

    ### Getting network and diffusion parameters
    G, graph_node, adjMatrix, node_degree, G_degree = getGraph()
    ps = getDiffusionParam()
    GP1_only_dict, GP2_only_dict, overlap_dict, other_dict = parseGeneInput(geneList1_fl, geneList2_fl, graph_node)
    degree_nodes = getDegreeNode(G_degree, node_degree, other_dict['node'])

    ### randomization
    
    ### plotting: ROC and PRC
    ### printing out results