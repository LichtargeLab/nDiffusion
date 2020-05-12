'''@author: minhpham'''
from process_Inputs import *
from utils import *
from writeSummary import *
from IPython import embed
import sys, os

### CHANGE HERE: Assigning network, gene inputs, and result directory ###
network_fl = '../data/networks/STRINGv10.txt'
geneList1_fl = '../data/genes/gs_genes.tsv'
geneList2_fl = '../data/genes/Replicate_Final_SigGenes_list.tsv'
result_fl = '../results/'
group1_name = geneList1_fl.split('/')[-1].split('.')[0]
group2_name = geneList2_fl.split('/')[-1].split('.')[0]

if __name__ == "__main__":
    ### Directory of the result folder
    result_fl_figure = result_fl + 'figures/'
    result_fl_raw = result_fl + 'raw_data/'
    if not os.path.exists(result_fl):
        os.makedirs(result_fl)
    if not os.path.exists(result_fl_figure):
        os.makedirs(result_fl_figure)
    if not os.path.exists(result_fl_raw):
        os.makedirs(result_fl_raw)        
    print('Running ...')

    ### Getting network and diffusion parameters
    G, graph_node, adjMatrix, node_degree, G_degree = getGraph(network_fl)
    ps = getDiffusionParam(adjMatrix)
    graph_node_index = getIndexdict(graph_node)
    GP1_only_dict, GP2_only_dict, overlap_dict, other_dict = parseGeneInput(geneList1_fl, geneList2_fl, graph_node, graph_node_index, node_degree)
    degree_nodes = getDegreeNode(G_degree, node_degree, other_dict['node'])

    # Combine exclusive genes and overlapped genes in each group, if there is an overlap
    if overlap_dict['node'] != []:
        GP1_all_dict = combineGroup(GP1_only_dict, overlap_dict)
        GP2_all_dict = combineGroup(GP2_only_dict, overlap_dict)
        Exclusives_dict = combineGroup(GP1_only_dict, GP2_only_dict)
   
    ### Diffusion experiments
    def getResults(gp1, gp2, result_fl, gp1_name, gp2_name, show = '', exclude=[]):
        auroc, z_auc, auprc, z_prc, pval = runrun(gp1, gp2, result_fl, gp1_name, gp2_name, show, degree_nodes, other_dict['node'], graph_node_index, graph_node, ps, exclude=exclude)
        return auroc, z_auc, auprc, z_prc, pval
    
    #### auroc, z-scores for auc, auprc, z-scores for auprc, KS pvals
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform

    if overlap_dict['node'] != []:
        # From group 1 exclusive to group 2 all:
        R_gp1o_gp2 = getResults(GP1_only_dict, GP2_all_dict, result_fl, group1_name+'Excl', group2_name, show = '__SHOW_1_')
        # From group 2 exclusive to group 1 all:
        R_gp2o_gp1 = getResults(GP2_only_dict, GP1_all_dict, result_fl, group2_name+'Excl', group1_name, show = '__SHOW_2_')     
        # From group 1 exclusive to group 2 exclusive:
        R_gp1o_gp2o = getResults(GP1_only_dict, GP2_only_dict, result_fl, group1_name+'Excl', group2_name+'Excl')
        # From group 2 exclusive to group 1 exclusive:
        R_gp2o_gp1o = getResults(GP2_only_dict, GP1_only_dict, result_fl, group2_name+'Excl', group1_name+'Excl')
        # From group 1 exclusive to the overlap
        R_gp1o_overlap = getResults(GP1_only_dict, overlap_dict, result_fl, group1_name+'Excl', 'Overlap')
        # From group 2 exclusive to the overlap
        R_gp2o_overlap = getResults(GP2_only_dict, overlap_dict, result_fl, group2_name+'Excl', 'Overlap')
        # From overlap to (group 1 exclusive and group 2 exlusive)
        R_overlap_exclusives = getResults(overlap_dict, Exclusives_dict, result_fl,'Overlap', 'Exclus')
        '''
        #### NOT Recommended: When including overlap in both groups. We will for sure recover the true positive in the overlap, hence inflating the performances ####
        # From group 1 all to group 2 all
        R_gp1_gp2 = getResults(GP1_all_dict, GP2_all_dict, result_fl, group1_name, group2_name, exclude = GP1_only_dict['index'])
        # From group 2 all to group 1 all
        R_gp2_gp1 = getResults(GP2_all_dict, GP1_all_dict, result_fl, group2_name, group1_name, exclude = GP2_only_dict['index'])
        '''
        ### Write output
        writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp1o_gp2, R_gp2o_gp1, R_gp1o_gp2o, R_gp2o_gp1o, R_gp1o_overlap, R_gp2o_overlap, R_overlap_exclusives)
    else: #when there is no overlap between two groups
        # From group 1 to group 2:
        R_gp1o_gp2o = getResults(GP1_only_dict, GP2_only_dict, result_fl, group1_name, group2_name, show = '__SHOW_1_')
        # From group 2 to group 1:
        R_gp2o_gp1o = getResults(GP2_only_dict, GP1_only_dict, result_fl, group2_name, group1_name, show = '__SHOW_2_')
        
        ### Write output
        writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp1o_gp2o=R_gp1o_gp2o, R_gp2o_gp1o=R_gp2o_gp1o)

    embed()
