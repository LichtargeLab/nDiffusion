'''@author: minhpham'''
from utils import *
import networkx as nx

def run_method(from_dict, to_dict, group1_name, group2_name, net_var, method, spl = False):
    print (method)
    name = 'from {} to {}'.format(group1_name, group2_name)
    print (name)
    degree_nodes, other, graph_node_index, graph_node, ps, G = net_var
    if method == 'Diffusion':
        ### experimental results  
        results = performance_run(from_dict['index'], to_dict['index'], graph_node, ps)
        #### Randomizing nodes where diffusion starts
        AUROCs_from_degree, AUPRCs_from_degree, scoreTPs_from_degree = runRand(from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='degree', node_type='FROM', repeat = 30)
        #### Randomizing nodes which are true positive
        AUROCs_to_degree, AUPRCs_to_degree, scoreTPs_to_degree = runRand(to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, graph_node, ps, rand_type='degree', node_type='TO', diffuseMatrix=results['diffuseMatrix'], repeat = 30)
    elif method == 'Shortest Path Length':
        ### experimental results  
        results = performance_run_spl(from_dict['index'], to_dict['index'], graph_node, G, spl=spl)
        #### Randomizing nodes where diffusion starts
        AUROCs_from_degree, AUPRCs_from_degree, scoreTPs_from_degree, spl_from = runRand_spl(from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, graph_node, G, rand_type='degree', node_type='FROM', spl = results['spl'], repeat = 30)
        #### Randomizing nodes which are true positive
        AUROCs_to_degree, AUPRCs_to_degree, scoreTPs_to_degree, spl_to = runRand_spl(to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, graph_node, G, rand_type='degree', node_type='TO', spl=spl_from, repeat = 30)
    
    ### Computing z-scores when comparing AUROC and AUPRC against random
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    z_auc_from = '%0.2f' %((results['auROC']-np.mean(AUROCs_from_degree))/np.std(AUROCs_from_degree))
    z_auc_to = '%0.2f' %((results['auROC']-np.mean(AUROCs_to_degree))/np.std(AUROCs_to_degree))

    ### Visualization of AUROCs
    print ('Experimental AUROC: '+ method + ' '+ name)
    plot_performance(results['fpr'], results['tpr'], results['auROC'], './', name, type = 'ROC', display = True)
    print ('Randomize ' + group2_name + ' : ' + method + ' ' + name)
    plotAUCrand(results['auROC'], AUROCs_to_degree, z_auc_to, './', 'Randomize ' + group2_name + ' : ' + method + ' ' + name, display=True)
    print ('Randomize ' + group1_name + ' : ' + method + ' ' + name)
    plotAUCrand(results['auROC'], AUROCs_from_degree, z_auc_from, './', 'Randomize ' + group1_name + ' : ' + method + ' ' + name, display=True)
    return results, z_auc_from, z_auc_to

def performance_run_spl(from_index, to_index, graph_node, G, exclude = [], spl=False):
    results = {}
    if spl == False:
        spl = {}
    if exclude == []:
        exclude = from_index
    score, classify, scoreTP, gene_write = [], [], [], []
    for i in range(len(graph_node)):
        s = 0
        if i not in exclude:
            x = graph_node[i]
            gene_write.append(x)
            for e in from_index:
                y = graph_node[e]
                try:
                    s += spl[x][y]
                except:
                    try:
                        spl_xy = nx.shortest_path_length(G, source = x, target = y)
                        s += spl_xy
                        try:
                            spl[x][y] = spl_xy
                        except:
                            spl[x] = {}
                            spl[x][y] = spl_xy
                        try:
                            spl[y][x] = spl_xy
                        except:
                            spl[y] = {}
                            spl[y][x] = spl_xy
                    except:
                        pass
            score.append(-s)
            if i in to_index:
                classify.append(1)
                scoreTP.append(-s)
            else:
                classify.append(0)
    results['classify'], results['score'], results['scoreTP'], results['genes'] = classify, score, scoreTP, gene_write
    results['spl'] = spl
    results['fpr'], results['tpr'], thresholds = roc_curve(classify, score, pos_label=1)
    results['auROC']= auc(results['fpr'], results['tpr'])
    results['precision'], results['recall'], thresholds = precision_recall_curve(classify, score, pos_label=1)
    results['auPRC'] = auc(results['recall'], results['precision'])
    return results 

def runRand_spl(node_degree_count, node_index, degree_nodes, other, graph_node_index, graph_node, G, rand_type, node_type, spl=False, repeat=5):
    AUROCs, AUPRCs, scoreTPs = [], [], []
    if rand_type == 'uniform':
        getRand = getRand_uniform
        var2 = other
    elif rand_type == 'degree':
        getRand = getRand_degree
        var2 = degree_nodes
    for i in range(repeat):
        rand_node = getRand(node_degree_count, var2)
        rand_index = getIndex(rand_node, graph_node_index)
        if node_type == 'TO':
            results = performance_run_spl(node_index, rand_index, graph_node, G, spl=spl)
        elif node_type == 'FROM':
            results = performance_run_spl(rand_index, node_index, graph_node, G, spl=spl)
        AUROCs.append(results['auROC'])
        AUPRCs.append(results['auPRC'])
        scoreTPs += results['scoreTP']
    return AUROCs, AUPRCs, scoreTPs, results['spl']

