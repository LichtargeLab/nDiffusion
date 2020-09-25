'''
Created on Sep 5, 2017

@author: minhpham
'''

#!/usr/bin/python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal
import collections
import networkx as nx
import sys
from decimal import Decimal
from scipy.stats import mstats
import seaborn as sns
import pandas as pd
from IPython import embed
import random
import scipy
from scipy.stats import f_oneway

    
def mappingParser (fl):
    dict_ = {}
    id_gene = {}
    gene_id = {}
    readFl = open(fl).readlines()
    for line in readFl:
        if '#' not in line:
            line = line.strip('\n').split('\t')
            ID = line[0]
            name = line[1]
            id_gene[ID] = name
            gene_id[name] = ID
            synonym = line[2]
            if ';;' in synonym:
                synonym = synonym.split(';;')
            else:
                synonym = [synonym]
            synonym.append(name)
            dict_[ID] = synonym
    return dict_, id_gene, gene_id

def netParser (fl):
    lst = []
    readFl = open(fl).readlines()
    for line in readFl:
        if '#' not in line:
            line = line.strip('\n').split('\t')
            line1 = '{}\t{}\t{}'.format(line[0], line[1], line[2])
            lst.append(line1)
    return lst

def gene2node (lst):
    pred_node = []
    error = []
    for i in lst:
        try:
            pred_node.append(name_id[i])
        except:
            error.append(i)
    return pred_node, error

def group_shortest_path_len (source_nodes, target_nodes):
    src_target_dict = {}
    src_target = []
    for s in source_nodes:
        for t in target_nodes:
            if t != s:
                try:
                    spl = shortest_path_len[s][t]
                    src_target.append(spl)         
                    src_target_dict[s] = {}
                    src_target_dict[s][t] = spl
                except:
                    pass
    return src_target, src_target_dict

def stat (data1, data2):
    d_less,ks_less = mstats.ks_twosamp(data1,data2, alternative = 'less')
    d_greater,ks_greater = mstats.ks_twosamp(data1,data2, alternative = 'greater')
    return d_less,ks_less, d_greater,ks_greater

def stat_lst (data_to_plot):
    dless1, ks_less1, dgreater1, ks_greater1 = stat(data_to_plot[0], data_to_plot[1])
    dless2, ks_less2, dgreater2, ks_greater2 = stat(data_to_plot[0], data_to_plot[2])
    dless3, ks_less3, dgreater3, ks_greater3 = stat(data_to_plot[1], data_to_plot[2])
    return dless1, ks_less1, dgreater1, ks_greater1, dless2, ks_less2, dgreater2, ks_greater2, dless3, ks_less3, dgreater3, ks_greater3

def boxplot (data_to_plot, xlabel, yaxis, name, network, result_fl):       
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    for box in bp['boxes']:
        box.set( color='#7570b3', linewidth=2)
        box.set( facecolor = '#1b9e77' )
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(xlabel)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    kruskal1 = kruskal(data_to_plot[0], data_to_plot[1])[1]
    kruskal2 = kruskal(data_to_plot[0], data_to_plot[2])[1]
    kruskal3 = kruskal(data_to_plot[1], data_to_plot[2])[1]
    Hstat, pval = kruskal(data_to_plot[0], data_to_plot[1], data_to_plot[2])  

    p = pval
    p1 = kruskal1
    p2 = kruskal2
    p3 = kruskal3
    txt = 'Kruskal Wallis:\nOverall:{}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}'.format(p, xlabel[0], xlabel[1], p1, xlabel[0], xlabel[2],
                                                                                p2, xlabel[1], xlabel[2], p3)
    print (txt)
    
    ax.set_ylabel(yaxis)
    fig.savefig('./{}/{}_{}_boxplot'.format(result_fl, name, network), bbox_inches='tight')
    fig.clear()

def histogram (data_to_plot, xlabel, yaxis, name, network, result_fl):       
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    colors = ['red', 'tan', 'lime']
    weights = []
    for i in data_to_plot:
        data = np.array(i)
        weight = np.ones_like(data)/float(len(data))
        weights.append(weight)
        
    ax.hist(data_to_plot, histtype='bar', color=colors, label=xlabel, weights = weights)
    ax.legend(prop={'size': 10})
    dless1, ks_less1, dgreater1,ks_greater1, dless2,ks_less2, dgreater2,ks_greater2, dless3,ks_less3, dgreater3, ks_greater3 = stat_lst(data_to_plot)
    
    txt = 'KS Test:\n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {} \n{}-{}: Less - {} , {}; Greater - {} , {}'.format(xlabel[0], xlabel[1], dless1,ks_less1, dgreater1,ks_greater1, 
    xlabel[0], xlabel[2], dless2, ks_less2, dgreater2, ks_greater2, xlabel[1], xlabel[2], dless3, ks_less3, dgreater3, ks_greater3)
    
    print (txt)
    ax.set_ylabel('Frequency')
    ax.set_xlabel(yaxis)
    fig.savefig('./{}/{}_{}_histogram'.format(result_fl, name, network), bbox_inches='tight')
    fig.clear()

def group_closeness_central (dict_, lst):
    lst_return = []
    for i in lst:
        if i in dict_:
            lst_return.append(dict_[i])
    return lst_return


def seaborn_violin (data_to_plot, xlabel, yaxis, name, network, result_fl):
    dict_ = {}
    for i in range(len(xlabel)):
        dict_[xlabel[i]] = data_to_plot[i]
        df = pd.DataFrame(columns=['Group', yaxis])
        for m in dict_:
            tmpdf = pd.DataFrame(columns=['Group',yaxis])
            tmpdf[yaxis]=dict_[m]
            tmpdf['Group']=[m] * len(dict_[m])
            df = pd.concat([df,tmpdf])

        sns.set_style("whitegrid")
        sns.set(font_scale=2)
        ax = sns.violinplot(x="Group", y = yaxis, data=df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(bottom = 0)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("./{}/{}_{}_violin.png".format(result_fl, name, network))
        fig.clf()


def boxplot1 (data_to_plot, xlabel, yaxis, name, network, statistics):       
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_to_plot)
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    for box in bp['boxes']:
        box.set( color='#7570b3', linewidth=2)
        box.set( facecolor = '#1b9e77' )
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    ax.set_xticklabels(xlabel)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    if statistics == 'One-way ANOVA':
        kruskal1 = f_oneway(data_to_plot[0], data_to_plot[4])[1]
        kruskal2 = f_oneway(data_to_plot[1], data_to_plot[4])[1]
        kruskal3 = f_oneway(data_to_plot[2], data_to_plot[4])[1]
        kruskal4 = f_oneway(data_to_plot[3], data_to_plot[4])[1]
        kruskal5 = f_oneway(data_to_plot[0], data_to_plot[1])[1]
        kruskal6 = f_oneway(data_to_plot[0], data_to_plot[2])[1]
        kruskal7 = f_oneway(data_to_plot[1], data_to_plot[2])[1]
        pval = f_oneway(data_to_plot[0], data_to_plot[1], data_to_plot[2], data_to_plot[3], data_to_plot[4]) 
    
    elif statistics == 'Kruskal Wallis':
        kruskal1 = kruskal(data_to_plot[0], data_to_plot[4])[1]
        kruskal2 = kruskal(data_to_plot[1], data_to_plot[4])[1]
        kruskal3 = kruskal(data_to_plot[2], data_to_plot[4])[1]
        kruskal4 = kruskal(data_to_plot[3], data_to_plot[4])[1]
        kruskal5 = kruskal(data_to_plot[0], data_to_plot[1])[1]
        kruskal6 = kruskal(data_to_plot[0], data_to_plot[2])[1]
        kruskal7 = kruskal(data_to_plot[1], data_to_plot[2])[1]
        Hstat, pval = kruskal(data_to_plot[0], data_to_plot[1], data_to_plot[2], data_to_plot[3], data_to_plot[4])  
    
    p = pval
    p1 = kruskal1
    p2 = kruskal2
    p3 = kruskal3
    p4 = kruskal4
    p5 = kruskal5
    p6 = kruskal6
    p7 = kruskal7
    txt = '{}:\nOverall:{}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}\n{}-{}: {}'.format(statistics, p, xlabel[0], xlabel[4], p1, xlabel[1], xlabel[4], p2, xlabel[2], xlabel[4], p3, xlabel[3], xlabel[4], p4, xlabel[0], xlabel[1], p5, xlabel[0], xlabel[2], p6,  xlabel[1], xlabel[2], p7)
    print (txt)
    ax.set_ylabel(yaxis)
    fig.savefig('./{}/{}_{}_boxplot'.format(result_fl, name, network), bbox_inches='tight')
    fig.clear()

def stat_lst1 (data_to_plot):
    dless1, ks_less1, dgreater1, ks_greater1 = stat(data_to_plot[0], data_to_plot[4])
    dless2, ks_less2, dgreater2, ks_greater2 = stat(data_to_plot[1], data_to_plot[4])
    dless3, ks_less3, dgreater3, ks_greater3 = stat(data_to_plot[2], data_to_plot[4])
    dless4, ks_less4, dgreater4, ks_greater4 = stat(data_to_plot[3], data_to_plot[4])
    dless5, ks_less5, dgreater5, ks_greater5 = stat(data_to_plot[0], data_to_plot[1])
    dless6, ks_less6, dgreater6, ks_greater6 = stat(data_to_plot[0], data_to_plot[2])
    dless7, ks_less7, dgreater7, ks_greater7 = stat(data_to_plot[1], data_to_plot[2])
    return (dless1, ks_less1, dgreater1, ks_greater1, dless2, ks_less2, dgreater2, ks_greater2, dless3, ks_less3, dgreater3, ks_greater3,
            dless4, ks_less4, dgreater4, ks_greater4, dless5, ks_less5, dgreater5, ks_greater5, dless6, ks_less6, dgreater6, ks_greater6, 
            dless7, ks_less7, dgreater7, ks_greater7)

def histogram1 (data_to_plot, xlabel, yaxis, name, network):       
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    weights = []
    for i in data_to_plot:
        data = np.array(i)
        weight = np.ones_like(data)/float(len(data))
        weights.append(weight) 
    ax.hist(data_to_plot, histtype='bar', label=xlabel, weights = weights)
    ax.legend(prop={'size': 10})
    
    dless1, ks_less1, dgreater1, ks_greater1, dless2, ks_less2, dgreater2, ks_greater2, dless3, ks_less3, dgreater3, ks_greater3,dless4, ks_less4, dgreater4, ks_greater4, dless5, ks_less5, dgreater5, ks_greater5, dless6, ks_less6, dgreater6, ks_greater6, dless7, ks_less7, dgreater7, ks_greater7 = stat_lst1(data_to_plot) 
    txt = 'KS Test:\n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {} \n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {}\n{}-{}: Less - {} , {}; Greater - {} , {}'.format(xlabel[0], xlabel[4], dless1,ks_less1, dgreater1,ks_greater1, xlabel[1], xlabel[4], dless2, ks_less2, dgreater2, ks_greater2, xlabel[2], xlabel[4], dless3, ks_less3, dgreater3, ks_greater3,xlabel[3], xlabel[4], dless4, ks_less4, dgreater4, ks_greater4,xlabel[0], xlabel[1],dless5, ks_less5, dgreater5, ks_greater5, xlabel[0], xlabel[2], dless6, ks_less6, dgreater6, ks_greater6,xlabel[1], xlabel[2], dless6, ks_less6, dgreater6, ks_greater6)   
    print (txt)
    ax.set_ylabel('Frequency')
    ax.set_xlabel(yaxis)
    fig.savefig('./{}/{}_{}_histogram'.format(result_fl, name, network), bbox_inches='tight')
    fig.clear()

    
    
    
def getDegree (pred_node):
    pred_degree = []
    for i in pred_node:
        pred_degree.append(node_degree[i])
    pred_degree_count = dict(Counter(pred_degree))
    return pred_degree_count

def readfl (fl):
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

def getRand (pred_degree_count, iteration=1):
    for i in G_degree:
        random.shuffle(degree_nodes[i])    
    rand_node = []
    rand_degree = {}
#     print 'total', len(pred_degree_count)
    for i in pred_degree_count:
        rand_degree[i] = []
        count = pred_degree_count[i]*iteration
        if len(degree_nodes[i]) >= count:
            rand_node += degree_nodes[i][0:count]
            rand_degree[i] += degree_nodes[i][0:count]
        else:
            lst = degree_nodes[i]
            modifier = 1
            count = 0
            if float(i) <= 100:
                increment = 1
            else:
                increment = float(i)/(100*2)
            while len(lst) < count and modifier <= float(i)/10 and count <= 500:
                try:
                    if len(degree_nodes[i+modifier]) < (count - len(lst)):
                        lst += degree_nodes[i+ modifier]
                    elif len(degree_nodes[i+modifier]) >= (count - len(lst)):
                        lst += degree_nodes[i+ modifier][0:(count-len(lst))]
                except:
                    pass
                try:
                    if len(degree_nodes[i-modifier]) < (count - len(lst)):
                        lst += degree_nodes[i- modifier]
                    elif len(degree_nodes[i-modifier]) > (count - len(lst)):
                        lst += degree_nodes[i- modifier][0:(count-len(lst))]
                except:
                    pass
                    modifier += increment
                    count += 1
                    overlap = set(rand_node).intersection(lst)
                    for item in overlap:
                        lst.remove(item)
                    rand_node += lst
                    rand_degree[i] += lst
    return rand_node, rand_degree

def ROC(score, classify, name):
    fpr, tpr, thresholds = roc_curve(classify, score, pos_label=1)
    roc_auc= auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize='x-large')
    plt.ylabel('True Positive Rate', fontsize='x-large')
    plt.legend(loc='lower right',fontsize='xx-large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.tight_layout()
    plt.savefig('{}/{}'.format(result_fl, name))
    plt.close()
    return roc_auc


def ROC_run (from_index, index, from_name):
    label = np.zeros(len(graph_node))
    for i in from_index:
        label[i] = 1
    diffuseMatrix = diffuse(label, ps)
    score, classify = [], []
    for i in range(len(graph_node)):
        if i not in from_index:
            score.append(diffuseMatrix[i])
            if i in index:
                classify.append(1)
            else:
                classify.append(0)
    auc1 = ROC(score, classify, 'Diffusion ROC from {} {}'.format(from_name, net_name))
    return auc1

def ROC_run_rand (from_index, to_index):
    label = np.zeros(len(graph_node))
    for i in from_index:
        label[i] = 1
    diffuseMatrix = diffuse(label, ps)
    score, classify = [], []
    for i in range(len(graph_node)):
        if i not in from_index:
            score.append(diffuseMatrix[i])
            if i in to_index:
                classify.append(1)
            else:
                classify.append(0)
    fpr, tpr, thresholds = roc_curve(classify, score, pos_label=1)
    a= auc(fpr, tpr)
    return a

def runRandt(repeat, to_degree_count, graph_node, from_index):
    Rand_indexes, AUCs = [], []
    for i in range(repeat):
        rand_node, rand_degree = getRand(to_degree_count, 1)
        rand_index = getIndex(rand_node, graph_node)
        a = ROC_run_rand(from_index, rand_index)
        Rand_indexes.append(rand_index)
        AUCs.append(a)
    return AUCs

def runRandf(repeat, from_degree_count, graph_node, to_index):
    Rand_indexes, AUCs = [], []
    for i in range(repeat):
        rand_node, rand_degree = getRand(from_degree_count, 1)
        rand_index = getIndex(rand_node, graph_node)
        a = ROC_run_rand(rand_index, to_index)
        Rand_indexes.append(rand_index)
        AUCs.append(a)
    return AUCs

def runRandf1(repeat, from_degree_count, graph_node, to_index):
    Rand_indexes, AUCs = [], []
    rand_node1, rand_degree = getRand(from_degree_count, iteration=100)
    print (rand_node1)
    for i in range(len(rand_node1)):
        rand_node = rand_node1[i]
        rand_index = getIndex(rand_node, graph_node)
        a = ROC_run_rand(rand_index, to_index)
        Rand_indexes.append(rand_index)
        AUCs.append(a)
    return AUCs

def runrun(findex, tindex, t, f, name):
    auc = ROC_run(findex, tindex, name)
    zt = (auc-np.mean(t))/np.std(t)
    zf = (auc-np.mean(f))/np.std(f)
    if mstats.normaltest(t)[1] > 0.05:
        zt_txt = zt
    else:
        zt_txt = '{} (invalid)'.format(zt)
    if mstats.normaltest(f)[1] > 0.05:
        zf_txt = zf
    else:
        zf_txt = '{} (invalid)'.format(zf)
    print (name, auc, zt_txt, zf_txt)
    return auc, zt, zf

def runrun1(findex, tindex, name, graph_node, fdegree_count, tdegree_count):
    t = runRandt(100, tdegree_count, graph_node, findex)
    f = runRandf(100, fdegree_count, graph_node, tindex)
    auc = ROC_run(findex, tindex, name)
    zt = (auc-np.mean(t))/np.std(t)
    zf = (auc-np.mean(f))/np.std(f)
    if mstats.normaltest(t)[1] > 0.05:
        zt_txt = zt
    else:
        zt_txt = '{} (invalid)'.format(zt)
    if mstats.normaltest(f)[1] > 0.05:
        zf_txt = zf
    else:
        zf_txt = '{} (invalid)'.format(zf)
    print (name, auc, zt_txt, zf_txt)
    return auc, zt, zf

if __name__ == "__main__":  
   
    net_name = sys.argv[1]
    net_fl = sys.argv[2]
    pred_fl = sys.argv[3]
    result_fl = sys.argv[4]
    mappingFile = './data/Mapping_Ccgdd16Sx91Extended.tsv'

    mapping, id_name, name_id  = mappingParser(mappingFile)
    net_lst = netParser(net_fl)
    G = nx.parse_edgelist(net_lst, data=(('weight',float),))
    
    AD_pred_read = open('./data/{}'.format(pred_fl)).readlines()
    AD_pred = []
    for line in AD_pred_read:
        line = line.strip('\n').split('\t')
        AD_pred.append(line[0])
    AD_pred_node, error_pred = gene2node (AD_pred)
    print (len(AD_pred_node))
    print (len(error_pred)) 

    AD_GS_read = open('./data/all_AD.txt').readlines()
    AD_GS = []
    for line in AD_GS_read:
        line = line.strip('\n').split('\t')
        AD_GS.append(line[0])

    AD_GS_node, error_GS = gene2node (AD_GS)
    print (len(AD_GS_node)) 
    print (len(error_GS))

    GS_pred = set(AD_GS_node).intersection(AD_pred_node)
    print (len(GS_pred))

    graph_node = G.nodes()
    other = list(set(graph_node) - set(AD_GS_node) - set(AD_pred_node))
    print (len(other))       
        

    shortest_path_len = nx.shortest_path_length(G)

    gs_gs, gs_gs_dict = group_shortest_path_len(AD_GS_node, AD_GS_node)
    gs_other, gs_other_dict = group_shortest_path_len(AD_GS_node, other)
    other_other, other_other_dict = group_shortest_path_len(other, other)

    p_p, p_p_dict = group_shortest_path_len(AD_pred_node, AD_pred_node)
    p_gs, p_gs_dict = group_shortest_path_len(AD_pred_node, AD_GS_node)
    p_other, p_other_dict = group_shortest_path_len(AD_pred_node, other)

    pgs_pred, pgs_pred_dict = group_shortest_path_len(GS_pred, AD_pred_node)
    pgs_gs, pgs_gs_dict = group_shortest_path_len(GS_pred, AD_GS_node)
    pgs_other, pgs_other_dict = group_shortest_path_len(GS_pred, other)

    print ('Shortest path length')

    gs_random, gs_random_dict = group_shortest_path_len(AD_GS_node, graph_node)
    random_random, random_random_dict = group_shortest_path_len(graph_node, graph_node)

    gs_random1 = []
    for i in range(100):
        a = random.sample(gs_random, 200)
        gs_random1 += a

    random_random1 = []
    for i in range(100):
        a = random.sample(random_random, 200)
        random_random1 += a

    boxplot1([p_p, p_gs, gs_gs, gs_random1, random_random1], ['pred_pred','pred_GS','GS_GS', 'GS-Rand', 'Rand-Rand'], 
            'Shortest path length', 'Shortest path length for all combinations', net_name, 'One-way ANOVA')
    boxplot1([p_p, p_gs, gs_gs, gs_random1, random_random1], ['pred_pred','pred_GS','GS_GS', 'GS-Rand', 'Rand-Rand'], 
            'Shortest path length', 'Shortest path length for all combinations', net_name, 'Kruskal Wallis')

    histogram1([p_p, p_gs, gs_gs, gs_random1, random_random1], ['pred_pred','pred_GS','GS_GS', 'GS-Rand', 'Rand-Rand'], 
            'Shortest path length', 'Shortest path length for all combinations', net_name)        
    
    embed()


