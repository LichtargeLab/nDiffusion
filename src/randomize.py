'''@author: minhpham'''
from process_Inputs import getIndex
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import random

### Computing AUROC and AUPRC for each run 
def performance_run (from_index, to_index, graph_node, ps):
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
    auROC= auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(classify, score, post_label=1)
    auPRC = auc(recall, precision)
    return auROC, auPRC, classify, score

### Performing degree-matched randomization
def getRand(pred_degree_count, degree_nodes, iteration=1):
    rand_node = []
    rand_degree = {}
    for i in pred_degree_count:
        rand_degree[i] = []
        count = pred_degree_count[i]*iteration
        lst = []
        modifier = 1
        cnt = 0
        if float(i) <= 100:
            increment = 1
        else:
            increment = 5
        while len(lst) < count and modifier <= float(i)/10 and cnt <= 500:
            degree_select = [n for n in degree_nodes.keys() if n <= i+modifier and n >= i-modifier]
            node_select = []
            for m in degree_select:
                node_select += degree_nodes[m]
            node_select = list(set(node_select))
            random.shuffle(node_select)
            try:
                lst += node_select[0:(count-len(lst))]
            except:
                pass
            modifier += increment
            cnt += 1
            overlap = set(rand_node).intersection(lst)
            for item in overlap:
                lst.remove(item)
            rand_node += lst
            rand_degree[i] += lst
    return rand_node, rand_degree

def runRandt(repeat, to_degree_count, graph_node, from_index):
    AUROCs, AUPRCs = [], []
    for i in range(repeat):
        rand_node, rand_degree = getRand(to_degree_count, 1)
        rand_index = getIndex(rand_node, graph_node)
        auROC, auPRC, classify, score = performance_run(from_index, rand_index)
        AUROCs.append(auROC)
        AUPRCs.append(auPRC)
    return AUROCs, AUPRCs

def runRandf(repeat, from_degree_count, graph_node, to_index):
    AUROCs, AUPRCs = [], []
    for i in range(repeat):
        rand_node, rand_degree = getRand(from_degree_count, 1)
        rand_index = getIndex(rand_node, graph_node)
        auROC, auPRC, classify, score = performance_run(rand_index, to_index)
        AUROCs.append(auROC)
        AUPRCs.append(auPRC)
    return AUROCs, AUPRCs

### Performing uniform randomization    
def getRand_uniform(pred_degree_count, other):
    number_rand = sum(pred_degree_count.values())
    #random.shuffle(other)
    #rand_node = other[:number_rand]
    rand_node = random.sample(other, number_rand)
    return rand_node

def runRandt_uniform(repeat, to_degree_count, graph_node, from_index):
    AUROCs, AUPRCs = [], []
    for i in range(repeat):
        rand_node = getRand_uniform(to_degree_count)
        rand_index = getIndex(rand_node, graph_node)
        auROC, auPRC, classify, score = performance_run(from_index, rand_index)
        AUROCs.append(auROC)
        AUPRCs.append(auPRC)
    return AUROCs, AUPRCs

def runRandf_uniform(repeat, from_degree_count, graph_node, to_index):
    AUROCs, AUPRCs = [], []
    for i in range(repeat):
        rand_node = getRand_uniform(from_degree_count)
        rand_index = getIndex(rand_node, graph_node)
        auROC, auPRC, classify, score = performance_run(rand_index, to_index)
        AUROCs.append(auROC)
        AUPRCs.append(auPRC)
    return AUROCs, AUPRCs