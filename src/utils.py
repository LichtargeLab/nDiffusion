'''@author: minhpham'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from randomize import *
import scipy.stats as stats
import seaborn as sns

### Plotting AUROC and AUPRC
def plot_performance(x_axis, y_axis, auc_, result_fl, name, type='ROC', plotting= True):
    if type == 'ROC':
          x_axis_name, y_axis_name = 'FPR', 'TPR'
    elif type == 'PRC':
          x_axis_name, y_axis_name = 'Recall', 'Precision'  
    header = '%20s\t%30s'%(y_axis_name,x_axis_name)
    np.savetxt(result_fl+'raw_data/'+name+type, np.column_stack((y_axis,x_axis)), delimiter ='\t',  header = header, comments='')
    if plotting == True:
          plt.figure()
          lw = 2
          plt.plot(x_axis, y_axis, color='darkorange', lw=lw, label='AU'+type+' = %0.2f' % auc_)
          plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.0])
          plt.xlabel(x_axis_name, fontsize='x-large')
          plt.ylabel(y_axis_name, fontsize='x-large')
          plt.legend(loc='lower right',fontsize='xx-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
          plt.tight_layout()
          plt.savefig('{}'.format(result_fl+'figures/'+name+'_'+type))
          plt.close()

### Plotting distribution of random AUCs
def plotAUCrand (roc_exp, roc_rands, z_text, name, type = 'density'):
    if type == 'density':
          sns.kdeplot(np.array(roc_rands) , color="gray", shade = True)
          plt.legend(loc = 'upper left')
          plt.annotate('AUC = %0.2f\nz = {}'.format(z_text) %roc_exp, xy = (roc_exp, 0), xytext = (roc_exp,10),color = 'orangered',fontsize = 'xx-large', arrowprops = dict(color = 'orangered',width = 2, shrink=0.05),va='center',ha='center')          
          plt.xlim([0,1])
          plt.xlabel("Random AUCs", fontsize='x-large')
          plt.ylabel("Density", fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    elif type == 'hist':
          plt.hist(roc_rands, color = 'gray', bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
          plt.annotate('AUC = %0.2f\nz = {}'.format(z_text) %roc_exp, xy = (roc_exp, 0), xytext = (roc_exp,10),color = 'orangered',fontsize = 'xx-large', arrowprops = dict(color = 'orangered',width = 2, shrink=0.05),va='center',ha='center')
          plt.xlim([0.0, 1.0])
          plt.xlabel('Random AUCs', fontsize='x-large')
          plt.ylabel('Count', fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

### Plotting distribution of experimental and random diffusion values
def plotDist (exp_dist, randFRd, randTOd, randFRu, randTOu, name, from_gp_name, to_gp_name):
    sns.kdeplot(np.log10(exp_dist) , color="red", label="Experiment", shade = True)
    sns.kdeplot(np.log10(randFRd) , color="darkgreen", label="Randomize "+from_gp_name+" (degree-matched)", shade= True)
    sns.kdeplot(np.log10(randTOd) , color="darkblue", label="Randomize "+to_gp_name+" (degree-matched)", shade = True)
    sns.kdeplot(np.log10(randFRu) , color="lightgreen", label="Randomize "+from_gp_name+" (uniform)", shade= True)
    sns.kdeplot(np.log10(randTOu) , color="lightskyblue", label="Randomize "+to_gp_name+" (uniform)", shade = True)
    plt.legend(loc = 'upper left')
    plt.xlabel("log10 (diffusion value)")
    plt.ylabel("Density")
    plt.savefig(name)
    plt.close()

### Computing z-scores of experimental AUC against random AUCs
def z_scores(exp, randf_degree, randt_degree, randf_uniform, randt_uniform):
    zf_degree = '%0.2f' %((exp-np.mean(randf_degree))/np.std(randf_degree))
    zt_degree = '%0.2f' %((exp-np.mean(randt_degree))/np.std(randt_degree))
    zf_uniform = '%0.2f' %((exp-np.mean(randf_uniform))/np.std(randf_uniform))
    zt_uniform = '%0.2f' %((exp-np.mean(randt_uniform))/np.std(randt_uniform))
    return zf_degree, zt_degree, zf_uniform, zt_uniform

### Performing KS test to compare distributions of diffusion values
def distStats(exp, randf_degree, randt_degree, randf_uniform, randt_uniform):
    pf_degree ='{:.2e}'.format(stats.ks_2samp(exp, randf_degree)[1]) 
    pt_degree ='{:.2e}'.format(stats.ks_2samp(exp, randt_degree)[1]) 
    pf_uniform ='{:.2e}'.format(stats.ks_2samp(exp, randf_uniform)[1]) 
    pt_uniform ='{:.2e}'.format(stats.ks_2samp(exp, randt_uniform)[1]) 
    return pf_degree, pt_degree, pf_uniform, pt_uniform

### For a given diffusion experiment: computing experimental and random values
##  from_dict is for the group where diffusion signals start and to_dict is for the true positive group 
def runrun(from_dict, to_dict, result_fl, group1_name, group2_name, show, degree_nodes, other, graph_node_index, graph_node, ps, exclude = []):
    name = 'from {} to {}'.format(group1_name, group2_name)
    ### experimental results  
    results = performance_run(from_dict['index'], to_dict['index'], graph_node, ps, exclude = exclude)
    plot_performance(results['fpr'], results['tpr'], results['auROC'], result_fl, '{}{}'.format(name, show), type = 'ROC')
    plot_performance(results['recall'], results['precision'],results['auPRC'], result_fl, '{}{}'.format(name, show), type = 'PRC')
    
    ### Degree-matched randomization
    #### Randomizing nodes where diffusion starts
    AUROCs_from_degree, AUPRCs_from_degree, scoreTPs_from_degree = runRand(from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, ps, rand_type='degree', node_type='FROM')
    #### Randomizing nodes which are true positive
    AUROCs_to_degree, AUPRCs_to_degree, scoreTPs_to_degree = runRand(to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, ps, rand_type='degree', node_type='TO', diffuseMatrix=results['diffuseMatrix'])
    
    ### Uniform randomization
    #### Randomizing nodes where diffusion starts
    AUROCs_from_uniform, AUPRCs_from_uniform, scoreTPs_from_uniform = runRand(from_dict['degree'], to_dict['index'], degree_nodes, other, graph_node_index, ps, rand_type='uniform', node_type='FROM')
    #### Randomizing nodes which are true positive
    AUROCs_to_uniform, AUPRCs_to_uniform, scoreTPs_to_uniform = runRand(to_dict['degree'], from_dict['index'], degree_nodes, other, graph_node_index, ps, rand_type='uniform', node_type='TO', diffuseMatrix=results['diffuseMatrix'])
    
    ### Computing z-scores when comparing AUROC and AUPRC against random
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    z_auc = z_scores(results['auROC'], AUROCs_from_degree, AUROCs_to_degree, AUROCs_from_uniform, AUROCs_to_uniform)
    z_prc = z_scores(results['auPRC'], AUPRCs_from_degree, AUPRCs_to_degree, AUPRCs_from_uniform, AUPRCs_to_uniform)

    ### Computing KS test p-values when comparing distribution of diffusion values against random
    #### z-scores: from_degree, to_degree, from_uniform, to_uniform
    pval = distStats(results['scoreTP'], scoreTPs_from_degree, scoreTPs_to_degree, scoreTPs_from_uniform, scoreTPs_to_uniform)
    
    if show != '':
        showx = '_'.join(show.split('_')[:-1])
        plotAUCrand(results['auROC'], AUROCs_to_degree, z_auc[1], result_fl+'figures/'+name + showx+'1_Zt_degree-matched_')
        plotAUCrand(results['auROC'], AUROCs_from_degree, z_auc[0], result_fl+'figures/'+name + showx+'2_Zf_degree-matched_')
        plotDist (results['scoreTP'], scoreTPs_from_degree, scoreTPs_to_degree, scoreTPs_from_uniform, scoreTPs_to_uniform, result_fl+'figures/'+name + showx+'3_DiffValDist_', group1_name, group2_name)
    return '%0.2f' %results['auROC'], z_auc, '%0.2f' %results['auPRC'], z_prc, pval
