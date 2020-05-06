'''@author: minhpham'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from randomize import performance_run

def plot_performance(score, classify, result_fl, name, type='ROC', plotting= True):
    if type == 'ROC':
          fpr, tpr, _ = roc_curve(classify, score, pos_label=1)
          x_axis, y_axis, x_axis_name, y_axis_name = fpr, tpr, 'FPR', 'TPR'
    elif type == 'PRC':
          precision, recall, _ = precision_recall_curve(classify, score, post_label=1)
          x_axis, y_axis, x_axis_name, y_axis_name = recall, precision, 'Recall', 'Precision'  
    header = '%20s\t%30s'%(y_axis_name,x_axis_name)
    np.savetxt(result_fl+'/'+name, np.column_stack((y_axis,x_axis)), delimiter ='\t',  header = header, comments='')
    auc_= auc(x_axis, y_axis)
    if plotting == True:
          plt.figure()
          lw = 2
          plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label=type+' curve (area = %0.2f)' % auc_)
          plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.0])
          plt.xlabel(x_axis_name, fontsize='x-large')
          plt.ylabel(y_axis_name, fontsize='x-large')
          plt.legend(loc='lower right',fontsize='xx-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
          plt.tight_layout()
          plt.savefig('{}/{}'.format(result_fl, name+type))
          plt.close()
    return auc_

def plotAUCrand (roc_exp, roc_rands, z_text, result_fl, name, type = 'density'):
    if type == 'density':
          plt.close()
          sns.kdeplot( np.array(roc_rands) , color="navy", shade = True)
          plt.legend(loc = 'upper left')
          plt.xlim([0,1])
          plt.xlabel("Random AUCs", fontsize='x-large')
          plt.ylabel("Density", fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    elif type == 'hist':
          plt.hist(roc_rands, color = 'gray', bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
          plt.annotate('AUC = %0.3f\nz = {}'.format(z_text) %roc_exp, xy = (roc_exp, 0), xytext = (roc_exp,10),color = 'darkorange',fontsize = 'xx-large', arrowprops = dict(color = 'darkorange',width = 2, shrink=0.05),va='center',ha='center')
          plt.xlim([0.0, 1.0])
          plt.xlabel('Random AUCs', fontsize='x-large')
          plt.ylabel('Count', fontsize='x-large')
          plt.xticks(fontsize='large')
          plt.yticks(fontsize='large')
    plt.tight_layout()
    plt.savefig('{}/{}'.format(result_fl, name))
    plt.close()
    
def runrun(findex, tindex, t, f, t_, f_, name, show):
    auROC, auPRC, classify, score = performance_run(from_index, to_index)
    auc_ = plot_performance(score, classify, '{}{}'.format(show, name_))
    ### Computing z-scores when comparing against degree-matched randomization
    zt = (auc-np.mean(t))/np.std(t)
    zf = (auc-np.mean(f))/np.std(f)
    zt_txt = '%0.3f' %zt
    zf_txt = '%0.3f' %zf
    ### Computing z-scores when comparing against uniform randomization
    zt_ = (auc-np.mean(t_))/np.std(t_)
    zf_ = (auc-np.mean(f_))/np.std(f_)
    zt_txt_ = '%0.3f' %zt_
    zf_txt_ = '%0.3f' %zf_
    if show != '':
        showx = '_'.join(show.split('_')[:-1])
        plotAUCrand(auc, t, zt_txt, showx+'1_Zt_degree-matched_'+name)
        plotAUCrand(auc, f, zf_txt, showx+'2_Zf_degree-matched_'+ name)
    print(name, auc, zt_txt, zt_txt_, zf_txt, zf_txt_)
    return auc, zt_txt, zf_txt, zt_txt_, zf_txt_
