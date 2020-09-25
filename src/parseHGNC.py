import pandas as pd 
from IPython import embed
from process_Inputs import *

#### Format HGNC files
fl = pd.read_csv('../data/networks/HGNC-090920.txt', sep = '\t')
fl_filter = fl.drop(['Approved name','Pubmed IDs', 'Vega IDs', 'Vega ID(supplied by Vega)','GtRNAdb ID (supplied by GtRNAdb)', 'LNCipedia ID (supplied by LNCipedia)'], axis = 1) 
fl_filter['NCBI Gene ID(supplied by NCBI)'] = fl_filter['NCBI Gene ID(supplied by NCBI)'].astype('Int64').astype('str')

for index, row in fl_filter.iterrows(): 
          omim = row['OMIM ID(supplied by OMIM)']
          try:
                    if ',' in omim: 
                              omim = omim.replace(' ', '').split(',') 
                              omims = [] 
                              for i in omim: 
                                        i_id = 'OMIM:'+i 
                                        omims.append(i_id) 
                              omims_txt = ','.join(str(x) for x in omims) 
                    else: 
                              omims_txt = 'OMIM:'+omim 
                    fl_filter['OMIM ID(supplied by OMIM)'].loc[index]=omims_txt 
          except:
                    pass

for index, row in fl_filter.iterrows(): 
          ncbi = row['NCBI Gene ID(supplied by NCBI)']
          if ncbi != 'nan':
                    ncbi_txt = 'NCBI:'+ncbi
                    fl_filter['NCBI Gene ID(supplied by NCBI)'].loc[index]=ncbi_txt 

# fl_filter.to_csv('../data/networks/HGNC_IDs.txt', index=False, sep = '\t')  

### Mapping files
network_fl = '../data/networks/STRINGv11.txt'
G, graph_node, adjMatrix, node_degree, G_degree = getGraph(network_fl)
hgnc = fl_filter['Approved symbol'].to_list()
hgnc_al= []
for index, row in fl_filter.iterrows(): 
          approved = row['Approved symbol'] 
          hgnc_al.append(approved) 
          try: 
                    previous = row['Previous symbols'] 
                    previous = previous.replace(' ', '').split(',') 
                    hgnc_al+= previous 
          except: 
                    pass 
          try: 
                    alias = row['Alias symbols'] 
                    alias = alias.replace(' ','').split(',') 
                    hgnc_al += alias 
          except: 
                    pass 
          
graphnode_tomap = ((set(graph_node).difference(hgnc)).intersection(hgnc_al))
gene_map = {}                                                                                                                                       
for index, row in fl_filter.iterrows(): 
          approved = row['Approved symbol'] 
          other = [] 
          try: 
                    previous = row['Previous symbols'] 
                    previous = previous.replace(' ', '').split(',') 
                    other+= previous 
          except: 
                    pass 
          key = set(other).intersection(graphnode_tomap) 
          if len(key)==1: 
                    name = list(key)[0] 
                    gene_map[name] = other 
                    gene_map[name].append(approved) 
          elif len(key) > 1: 
                    for i in key: 
                              gene_map[i] = other 
                              gene_map[i].append(approved)

notmap = set(graphnode_tomap).difference(gene_map)
for index, row in fl_filter.iterrows(): 
          approved = row['Approved symbol'] 
          other = [] 
          try: 
                    previous = row['Previous symbols'] 
                    previous = previous.replace(' ', '').split(',') 
                    other+= previous 
          except: 
                    pass 
          try: 
                    alias = row['Alias symbols'] 
                    alias = alias.replace(' ','').split(',') 
                    other += alias 
          except: 
                    pass 
          key = set(other).intersection(notmap) 
          if len(key)==1: 
                    name = list(key)[0] 
                    gene_map[name] = other 
                    gene_map[name].append(approved) 
          elif len(key) > 1: 
                    for i in key: 
                              gene_map[i] = other 
                              gene_map[i].append(approved)

for index, row in fl_filter.iterrows(): 
          approved = row['Approved symbol'] 
          if approved in graph_node:
                    other = [] 
                    try: 
                              previous = row['Previous symbols'] 
                              previous = previous.replace(' ', '').split(',') 
                              other+= previous 
                    except: 
                              pass 
                    try: 
                              alias = row['Alias symbols'] 
                              alias = alias.replace(' ','').split(',') 
                              other += alias 
                    except: 
                              pass 
                    gene_map[approved] = other

out = open('../data/networks/gene_symbols.txt', 'w')
out.write('Gene\tAlias\n')
for i in gene_map:
          if gene_map[i] != []:
                    try:
                              alias_lst = list(set(gene_map[i].remove(i)))
                    except:
                              alias_lst = list(set(gene_map[i]))
                    alias_txt = ','.join(str(x) for x in alias_lst)
                    out.write('{}\t{}\n'.format(i, alias_txt))
out.close()
          



