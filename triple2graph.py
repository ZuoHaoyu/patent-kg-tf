import json
import networkx as nx
import pandas
import pandas as pd
import matplotlib.pyplot as plt
def file_reading(file):
    with open (file,'r',encoding='utf-8') as f:

        lines = f.readlines()
        for line in lines:
            dic = json.loads(line)
    return (dic)
if __name__ == '__main__':
    # dic = file_reading('patent_example_test_1_fi_tail_distance_adv_1.json')
    # print (dic)
    head_name =  ['head','relation','tail']
    data = {'head':[],'relation':[], 'tail':[]}
    df = pandas.DataFrame(data)
    print (df)
    with open ('patent_example_test_1_fi_tail_distance_adv_1.json', 'r', encoding='utf-8') as f:

        lines = f.readlines()
        for line in lines:
            dic = json.loads(line)
            triples = dic['tri']
            print(len(dic['tri']))
            for triple in triples:
                print (triple)
                # triple ['h']
                # triple ['t']
                # triple ['r']
                df_added = pd.DataFrame ({'head':[triple['h']],'relation':[triple['r']], 'tail':[triple ['t']]})
                df=df.append (df_added,ignore_index=True)


    G = nx.from_pandas_edgelist(df,"head","tail",edge_attr=True, create_using=nx.MultiDiGraph())
    plt.figure (figsize = (12,12))
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()

