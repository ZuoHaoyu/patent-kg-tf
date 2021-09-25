import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
dic = {"appln_id": "508075381", "relationships": [{"h": "invention", "r": "relate to", "t": "automatic processing equipment", "c": 0.0653881374746561}, {"h": "invention", "r": "relate to", "t": "field", "c": 0.0643290039151907}, {"h": "automatic processing equipment", "r": "shift", "t": "forks", "c": 0.06725621409714222}, {"h": "turnover device", "r": "arrange", "t": "betweenfirst horizontal moving device", "c": 0.1125033088028431}, {"h": "turnover device", "r": "arrange", "t": "second horizontal moving device", "c": 0.10039059817790985}, {"h": "first polishing device", "r": "arrange beside", "t": "first horizontal moving device", "c": 0.07200439274311066}, {"h": "oil dripping device", "r": "arrange beside", "t": "second horizontal moving device", "c": 0.07251366972923279}, {"h": "lower surfaces", "r": "ground", "t": "second polishing device", "c": 0.08165158331394196}, {"h": "lubricating fluid", "r": "coat through", "t": "oil-drippingdevice", "c": 0.0768573172390461}]}

def drawing(dic):
    head = []
    tail = []
    rel = []
    for relationships in dic['relationships']:
        head.append (relationships['h'])
        tail.append(relationships['t'])
        rel.append(relationships['r'])
    kg_df = pd.DataFrame({'source': head, 'target': tail, 'edge': rel})
    G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                edge_attr=True)
    plt.figure(figsize=(12, 12))
    pos = nx.planar_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', pos=pos)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G,pos= pos)
    plt.show()
drawing(dic)