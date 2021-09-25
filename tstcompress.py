import numpy as np

tokenid2word_mapping=[0, 0, 0, 1, 2, 2, 2, 3, 3, 3]
attention  = a = np.arange(100).reshape(10,10)
def matrix_compress(attention, tokenid2word_mapping):
    tokenid_set = list(set(tokenid2word_mapping))
    operate_list=[]
    for tokenid in tokenid_set:
        operate_list.append( [i for i, x in enumerate(tokenid2word_mapping) if x == tokenid])
    b = np.vstack([np.mean(attention[operate_list[n], :], 0) for n in range(len(tokenid_set))])
    c = np.vstack([np.sum(b[:, operate_list[n]], 1) for n in range(len(tokenid_set))]).T
    return c
c=matrix_compress(attention,tokenid2word_mapping)
print(c)