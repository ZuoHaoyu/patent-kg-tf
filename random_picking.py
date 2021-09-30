import json
import random
data='patent_data/patent_dic.json'
with open (data,'r',encoding='utf-8') as f:
    dic=json.load(f)
a = dic.keys()
sample=dict()
b=random.sample(a,200)
for i in b:
    sample[i]=dic[i]

with open ('patent_data/sample.txt','w',encoding='utf-8',newline='') as f1,\
    open ('patent_data/sample_json.json','w',encoding='utf-8') as f2:
    json.dump(sample,f2)
    for i in  b:
        f1.write(sample[i]["appln_abstract"]+'\n')
