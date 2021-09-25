import numpy as np
import torch
import re
from constant import invalid_relations_set
alphabet = re.compile(r'^[a-zA-Z]+$')

from copy import copy
from collections import defaultdict
from spacy.matcher import PhraseMatcher

def build_graph(matrix):
    graph = defaultdict(list) 

    for idx in range(0, len(matrix)):
        for col in range(idx+1, len(matrix)):
            graph[idx].append((col, matrix[idx][col] ))
    return graph
###candidate_facts_filter###

def check_single_relation_validity(relations):
    if relations.lower() in invalid_relations_set or relations.isnumeric():
        return False
    else:
        return True

def candidate_facts_relation_filter(candidate_facts,id2token):
    filtered_candidate_facts=[]
    for candidate_fact in candidate_facts:
        head=candidate_fact[0][0]
        tail=candidate_fact[0][2]
        relations=candidate_fact[0][1]
#         print ('%s %s %s'%(id2token[head],id2token[tail],id2token[relations]))
#         print ( len(id2token[relations]) > 0 )
#         print (id2token[relations])
#         print (check_single_relation_validity(id2token[relations]) )
#         print (id2token[head].lower() not in invalid_relations_set )
#         print (id2token[tail].lower() not in invalid_relations_set)
        if len(id2token[relations]) > 0 and check_single_relation_validity(id2token[relations]) and id2token[head].lower() not in invalid_relations_set and id2token[tail].lower() not in invalid_relations_set:
            filtered_candidate_facts.append(candidate_fact)
#     print ('filterd_c_f{} '.format(filtered_candidate_facts))
    if len(filtered_candidate_facts)>0:
        filtered_candidate_facts = sorted(filtered_candidate_facts, key=lambda x: x[1], reverse=True)

        filtered_candidate_facts = filtered_candidate_facts[0]
        return (filtered_candidate_facts)
    if len(filtered_candidate_facts)==0:
        return []

## candidate_facts_filter### end




def BFS(s, end, graph, id2token, max_size=-1, black_list_relation=[]):
#     print ('s %s, end %s ' % (str(s),str(end)))
    visited = [False] * (max(graph.keys())+100) 
  
    # Create a queue for BFS 
    queue = [] 

    # Mark the source node as  
    # visited and enqueue it 
    queue.append((s, [(s, 0)]))
    
    found_paths = []

    visited[s] = True
    
    while queue: 

        s, path = queue.pop(0)

        # Get all adjacent vertices of the 
        # dequeued vertex s. If a adjacent 
        # has not been visited, then mark it 
        # visited and enqueue it 
        for i, conf in graph[s]:
            if i == end:
                found_paths.append(path+[(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path)+[(i, conf)]))
                visited[i] = True
    
    candidate_facts = []
    for path_pairs in found_paths:
        if len(path_pairs) < 3:
            continue
        path = []
        cum_conf = 0
        for (node, conf) in path_pairs:
            path.append(node)
            cum_conf += conf
 ###### can add the relation_adv_filter to  here in the black list relation
        if path[1] in black_list_relation:
            continue

        candidate_facts.append((path, cum_conf))
    candidate_facts= candidate_facts_relation_filter(candidate_facts,id2token)
#     candidate_facts = sorted(candidate_facts, key=lambda x: x[1], reverse=True)

    return candidate_facts



def is_word(token):
    if len(token) == 1 and alphabet.match(token) == None:
        return False
    return True

def create_mapping(sentence, return_pt=False, nlp = None, tokenizer=None):
    '''Create a mapping
        nlp: spacy model
        tokenizer: huggingface tokenizer
    '''
    matcher1 = PhraseMatcher(nlp.vocab,attr='POS')

    terms = ['relate to','positioned through','engageable with','so as to','makes it possible to','to receive']

    # 'in regard to' ,'in response to', 'make it possible to'

    # 注意：只有使用 nlp.make_doc 才能加速
    patterns=[nlp(term) for term in terms]
    matcher1.add('VerbPhrase',patterns)
    
    
    doc = nlp(sentence)
    
    
    
    
        ###test for verb phrases start###
    matches = matcher1(doc)
    verb_phrase_start_chunk = []
    verb_phrase_end_chunk = []
    verb_phrase_chunk=[]
    # print (matches)
    for match_id, start, end in matches:
        span = doc[start:end]
        verb_phrase_start_chunk.append(start)
        verb_phrase_end_chunk.append(end)
        verb_phrase_chunk.append(span.text)
#             print("start: %s, end: %s text :%s" % (start,end,span.text))

    ### test for verb phrase ends####



#         for sent in nlp(sentence).sents:
#             print ("sent:{}".format(sent))
#             print ("sent start :%s sent end :%s"% (sent.start, sent.end))

    noun_start_chunk = []
    noun_end_chunk = []
    noun_chunks = []

    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)
        noun_start_chunk.append(chunk.start)
        noun_end_chunk.append(chunk.end)

# sometimes verb_phrase_start_chunk is overlapping with noun_start_chunk , so pop the overlapping in the verb_phrase_chunk

    for i,start in enumerate(verb_phrase_start_chunk):
        if start in noun_start_chunk:
            verb_phrase_start_chunk.pop(i)
            verb_phrase_end_chunk.pop(i)
            verb_phrase_chunk.pop(i)


    for i,end in enumerate(verb_phrase_end_chunk):
        if end in noun_end_chunk:
            verb_phrase_start_chunk.pop(i)
            verb_phrase_end_chunk.pop(i)
            verb_phrase_chunk.pop(i)


#         start_chunk= sorted(start_chunk + verb_phrase_start_chunk)
#         end_chunk = sorted (end_chunk +verb_phrase_end_chunk)
    chunk_id = 0
    token2id ={ }
    mode = 1
    sentence_mapping = []

    #### modification starts below,
    ### token2id [sentence_mapping[-1]]= len(token2id) --->token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
    ### because token2id is a dictionary, when you len it, it will automatically remove the duplicated keys, this will
    ### cause problems when the words occurs more than once


    dic = dict (zip (noun_start_chunk+verb_phrase_start_chunk,noun_chunks+verb_phrase_chunk))


    dic = {k: v for k, v in sorted(dic.items(), key=lambda dic:dic[0],reverse= False)}


    start_chunk= list(dic.keys())
    chunks = list(dic.values())
    end_chunk = sorted(noun_end_chunk+ verb_phrase_end_chunk)
    # print ('start_chunk: %s, end_chunk: %s'%(str(start_chunk),str(end_chunk)))
    # print (dic)
    for idx,token in enumerate (doc):
        if idx in start_chunk:


            mode= 1
            sentence_mapping.append (chunks[chunk_id])
            token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
#                 token2id [sentence_mapping[-1]]= len(token2id)
            chunk_id +=1 
            continue
        if idx in end_chunk:
            mode = 0


        if mode == 0:
            sentence_mapping.append (token.text)
            token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
#                 token2id [sentence_mapping[-1]]= len(token2id)

    ### modication ends here 
    
    
    
    
####testing cross starts below  #####
#     tokens = list(doc)

#     chunk2id = {}

#     start_chunk = []
#     end_chunk = []
#     noun_chunks = []
#     for chunk in doc.noun_chunks:
#         noun_chunks.append(chunk.text)
#         start_chunk.append(chunk.start)
#         end_chunk.append(chunk.end)

#     sentence_mapping = []
#     token2id = {}
#     mode = 0 # 1 in chunk, 0 not in chunk
#     chunk_id = 0
#     for idx, token in enumerate(doc):
#         if idx in start_chunk:
#             mode = 1
#             sentence_mapping.append(noun_chunks[chunk_id])
#         #### modification starts below,
#         ### token2id [sentence_mapping[-1]]= len(token2id) --->token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
#         ### because token2id is a dictionary, when you len it, it will automatically remove the duplicated keys, this will
#         ### cause problems when the words occurs more than once.   
#             token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
# #            token2id [sentence_mapping[-1]]= len(token2id)
#             chunk_id += 1
#         elif idx in end_chunk:
#             mode = 0

#         if mode == 0:
#             sentence_mapping.append(token.text)
#             token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
# #            token2id [sentence_mapping[-1]]= len(token2id)
#          ### modication ends here 

####testing corss ends above ######    
    
    
    token_ids = []
    tokenid2word_mapping = []
    ###modification starts below, adding id2token
    id2token = {key:value for key,value in enumerate(sentence_mapping)}
    ###modification ends here
    for token in sentence_mapping:
        subtoken_ids = tokenizer(str(token), add_special_tokens=False)['input_ids']
        tokenid2word_mapping += [ token2id[token] ]*len(subtoken_ids)
        token_ids += subtoken_ids

    tokenizer_name = str(tokenizer.__str__)
    if 'GPT2' in tokenizer_name:
        outputs = {
            'input_ids': token_ids,
            'attention_mask': [1]*(len(token_ids)),
        }

    else:
        outputs = {
            'input_ids': [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id],
            'attention_mask': [1]*(len(token_ids)+2),
            'token_type_ids': [0]*(len(token_ids)+2)
        }

    if return_pt:
        for key, value in outputs.items():
            outputs[key] = torch.from_numpy(np.array(value)).long().unsqueeze(0)
    ###modification starts below, adding id2token
    return outputs, tokenid2word_mapping, token2id, noun_chunks,id2token,noun_start_chunk

def compress_attention(attention, tokenid2word_mapping, operator=np.mean):

    new_index = []
    
    prev = -1
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))

    new_matrix = np.array(new_matrix)

    attention = np.array(new_matrix).T

    prev = -1
    new_index=  []
    for idx, row in enumerate(attention):
        token_id = tokenid2word_mapping[idx]
        if token_id != prev:
            new_index.append( [row])
            prev = token_id
        else:
            new_index[-1].append(row)

    
    new_matrix = []
    for row in new_index:
        new_matrix.append(operator(np.array(row), 0))
    
    new_matrix = np.array(new_matrix)

    return new_matrix.T

def index2word(tokenid2word_mapping, token2id):
    tokens = []
    prev = -1
    for token_id in tokenid2word_mapping:
        if token_id == prev:
            continue

        tokens.append(token2id[token_id])
        prev = token_id

    return tokens



if __name__ == '__main__':
    import en_core_web_sm
    from transformers import AutoTokenizer, BertModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoder = BertModel.from_pretrained('bert-base-cased')
    nlp = en_core_web_sm.load()

    sentence = 'Rolling Stone wrote: “No other pop song has so thoroughly challenged artistic conventions”'
    sentence = 'Dylan sing "Time They Are Changing"'
    inputs, tokenid2word_mapping, token2id, noun_chunks  = create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer)

    outputs = encoder(**inputs, output_attentions=True)
    print(noun_chunks, tokenid2word_mapping, token2id)
