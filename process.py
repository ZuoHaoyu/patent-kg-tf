from utils import compress_attention, create_mapping, BFS, build_graph, is_word
from multiprocessing import Pool
import spacy
import en_core_web_md
import torch
from transformers import AutoTokenizer, BertModel, GPT2Model
from constant import invalid_relations_set


def process_matrix(attentions, layer_idx = -1, head_num = 0, avg_head=False, trim=True, use_cuda=True):
    if avg_head:
        if use_cuda:
            attn =  torch.mean(attentions[0][layer_idx], 0).cpu()
        else:
            attn = torch.mean(attentions[0][layer_idx], 0)
        attention_matrix = attn.detach().numpy()
    else:
        attn = attentions[0][layer_idx][head_num]
        if use_cuda:
            attn = attn.cpu()
        attention_matrix = attn.detach().numpy()

    attention_matrix = attention_matrix[1:-1, 1:-1]

    return attention_matrix

def bfs(args):
    s, end, graph, id2token,max_size, black_list_relation = args
    return BFS(s, end, graph, id2token,max_size, black_list_relation)


def check_relations_validity(relations):
    for rel in relations:
        if rel.lower() in invalid_relations_set or rel.isnumeric():
            return False
    return True

def global_initializer(nlp_object):
    global spacy_nlp
    spacy_nlp = nlp_object


def filter_relation_sets(params):
    triplet, id2token = params
    triplet_idx = triplet[0]
    confidence = triplet[1]
    head, tail = triplet_idx[0], triplet_idx[-1]
    if head in id2token and tail in id2token:
        head = id2token[head]
        tail = id2token[tail]
        relations = [ spacy_nlp(id2token[idx])[0].lemma_  for idx in triplet_idx[1:-1] if idx in id2token ]
        if len(relations) > 0 and check_relations_validity(relations) and head.lower() not in invalid_relations_set and tail.lower() not in invalid_relations_set:
            return {'h': head, 't': tail, 'r': relations, 'c': confidence }
    return {}



def filter_relation_only(all_relation_pairs):
    all_triplets_c = []
    triplets = []
    cs = []
    for i in all_relation_pairs:
        triplet_c, id2token = i
        all_triplets_c.append(triplet_c)

    for triplet, c in all_triplets_c:
        triplets.append(triplet)
        cs.append(c)


    relation_ls = list()
    for triple in triplets:
        if triple[1] not in relation_ls:
            relation_ls.append(triple[1])
    dic = dict()
    for r in relation_ls:
        dic[r] = []

    for triplet, c in all_triplets_c:
        dic[triplet[1]].append((triplet[0], triplet[2], c))

    value_filtered = []
    for key, value in dic.items():
        value_sorted= sorted(value, key=lambda x: x[2],reverse=True)
        # print ('old')
        # print (value_sorted)
        h_golden = value_sorted[0][0]
        count = 0
        for value_ in value_sorted:

            if value_[0] == h_golden:
                count = count+1
                value_filtered.append((([value_[0],key,value_[1]],value_[2]),id2token))
            if count ==2:
                break



    return (value_filtered)




def filter_tail_only(all_relation_pairs):
    all_triplets_c = []
    triplets = []
    cs = []
    for i in all_relation_pairs:
        triplet_c, id2token = i
        all_triplets_c.append(triplet_c)

    for triplet, c in all_triplets_c:
        triplets.append(triplet)
        cs.append(c)


    tail_ls = list()
    for triple in triplets:
        if triple[2] not in tail_ls:
            tail_ls.append(triple[2])
    dic = dict()
    for tail in tail_ls:
        dic[tail] = []

    for triplet, c in all_triplets_c:
        dic[triplet[2]].append((triplet[0], triplet[1], c))

    value_filtered = []
    for key, value in dic.items():
        value_sorted= sorted(value, key=lambda x: x[2],reverse=True)
        # print ('old')
        # print (value_sorted)
        h_golden = value_sorted[0][0]
        count = 0
        for value_ in value_sorted:

            if value_[0] == h_golden:
                count = count+1
                value_filtered.append((([value_[0],value_[1],key],value_[2]),id2token))
            if count ==2:
                break



    return (value_filtered)


def parse_sentence(sentence, tokenizer, encoder, nlp, use_cuda=True):
    '''Implement the match part of MAMA

    '''
    tokenizer_name = str(tokenizer.__str__)
###modification starts below, adding id2token in the returned parameters
    inputs, tokenid2word_mapping, token2id, noun_chunks,id2token,start_chunk = create_mapping(sentence, return_pt=True, nlp=nlp, tokenizer=tokenizer)
    if inputs['input_ids'].size(1)>=512:
        # print (inputs['input_ids'].size(1))
        #
        return []
###modification ends here
    with torch.no_grad():
        if use_cuda:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
        outputs = encoder(**inputs, output_attentions=True)
    trim = True
    if 'GPT2' in tokenizer_name:
        trim  = False

    '''
    Use average of last layer attention : page 6, section 3.1.2
    '''
    #  attention , outputs is bsz * heads* seq_len*seq_len,
    #  setting avg_head to false

    attention = process_matrix(outputs[2], avg_head=True, trim=trim, use_cuda=use_cuda)

    merged_attention = compress_attention(attention, tokenid2word_mapping)
    attn_graph = build_graph(merged_attention)

#############modification starts here ################
        
    i=0
    id2noun_chunk_token=dict()
    for key,value in  id2token.items():
        if value in noun_chunks:

            id2noun_chunk_token[i]=key

            i=i+1


                
#############modification ends here ###########################
    #############
    
    def head_tail_pairs_append(head_tail_pairs,sent,id2noun_chunk_token,start_chunk,token2id):
#             print ('sent.start:%s, sent.end:%s'%(sent.start,sent.end))

        for i1,start1 in enumerate(start_chunk):
#                 print ('i1:%s, start1 :%s' %(i1, start1))
            if start1 < sent.start:
                continue
            if start1 >=sent.end:
                break
            for i2,start2 in enumerate(start_chunk):
    #             print ("i2:%s, start2 :%s"%(i2, start2))
                if start2<start1 :
                    continue
                if start2>= sent.end: 
                    break
                if i2-i1>3:
                    break
                if start1!=start2:
    #                 print ("succeed, start1 :%s,start2:%s"%(noun_chunks[i1],noun_chunks[i2]))
#                         print ('sc')
                    if i2==len(id2noun_chunk_token) and len(id2noun_chunk_token)!=len(start_chunk):
                        break
                    head_tail_pairs.append((id2noun_chunk_token[i1],id2noun_chunk_token[i2]))

            ##
        return head_tail_pairs
    tail_head_pairs = []
    for sent in nlp(sentence).sents:
        tail_head_pairs = head_tail_pairs_append(tail_head_pairs,sent,id2noun_chunk_token,start_chunk,token2id)
#should starts from here
# add'same sentence ' limit here
# first, the pair is limited in the same sentence, 2nd, the funciton is preventing meaningless iteration beyond same sentence.         
     
#     tail_head_pairs = []
#     for head in noun_chunks:
#         for tail in noun_chunks:
#             if head != tail:
#                 tail_head_pairs.append((token2id[head], token2id[tail]))

##should ends here
                    

    
    
    
    
#     tail_head_pairs = []
#     for head in noun_chunks:
#         for tail in noun_chunks:
#             if head != tail:
#                 tail_head_pairs.append((token2id[head], token2id[tail]))

    # black_list_relation = list(set([ token2id[n]  for n in noun_chunks ]))
    black_list_relation = list ( set(id2noun_chunk_token.values()))
    # adding adv to the black_list_relation£
    for key in token2id.keys():
        pos_parser = nlp(key)
        for pos in pos_parser:
            if pos.pos_ == 'ADV':
                black_list_relation.append(token2id[key])
    black_list_relation =list(set(black_list_relation))

    all_relation_pairs = []
    ###modicaition starts below, delete the origin id2token code, which is incorrectbb
    #id2token = { value: key for key, value in token2id.items()}
    ### modicaition endes here

    #
    # with Pool(10) as pool:
    #     params = [  ( pair[0], pair[1], attn_graph, id2token,max(tokenid2word_mapping), black_list_relation, ) for pair in tail_head_pairs]
    #     for output in pool.imap_unordered(bfs, params):
    #         if len(output):
    #             all_relation_pairs += [(output, id2token)]

                    # to test the BFS , don't use pool#
    for pair in tail_head_pairs:
        outputs= BFS(pair[0], pair[1], attn_graph, id2token,max(tokenid2word_mapping), black_list_relation, )

        if len(outputs):
            all_relation_pairs += [(outputs, id2token)]

                        # to test the BFS, don't use pool#
    #
    # print (all_relation_pairs
    all_relation_pairs = filter_relation_only(all_relation_pairs)
    all_relation_pairs = filter_tail_only(all_relation_pairs)
    triplet_text = []
#     with Pool(10, global_initializer, (nlp,)) as pool:
#         for triplet in pool.imap_unordered(filter_relation_sets, all_relation_pairs):
#             if len(triplet) > 0:
#                 triplet_text.append(triplet)
    
    for i in all_relation_pairs:
        triplet, id2token = i
        triplet_idx = triplet[0]
        confidence = triplet[1]
        relations = triplet_idx[1]
        head, tail = triplet_idx[0], triplet_idx[-1]
#         print ('triple_idx: [1] %s id2token[relations] : %s' % (str(triplet_idx[1]),id2token[relations]))

        if head in id2token and tail in id2token and relations in id2token:
            head = id2token[head]
            tail = id2token[tail]
#             relations = nlp(id2token[relations]).lemma_
            relations_ = nlp(id2token[relations])
            relations = " ".join([token.lemma_ for token in relations_])
#             relations = id2token[relations]
            triplet_text.append( {'h': head, 't': tail, 'r': relations, 'c': confidence })    
    return triplet_text
#     for i in all_relation_pairs:
#         i
#     return {'h': head, 't': tail, 'r': relations, 'c': confidence }
#     triplet_text = all_relation_pairs 
#     return all_relation_pairs


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    nlp = en_core_web_md.load()
    selected_model = 'gpt2-medium'

    use_cuda = True


    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    encoder = GPT2Model.from_pretrained(selected_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()

    target_file = [
        '../../Documents/KGERT-v2/datasets/squad_v1.1/train-v1.1.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1.1/wiki_dev_2020-18.json',
        # '../../Documents/KGERT-v2/datasets/squad_v1/dev-v1.1.json',
    ]

    output_filename = [
        'train_v1.1.jsonl',
        # 'wiki_2020-18.jsonl',
        # 'dev-v1.1.jsonl',
    ]

    for target_file, output_filename in zip(target_file, output_filename):
        with open(target_file, 'r') as f:
            dataset = json.load(f)

        output_filename = selected_model +'_'+ output_filename

        print(target_file, output_filename)

        f = open(output_filename,'w')
        for data in tqdm(dataset['data'], dynamic_ncols=True):
            for para in data['paragraphs']:
                context = para['context']
                for sent in nlp(context).sents:
                    for output in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()

                for question in para['qas']:
                    question = question['question']
                    for output in parse_sentence(question, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        f.write(json.dumps(output)+'\n')
                f.flush()
        f.close()