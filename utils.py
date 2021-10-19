import numpy as np
import re
from constant import invalid_relations_set
alphabet = re.compile(r'^[a-zA-Z]+$')

from copy import copy
from collections import defaultdict
from spacy.matcher import PhraseMatcher

from bert import tokenization, modeling
import tensorflow as tf





def phrasal_verb_recognizer(parsed) :
    # TODO need test and add into the mapping part
    for token in parsed:
        if token.dep_ == "prt" and token.head.pos_ == "VERB":
            verb = token.head.orth_
            particle = token.orth_

class AttnMapExtractor(object):
  """Runs BERT over examples to get its attention maps."""

  def __init__(self, bert_config_file, init_checkpoint,
               max_sequence_length=128, debug=False):
    make_placeholder = lambda name: tf.placeholder(
        tf.int32, shape=[None, max_sequence_length], name=name)
    self._input_ids = make_placeholder("input_ids")
    self._segment_ids = make_placeholder("segment_ids")
    self._input_mask = make_placeholder("input_mask")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
    self._attn_maps = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=self._input_ids,
        input_mask=self._input_mask,
        token_type_ids=self._segment_ids,
        use_one_hot_embeddings=True).attn_maps

    if not debug:
      print("Loading BERT from checkpoint...")
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tf.trainable_variables(), init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

  def get_attn_maps(self, sess, inputs):

    feed = {
        self._input_ids: np.vstack(inputs['input_ids']).reshape(-1,512),  #vertical stack up
        self._segment_ids: np.vstack(inputs['segment_ids']).reshape(-1,512),
        self._input_mask: np.vstack(inputs['input_mask']).reshape(-1,512)
    }
    return sess.run(self._attn_maps, feed_dict=feed)

def create_mapping(sentence, return_pt=False, nlp=None, tokenizer=None):
    """
    #####
        EXAMPLE Sentence:'the magnetic bearing is leviated by the magnetic force'
        verb_phrase_chunk:['leviated by']
        noun_chunks:['the magnetic bearing', 'the magnetic force']
        sentence_mapping:['the magnetic bearing', 'is', 'leviated by', 'the magnetic force']
        token2id:{'the magnetic bearing': 0, 'is': 1, 'leviated by': 2, 'the magnetic force': 3}
        tokenid2word_mapping:[0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3]
        token_ids:[1103, 8364, 7343, 1110, 5837, 7137, 1906, 1118, 1103, 8364, 2049]
        id2token:{0: 'the magnetic bearing', 1: 'is', 2: 'leviated by', 3: 'the magnetic force'}

    :param sentence:
    :param return_pt:
    :param nlp: spacy model
    :param tokenizer: huggingface tokenizer
    :return:
    """


    matcher1 = PhraseMatcher(nlp.vocab, attr='POS')
    terms = ['relate to', 'positioned through', 'engageable with', 'so as to', 'makes it possible to', 'to receive']
    patterns = [nlp(term) for term in terms]
    matcher1.add('VerbPhrase', patterns)

    doc = nlp(sentence)


    matches = matcher1(doc)
    verb_phrase_start_chunk = []
    verb_phrase_end_chunk = []
    verb_phrase_chunk = []
    start_holder=-1
    end_holder=-2
    for match_id, start, end in matches:
        start_holder=start
        if start_holder<end_holder:
            continue
        span = doc[start:end]
        verb_phrase_start_chunk.append(start)
        verb_phrase_end_chunk.append(end)
        verb_phrase_chunk.append(span.text)
        end_holder=end






    noun_start_chunk = []
    noun_end_chunk = []
    noun_chunks = []
    for chunk in doc.noun_chunks:
        noun_chunks.append(chunk.text)
        noun_start_chunk.append(chunk.start)
        noun_end_chunk.append(chunk.end)

    # sometimes verb_phrase_start_chunk is overlapping with noun_start_chunk , so pop the overlapping in the verb_phrase_chunk

    for i, start in enumerate(verb_phrase_start_chunk):
        if start in noun_start_chunk:
            verb_phrase_start_chunk.pop(i)
            verb_phrase_end_chunk.pop(i)
            verb_phrase_chunk.pop(i)

    for i, end in enumerate(verb_phrase_end_chunk):
        if end in noun_end_chunk:
            verb_phrase_start_chunk.pop(i)
            verb_phrase_end_chunk.pop(i)
            verb_phrase_chunk.pop(i)

    #         start_chunk= sorted(start_chunk + verb_phrase_start_chunk)
    #         end_chunk = sorted (end_chunk +verb_phrase_end_chunk)
    chunk_id = 0
    token2id = {}
    mode = 1
    sentence_mapping = []

    #### modification starts below,
    ### token2id [sentence_mapping[-1]]= len(token2id) --->token2id [sentence_mapping[-1]]= len(sentence_mapping)-1
    ### because token2id is a dictionary, when you len it, it will automatically remove the duplicated keys, this will
    ### cause problems when the words occurs more than once

    dic = dict(zip(noun_start_chunk + verb_phrase_start_chunk, noun_chunks + verb_phrase_chunk))

    dic = {k: v for k, v in sorted(dic.items(), key=lambda dic: dic[0], reverse=False)}

    start_chunk = list(dic.keys())
    chunks = list(dic.values())
    end_chunk = sorted(noun_end_chunk + verb_phrase_end_chunk)


    #  create sentence mapping and token2id
    # TODO the name of token2id is not appropriate, should be combined_token2id
    # TODO sentence mapping is also not appropriate , should be sentence_seg


    mode=0
    for idx, token in enumerate(doc):

        if idx in start_chunk:
            mode = 1
            sentence_mapping.append(chunks[chunk_id])
            token2id[sentence_mapping[-1]] = len(sentence_mapping) - 1
            chunk_id += 1
            continue

        if idx in end_chunk:
            mode = 0

        if mode == 0:
            sentence_mapping.append(token.text)
            token2id[sentence_mapping[-1]] = len(sentence_mapping) - 1


    token_ids = []
    tokenid2word_mapping = []
    ###modification starts below, adding id2token
    id2token = {key: value for key, value in enumerate(sentence_mapping)}
    ###modification ends here


    for token in sentence_mapping:
        subtoken_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(token)))
        tokenid2word_mapping += [token2id[token]] * len(subtoken_ids)
        token_ids += subtoken_ids
    input_token_ids=tokenizer.convert_tokens_to_ids(["[CLS]"]) +token_ids+tokenizer.convert_tokens_to_ids(["[SEP]"])
    length_after_tknizer=len(input_token_ids)

    outputs = {
        'input_ids': input_token_ids,
        'input_mask': [1] * (len(token_ids) + 2),
        'segment_ids': [0] * (len(token_ids) + 2)

            }
    while len(outputs['input_ids']) < 512:
      outputs['input_ids'].append(0)
      outputs['input_mask'].append(0)
      outputs['segment_ids'].append(0)

    ###modification starts below, adding id2token
    return outputs, tokenid2word_mapping, token2id, noun_chunks, id2token, noun_start_chunk,length_after_tknizer


def get_certain_attention_matrix(attentions, layer_idx=-1, head_num=0, use_cuda=True):
    """
    To get the attention matrix in the BERT 144 attention heads.
    can be average of a certain layer, or a certain head of a certian layer.
    :param attentions: the attention outputs from BERT
    :param layer_idx: the idx of layer , [0,11]
    :param head_num: the idx of head number in a certain layer, [0,11]
    :param avg_head: Bool value, compute the average or not
    :param use_cuda:
    :return:
    """
    # TODO use_cuda modify .cpu to .cuda()


    attn = attentions[layer_idx][head_num]
    if use_cuda:
        attn = attn.cuda()
    # attention_matrix = attn.detach().numpy()
    # TODO use_cuda here is not activated
    attention_matrix = attn[:, :]

    return attention_matrix

def matrix_compress(attention, tokenid2word_mapping):
    """
    For attention from a phrase, the mean ofthe attention weights is calculated over the tokens. For attention to a phrase, the sum of
    attention weights is calculated over the tokens. These transformations preserve the property
    that the attention from one to other sums to be one
    :param attention:
    :param tokenid2word_mapping:
    :return:
    """

    operate_list=[]

    # tokenid_set should be in the same sequence of a deduplicated tokenid2word_mapping list

    def depulicate (x):
        mark=-1
        depulicated_x=[]
        for i in x:
            if i !=mark:
                depulicated_x.append(i)
                mark=i
        return depulicated_x

    tokenid_set = depulicate(tokenid2word_mapping)


    for tokenid in tokenid_set:

        operate_list.append( [i for i, x in enumerate(tokenid2word_mapping) if x == tokenid])

    #     shouldn't be the set of tokenid2wordmapping, will remove duplicate words' positio
    #     the lenght should be the word2id
    #     operate_list, remake
    #     should combine the

    b = np.vstack([np.mean(attention[operate_list[n], :], 0) for n in range(len(tokenid_set))])
    c = np.vstack([np.sum(b[:, operate_list[n]], 1) for n in range(len(tokenid_set))]).T
    return c


def hidesep(attn):
    """
    hide the CLS and SEP token in the matrix, and convert the remaining part to a larger part
    :param attn: the processing matrix
    :return:  processed matrix
    """
    attn = np.array(attn)

    attn[:, 0] = 0
    attn[:, -1] = 0

    attn /= attn.sum(axis=-1, keepdims=True)

    # the 0 on axis -1 is processed after the ampify, otherwise on the first row and last row, the denominator will be 0
    attn[0, :] = 0
    attn[-1, :] = 0
    return attn

def build_graph(matrix):
    """

    :param matrix:  convert the lower part of the matrix into a dictionary for later index.
                        ATTETION: ONLY THE  LOWER PART

    :return: a dictionary contain the key node and query node and the attention score between.
    defaultdict(<class 'list'>, {3: [(2, 0.3744966), (1, 0.3834611), (0, 0.03559453)], 2: [(1, 0.34131923), (0, 0.022888258)]})
    """

    graph = defaultdict(list)
    for idx in range(len(matrix)-1,-1,-1):
        for col in range(idx-1, -1,-1):
            graph[idx].append((col, matrix[idx][col]))
    return graph



def check_single_relation_validity(relations):
    if relations.lower() in invalid_relations_set or relations.isnumeric():
        return False
    else:
        return True

def candidate_facts_relation_filter(candidate_facts,id2token):
    filtered_candidate_facts=[]
    for candidate_fact in candidate_facts:
        head=candidate_fact[0][2]
        tail=candidate_fact[0][0]
        relations=candidate_fact[0][1]

        if len(id2token[relations]) > 0 and check_single_relation_validity(id2token[relations]) and id2token[head].lower() not in invalid_relations_set and id2token[tail].lower() not in invalid_relations_set:
            # filtered_candidate_facts.append(candidate_fact)
            filtered_candidate_facts.append(([head,relations,tail],candidate_fact[1]))
#     print ('filterd_c_f{} '.format(filtered_candidate_facts))
    if len(filtered_candidate_facts)>0:
        filtered_candidate_facts = sorted(filtered_candidate_facts, key=lambda x: x[1], reverse=True)

        filtered_candidate_facts = filtered_candidate_facts[0]
        return (filtered_candidate_facts)
    if len(filtered_candidate_facts)==0:
        return []

def reverse_bfs(s, end, graph, id2token, max_size=-1 ,black_list_relation=[]):
    """

    :param s:  start node, NOTE that in the reverse searching, start and end is reversed.
    :param end:  end node, NOTE that in the reverse searching, start and end is reversed.
    :param graph: the dictionary containing the attention
    :param id2token:  id2token:{0: 'the magnetic bearing', 1: 'is', 2: 'leviated by', 3: 'the magnetic force'}
    :param max_size:
    :param black_list_relation: the black list relation including h,t and self-defined relation
    :return: candidate facts
    """
    #     print ('s %s, end %s ' % (str(s),str(end)))
    visited = [False] * (max(graph.keys()) + 100)

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
                found_paths.append(path + [(i, conf)])
                break
            if visited[i] == False:
                queue.append((i, copy(path) + [(i, conf)]))
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
    candidate_facts = candidate_facts_relation_filter(candidate_facts, id2token)

    return candidate_facts


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
