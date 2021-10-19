from utils import reverse_bfs,get_certain_attention_matrix, matrix_compress,create_mapping, build_graph,hidesep
from utils import filter_relation_only, filter_tail_only
from multiprocessing import Pool
from bert import tokenization, modeling
import tensorflow as tf
import numpy as np
import os
import time


def process_sentence(sentence, tokenizer, nlp,extractor, use_cuda=True):
    """
    process a single stence into triplets result --> ({'h': head, 't': tail, 'r': relations, 'c': confidence})
    :param sentence: sentence
    :param tokenizer:  the bert tokenizaer to do embedding
    :param nlp:  spacy mode to recognize noun phrase and verb phrase
    :param extractor: attention extractor sess
    :param use_cuda:
    :return: riplets result --> ({'h': head, 't': tail, 'r': relations, 'c': confidence})
    """

    ###modification starts below, adding id2token in the returned parameters
    inputs, tokenid2word_mapping, token2id, noun_chunks, id2token, start_chunk, length_after_tokenizer = create_mapping(
        sentence,
        return_pt=True, nlp=nlp,
        tokenizer=tokenizer)
    # # can't process a sentence longer than 512 in BERT
    # if inputs['input_ids'].size(1) >= 512:
    #     return []
    # # no gradients, save the RAM
    # with torch.no_grad():
    #     if use_cuda:
    #         for key in inputs.keys():
    #             inputs[key] = inputs[key].cuda()
    #     outputs = encoder(**inputs, output_attentions=True)
    # trim = True
    # if 'GPT2' in tokenizer_name:
    #     trim = False
    #
    # '''
    # Use average of last layer attention : page 6, section 3.1.2
    # '''
    # #  attention , outputs is bsz * heads* seq_len*seq_len,
    # #  setting avg_head to false

    print("Extracting attention maps...")
    feature_dicts_with_attn = []
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    start_time= time.time()
    with tf.Session( ) as sess:
        sess.run(tf.global_variables_initializer())  # ????
        attns = extractor.get_attn_maps(sess, inputs)
        attns = tf.squeeze(attns)
        attns = attns[:, :, 0:length_after_tokenizer, 0:length_after_tokenizer]
        attns = attns.eval(session=sess)
    # TODO time count found the session part takes 52.79 s, the all processing time is 53.76s

    end_time=time.time()
    time_elapse=end_time-start_time
    attention = get_certain_attention_matrix(attns, layer_idx=7, head_num=9, use_cuda=use_cuda)

    attention = hidesep(attention)
    attention = attention[1:-1, 1:-1]
    # merged_attention = compress_attention(attention, tokenid2word_mapping)
    merged_attention = matrix_compress(attention, tokenid2word_mapping)



    attn_graph = build_graph(merged_attention)
    # TODO at is not recoginized when at the start?
    #############modification starts here ################

    i = 0
    id2noun_chunk_token = dict()
    for key, value in id2token.items():
        if value in noun_chunks:
            id2noun_chunk_token[i] = key

            i = i + 1

    #############modification ends here ###########################
    #############

    def head_tail_pairs_append(head_tail_pairs, sent, id2noun_chunk_token, start_chunk, token2id):
        #             print ('sent.start:%s, sent.end:%s'%(sent.start,sent.end))

        for i1, start1 in enumerate(start_chunk):
            #                 print ('i1:%s, start1 :%s' %(i1, start1))
            if start1 < sent.start:
                continue
            if start1 >= sent.end:
                break
            for i2, start2 in enumerate(start_chunk):
                #             print ("i2:%s, start2 :%s"%(i2, start2))
                if start2 < start1:
                    continue
                if start2 >= sent.end:
                    break
                if i2 - i1 > 3:
                    break
                if start1 != start2:
                    #                 print ("succeed, start1 :%s,start2:%s"%(noun_chunks[i1],noun_chunks[i2]))
                    #                         print ('sc')
                    if i2 == len(id2noun_chunk_token) and len(id2noun_chunk_token) != len(start_chunk):
                        break
                    head_tail_pairs.append((id2noun_chunk_token[i1], id2noun_chunk_token[i2]))

            ##
        return head_tail_pairs

    tail_head_pairs = []
    for sent in nlp(sentence).sents:
        tail_head_pairs = head_tail_pairs_append(tail_head_pairs, sent, id2noun_chunk_token, start_chunk, token2id)
    # should starts from here
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
    black_list_relation = list(set(id2noun_chunk_token.values()))
    # adding adv to the black_list_relation£
    for key in token2id.keys():
        pos_parser = nlp(key)
        for pos in pos_parser:
            if pos.pos_ == 'ADV':
                black_list_relation.append(token2id[key])
    black_list_relation = list(set(black_list_relation))

    all_relation_pairs = []
    ###modicaition starts below, delete the origin id2token code, which is incorrectbb
    # id2token = { value: key for key, value in token2id.items()}
    ### modicaition endes here

    #
    # with Pool(10) as pool:
    #     params = [  ( pair[0], pair[1], attn_graph, id2token,max(tokenid2word_mapping), black_list_relation, ) for pair in tail_head_pairs]
    #     for output in pool.imap_unordered(bfs, params):
    #         if len(output):
    #             all_relation_pairs += [(output, id2token)]

    # to test the BFS , don't use pool#
    # here reversed the head and tail
    for pair in tail_head_pairs:
        outputs = reverse_bfs(pair[1], pair[0], attn_graph, id2token, max(tokenid2word_mapping), black_list_relation, )

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
            triplet_text.append({'h': head, 't': tail, 'r': relations, 'c': confidence})
    return triplet_text
#     for i in all_relation_pairs:
#         i
#     return {'h': head, 't': tail, 'r': relations, 'c': confidence }
#     triplet_text = all_relation_pairs
#     return all_relation_pairs
