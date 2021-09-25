from constant import invalid_relations_set

# from REL.db.generic import GenericLookup
sqlite_path = "../../Documents/wiki_2019/generated"
# emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")



# def Map(head, relations, tail, top_first=True, best_scores = True):
#     if head == None or tail == None or relations == None:
#         return {}
#     head_p_e_m = emb.wiki(str(head), 'wiki')
#     if head_p_e_m is None:
#         return {}
#     tail_p_e_m = emb.wiki(str(tail), 'wiki')
#     if tail_p_e_m is None:
#         return {}
#     tail_p_e_m = tail_p_e_m[0]
#     head_p_e_m = head_p_e_m[0]
#     valid_relations = [ r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1 ]
#     if len(valid_relations) == 0:
#         return {}


#     return { 'h': head_p_e_m[0], 't': tail_p_e_m[0], 'r': '_'.join(valid_relations)  }


def Map(head, relations, tail, top_first=True, best_scores = True):
#     if head == None or tail == None or relations == None:
#         return {}
#     head_p_e_m = emb.wiki(str(head), 'wiki')
#     if head_p_e_m is None:
#         head_p_e_m = head
#     else:
#         head_p_e_m = head_p_e_m[0][0]   
#     tail_p_e_m = emb.wiki(str(tail), 'wiki')
#     if tail_p_e_m is None:
#         tail_p_e_m = tail
#     else:
#         tail_p_e_m = tail_p_e_m[0][0]
     


#     valid_relations = [ r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1 ]
#     if len(valid_relations) == 0:
#         return {}
    head_p_e_m = head
    tail_p_e_m = tail
####改了这下面### 应该是因为 str被循环拆分开了 ####
#     print ('1 %s, 2 %s 3 %s ' % (relations not in invalid_relations_set,relations.isalpha(),len(relations) > 1))
#     valid_relations = [ r for r in relations if r not in invalid_relations_set and r.isalpha() and len(r) > 1 ]
    if relations not in invalid_relations_set  and len(relations) > 1:
        valid_relations = relations 
#         print (valid_relations)
        return { 'h': head_p_e_m, 't': tail_p_e_m, 'r': valid_relations }
    else:
        return {}
#####改了这上面， r:_join(valid_relations) 其实他默认的是relation是词组###


def deduplication(triplets):
    unique_pairs = []
    pair_confidence = []
    for t in triplets:
        key = '{}\t{}\t{}'.format(t['h'], t['r'], t['t'])
        conf = t['c']
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)
    
    unique_triplets = []
    for idx, unique_pair in enumerate(unique_pairs):
        h, r, t = unique_pair.split('\t')
        unique_triplets.append({ 'h': h, 'r': r, 't': t , 'c': pair_confidence[idx]})

    return unique_triplets


    

if __name__ == "__main__":
    emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")
    p_e_m = emb.wiki("Bob", 'wiki')[:10]
    print(p_e_m)
