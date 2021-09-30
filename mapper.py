from constant import invalid_relations_set
sqlite_path = "../../Documents/wiki_2019/generated"

def re_join(head, relations, tail):

    head_p_e_m = head
    tail_p_e_m = tail
####改了这下面### 应该是因为 str被循环拆分开了 ####

    if relations not in invalid_relations_set  and len(relations) > 1:
        valid_relations = relations 
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


    

# if __name__ == "__main__":
#
#     p_e_m = emb.wiki("Bob", 'wiki')[:10]
#     print(p_e_m)
