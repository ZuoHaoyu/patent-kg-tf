from re_utils import phrasal_verb_recognizer,create_mapping,matrix_compress,get_certain_attention_matrix
from transformers import AutoTokenizer, BertModel, GPT2Model
import spacy
import torch
import os
use_cuda=False
language_model  ='bert-large-cased'
spacy_model = 'en_core_web_md'
from bert import tokenization, modeling
import tensorflow as tf
import numpy as np
encoder = BertModel.from_pretrained(language_model)
from collections import defaultdict

# disable the dropout by setting to evaluation mode
encoder.eval()

if use_cuda:
    encoder = encoder.cuda()

nlp =spacy.load(spacy_model)


def hidesep(attn):
    attn = np.array(attn)

    attn[:, 0] = 0
    attn[:, -1] = 0

    attn /= attn.sum(axis=-1, keepdims=True)

    # the 0 on axis -1 is processed after the ampify, otherwise on the first row and last row, the denominator will be 0
    attn[0, :] = 0
    attn[-1, :] = 0
    return attn


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


  # def get_attn_maps(self, sess, examples):
  #   feed = {
  #       self._input_ids: np.vstack([e.input_ids for e in examples]),  #vertical stack up
  #       self._segment_ids: np.vstack([e.segment_ids for e in examples]),
  #       self._input_mask: np.vstack([e.input_mask for e in examples])
  #   }
  #   return sess.run(self._attn_maps, feed_dict=feed)





  def get_attn_maps(self, sess, inputs):




    feed = {
        self._input_ids: np.vstack(inputs['input_ids']).reshape(-1,512),  #vertical stack up
        self._segment_ids: np.vstack(inputs['segment_ids']).reshape(-1,512),
        self._input_mask: np.vstack(inputs['input_mask']).reshape(-1,512)
    }
    return sess.run(self._attn_maps, feed_dict=feed)



# 1
bert_dir="bert/uncased_L-12_H-768_A-12"
cased=False
tokenizer = tokenization.FullTokenizer(
    vocab_file=os.path.join(bert_dir, "vocab.txt"),
    do_lower_case=cased)
# 1
sentence="a millisecond-level control unit is electrically connected between the relay and the DCS control cabinet"
# sentence = "the magnetic force provided levitates the shaft"
# sentence = "a bearingless hub assembly comprises a rim to receive a tube magnet"
# create is modified, inputs format is modified
inputs, tokenid2word_mapping, token2id, noun_chunks, id2token, start_chunk,length_after_tokenizer = create_mapping(sentence,
                                                                                            return_pt=True, nlp=nlp,
                                                                                            tokenizer=tokenizer)

extractor = AttnMapExtractor(
    os.path.join(bert_dir, "bert_config.json"),
    os.path.join(bert_dir, "bert_model.ckpt"),
    max_sequence_length=512, debug=False
)

print("Extracting attention maps...")
feature_dicts_with_attn = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # ????
    attns = extractor.get_attn_maps(sess, inputs)
    attns=tf.squeeze(attns)
    attns=attns[:,:,0:length_after_tokenizer,0:length_after_tokenizer]
    attns=attns.eval(session=sess)


# 0
# with torch.no_grad():
#     if use_cuda:
#         for key in inputs.keys():
#             inputs[key] = inputs[key].cuda()
#     outputs = encoder(**inputs, output_attentions=True)
# trim = True
# 0


'''
Use average of last layer attention : page 6, section 3.1.2
'''
#  attention , outputs is bsz * heads* seq_len*seq_len,
#  setting avg_head to false



attention = get_certain_attention_matrix(attns,layer_idx=7,head_num=9, avg_head=False, use_cuda=use_cuda)



attention=hidesep(attention)
attention=attention[1:-1,1:-1]
# merged_attention = compress_attention(attention, tokenid2word_mapping)
merged_attention=matrix_compress(attention,tokenid2word_mapping)
# attn_graph = build_graph(merged_attention)


def build_graph(matrix):

    graph = defaultdict(list)
    for idx in range(len(matrix)-1,-1,-1):
        for col in range(idx-1, -1,-1):
            graph[idx].append((col, matrix[idx][col]))
    return graph
graph=build_graph(merged_attention)

# print(merged_attention)
# can't process a sentence longer than 512 in BERT





