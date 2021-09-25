import sys, os
from process import parse_sentence
from mapper import Map, deduplication
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_md
from tqdm import tqdm
import json
import spacy
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

## the format of how to store the relations##
## appln_id:{__: {name:__ CPC:___ relations:___} }
file_path= '../patstat_process/patent_dic.json'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('--input_filename', default = '../../patstat_process/patent_paper_example2.json',type=str, help='text file as input')
# parser.add_argument('input_filename', type=str, help='text file as input')
parser.add_argument('--output_filename', default='../../patstat_process/patent_paper_result_example2.json',type = str, help='output text file')
# parser.add_argument('output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-large-cased',
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl','allenai/scibert_scivocab_uncased'],
                    help='which language model to use')
# parser.add_argument('--language_model',default='bert-base-cased',
#                     choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl','allenai/scibert_scivocab_uncased'],
#                     help='which language model to use')
parser.add_argument('--use_cuda', default=True,
                        type=str2bool, nargs='?',
                        help="Use cuda?")
parser.add_argument('--include_text_output', default=False,
                        type=str2bool, nargs='?',
                        help="Include original sentence in output")
parser.add_argument('--spacy_model',default='en_core_web_md',
                    choices=['en_core_sci_lg', 'en_core_web_md', 'en_ner_bc5cdr_md'],
                    help='which spacy model to use')
parser.add_argument('--threshold', default=0.01,
                        type=float, help="Any attention score lower than this is removed")

args = parser.parse_args()

use_cuda = args.use_cuda
nlp = spacy.load(args.spacy_model)

language_model = args.language_model

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    if 'gpt2' in language_model:
        encoder = GPT2Model.from_pretrained(language_model)
    else:
        encoder = BertModel.from_pretrained(language_model)
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output
    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:

        bulk = json.load(f)



        for key, value in tqdm(bulk.items()):
            ##modification starts below###
            #         sentence= line.strip()
            sentence = re.sub(' +', ' ', value['appln_abstract'].strip().lower())
            # modification ends here###
            if len(sentence):
                valid_triplets = []
                for sent in nlp(sentence).sents:
                    # Match

                    for triplets in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        valid_triplets.append(triplets)

                print(valid_triplets)
                if len(valid_triplets) > 0:
                    # Map
                    mapped_triplets = []
                    articles = ['a', 'an', 'the']
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                        for article in articles:
                            head = head.lstrip(article).strip()
                            tail = tail.lstrip(article).strip()
                            relations = relations.lstrip(article).strip()
                        conf = triplet['c']
                        if conf < args.threshold:
                            continue
                        mapped_triplet = Map(head, relations, tail)
                        if 'h' in mapped_triplet:
                            mapped_triplet['c'] = conf
                            mapped_triplets.append(mapped_triplet)
                    output = {'appln_id': key, 'relationships': deduplication(mapped_triplets)}

                    if include_sentence:
                        output['sent'] = sentence
                    if len(output['tri']) > 0:
                        g.write(json.dumps(output) + '\n')
