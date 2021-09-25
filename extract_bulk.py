import sys, os
from process import parse_sentence
from mapper import Map, deduplication
from transformers import AutoTokenizer, BertModel, GPT2Model
import argparse
import en_core_web_md
from tqdm import tqdm
import json
import spacy
from spacy.language import Language
import re
import numpy as np
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

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
parser.add_argument('--input_filename', default = 'patent_data/patent_dic.json',type=str, help='text file as input')
# parser.add_argument('input_filename', type=str, help='text file as input')
parser.add_argument('--output_filename', default='patent_data/patent_dic_result_2nd.json',type = str, help='output text file')
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
parser.add_argument('--threshold', default=0.041,
                        type=float, help="Any attention score lower than this is removed")

args = parser.parse_args()

use_cuda = args.use_cuda
nlp = spacy.load(args.spacy_model)
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i + 1].is_sent_start = True
    return doc
nlp.add_pipe("set_custom_boundaries", before="parser")



# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer


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
    count = 0
    with open(input_filename, 'r') as f, open(output_filename, 'a+') as g:

        bulk = json.load(f)



        for key, value in tqdm(bulk.items()):
            count = count+1
            # if count==150580 or count ==87932:
            if count<=370494:
                continue
            #  number hasnt fixed, 87931 ,150579 , 183765 with '\t' inside
            ##modification starts below###
            #         sentence= line.strip()

            if type(value['appln_abstract']) is str:
                sentence = re.sub(' +', ' ', value['appln_abstract'].strip().lower())
            # modification ends here###
            # if len(sentence):
                valid_triplets = []
                for sent in nlp(sentence).sents:
                    # Match

                    for triplets in parse_sentence(sent.text, tokenizer, encoder, nlp, use_cuda=use_cuda):
                        valid_triplets.append(triplets)


                if len(valid_triplets) > 0:
                    # Map
                    mapped_triplets = []
                    articles = ['a ', 'an ', 'the ']
                    con_list = []
                    for triplet in valid_triplets:
                        con_list.append(triplet['c'])
                    median = np.median(con_list)
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                        for article in articles:
                            head = head.replace(article, '')
                            tail = tail.replace(article, '')
                            relations = relations.replace(article, '')
                        conf = triplet['c']
                        # if conf < args.threshold:
                        # if conf< median:
                        if conf<0.01:
                            continue
                        mapped_triplet = Map(head, relations, tail)
                        if 'h' in mapped_triplet:
                            mapped_triplet['c'] = conf
                            mapped_triplets.append(mapped_triplet)
                    output = {'appln_id': key, 'relationships': deduplication(mapped_triplets)}

                    if include_sentence:
                        output['sent'] = sentence
                    if len(output['relationships']) > 0:
                        g.write(json.dumps(output) + '\n')
