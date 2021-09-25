import sys, os
from re_process import process_sentence
from re_utils import AttnMapExtractor
from mapper import Map, deduplication
import argparse
from tqdm import tqdm
import json
import en_core_web_md
from spacy.language import Language
import re
import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from bert import tokenization, modeling
import numpy as np

## the format of how to store the relations##
## appln_id:{__: {name:__ CPC:___ relations:___} }

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

file_path= '../patstat_process/patent_dic.json'


def str2bool(v):

    """
    standarlize the str to 'True' or 'False'
    """


    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# parse the input argument
parser = argparse.ArgumentParser(description='Process lines of text corpus into knowledgraph')
parser.add_argument('--input_filename', default = 'patent_data/patent_test_example.json',type=str, help='text file as input')
# parser.add_argument('input_filename', type=str, help='text file as input')
parser.add_argument('--output_filename', default='patent_data/patent_test_example_result.json',type = str, help='output text file')
# parser.add_argument('output_filename', type=str, help='output text file')
parser.add_argument('--language_model',default='bert-large-cased',
                    choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl','allenai/scibert_scivocab_uncased'],
                    help='which language model to use')
# parser.add_argument('--language_model',default='bert-base-cased',
#                     choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl','allenai/scibert_scivocab_uncased'],
#                     help='which language model to use')
parser.add_argument('--use_cuda', default=False,
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
nlp =spacy.load(args.spacy_model)
# overwrite to add the boundry rule, add ';'
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





if __name__ == '__main__':
    language_model = args.language_model
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output
    cased = True
    # tokenizer = AutoTokenizer.from_pretrained(language_model)
    # encoder = BertModel.from_pretrained(language_model)
    bert_dir = "bert/uncased_L-12_H-768_A-12"

    extractor = AttnMapExtractor(
        os.path.join(bert_dir, "bert_config.json"),
        os.path.join(bert_dir, "bert_model.ckpt"),
        max_sequence_length=512, debug=False
    )

    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, "vocab.txt"),
        do_lower_case=cased)


    # disable the dropout by setting to evaluation mode

    count = 0

    with open(input_filename, 'r') as f, open(output_filename, 'a+') as g:

        bulk = json.load(f)
        # the patent data is in the format of dictionary
        # {"appln_id": {"appln_title":....,"appln_abstract":...,"CPC_class_symbol":[..,..]},"appln_id":......}

        for key, value in tqdm(bulk.items()):
            count = count+1
            # if count==150580 or count ==87932:

            #  number hasnt fixed, 87931 ,150579 , 183765 with '\t' inside


            if type(value['appln_abstract']) is str:
                sentence = re.sub(' +', ' ', value['appln_abstract'].strip().lower())
            # modification ends here###
            # if len(sentence):
                valid_triplets = []
                count=0
                for sent in nlp(sentence).sents:
                    for triplets in process_sentence(sent.text, tokenizer,nlp,extractor, use_cuda=use_cuda):
                        valid_triplets.append(triplets)

                # TODO time count found the session part takes 52.79 s, the all processing time is 53.76s
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