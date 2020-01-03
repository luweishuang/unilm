from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tqdm
import argparse
from multiprocessing import Pool
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--bert_model", default="bert-large-cased", type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--do_lower_case",
                    default=False,
                    help="Set this flag if you are using an uncased model.")
args = parser.parse_args()


def process_detokenize(chunk):
    twd = TreebankWordDetokenizer()
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    r_list = []
    for idx, line in chunk:
        line = line.strip().replace('``', '"').replace('\'\'', '"').replace('`','\'')
        s_list = [twd.detokenize(x.strip().split(
            ' '), convert_parentheses=True) for x in line.split('<S_SEP>')]
        tk_list = [tokenizer.tokenize(s) for s in s_list]
        r_list.append((idx, s_list, tk_list))
    return r_list


def read_tokenized_file(fn):
    with open(fn, 'r', encoding='utf-8') as f_in:
        l_list = [l for l in f_in]
    num_pool = min(args.processes, len(l_list))
    p = Pool(num_pool)
    chunk_list = partition_all(int(len(l_list)/num_pool), list(enumerate(l_list)))
    r_list = []
    with tqdm(total=len(l_list)) as pbar:
        for r in p.imap_unordered(process_detokenize, chunk_list):
            r_list.extend(r)
            pbar.update(len(r))
    p.close()
    p.join()
    r_list.sort(key=lambda x: x[0])
    return [x[1] for x in r_list], [x[2] for x in r_list]


def append_sep(s_list):
    r_list = []
    for i, s in enumerate(s_list):
        r_list.append(s)
        r_list.append('[SEP_{0}]'.format(min(9, i)))
    return r_list[:-1]



def main():
    ## print('convert into src/tgt format')
    with open(os.path.join(args.output_dir, split_out + '.src'), 'w', encoding='utf-8') as f_src, open(os.path.join(args.output_dir, split_out +'.tgt'), 'w', encoding='utf-8') as f_tgt, open(os.path.join(args.output_dir, split_out+'.slv'), 'w', encoding='utf-8') as f_slv:
        for src, tgt, lb in tqdm(zip(article_tk, summary_tk, label)):
            # source
            src_tokenized = [' '.join(s) for s in src]
            if args.src_sep_token:
                f_src.write(' '.join(append_sep(src_tokenized)))
            else:
                f_src.write(' '.join(src_tokenized))
            f_src.write('\n')
            # target (silver)
            slv_tokenized = [s for s, extract_flag in zip(
                src_tokenized, lb) if extract_flag]
            f_slv.write(' [X_SEP] '.join(slv_tokenized))
            f_slv.write('\n')
            # target (gold)
            f_tgt.write(' [X_SEP] '.join(
                [' '.join(s) for s in tgt]))
            f_tgt.write('\n')


if __name__ == "__main__":
    main()
