from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import codecs
import re
import numpy as np
import json
import argparse
import list2bert

def load_conllu(file):
  with codecs.open(file, 'rb') as f:
    reader = codecs.getreader('utf-8')(f)
    buff = []
    for line in reader:
      line = line.strip()
      if line and not line.startswith('#'):
        if not re.match('[0-9]+[-.][0-9]+', line):
          buff.append(line.split('\t')[1])
      elif buff:
        yield buff
        buff = []
    if buff:
      yield buff

def load_emb(file):
  with codecs.open(file, 'r') as fi:
    line = fi.readline()
    n, dim = line.strip().split()
    print ('Loading input embedding from {} with number:{}, dimension:{}'.format(file, n, dim))
    embeddings = np.zeros([int(n), int(dim)], dtype=np.float32)
    line = fi.readline()
    tok2id = {}
    cur_idx = 0
    while line:
      line = line.split(' ')
      embeddings[cur_idx] = line[1:]
      #tokens.append(line[0])
      tok2id[line[0]] = cur_idx
      cur_idx += 1
      line = fi.readline()
  print ('Finish loading input embeddings, map length:{}, matrix type:{}'.format(len(tok2id), embeddings.shape))
  return embeddings, tok2id
      
def to_raw(sents, file):
  with codecs.open(file, 'w') as fo:
    for sent in sents:
      fo.write((" ".join(sent)).encode('utf-8')+'\n')

def list_to_bert(sents, bert_config_file, vocab_file, init_checkpoint, bert_file, layer, 
                  max_seq=256, batch_size=8, emb=None, tok2id=None):
  output_file = bert_file
  bert_layer = layer
  max_seq_length = max_seq
  args = list2bert.Args(vocab_file, bert_config_file, init_checkpoint, output_file, max_seq_length, bert_layer)
  args.batch_size = batch_size

  list2bert.list2bert(sents, args, emb=emb, tok2id=tok2id)

def merge(bert_file, merge_file, sents, merge_type='sum'):
  n = 0
  n_unk = 0
  n_tok = 0
  fo = codecs.open(merge_file, 'w')
  with codecs.open(bert_file, 'r') as fin:
    line = fin.readline()
    while line:
      if n % 100 == 0:
        print ("\r%d" % n, end='')
      bert = json.loads(line)
      tokens = []
      merged = {"linex_index": bert["linex_index"], "features":[]}
      i = 0
      while i < len(bert["features"]):
        item = bert["features"][i]
        if item["token"]=="[CLS]" or item["token"]=="[SEP]":
          merged["features"].append(item)
        elif item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
          tmp_layers = []
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            #merged["features"][-1]["layers"][j]["values"] = list(np.array(layer["values"]) + np.array(item["layers"][j]["values"]))
            # j-th layer
            tmp_layers.append([np.array(layer["values"])])
            tmp_layers[j].append(np.array(item["layers"][j]["values"]))

          item = bert["features"][i+1]
          while item["token"].startswith("##") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
            for j, layer in enumerate(merged["features"][-1]["layers"]):
              # j-th layer
              tmp_layers[j].append(np.array(item["layers"][j]["values"]))
            i += 1
            item = bert["features"][i+1]
          for j, layer in enumerate(merged["features"][-1]["layers"]):
            if merge_type == 'sum':
              merged["features"][-1]["layers"][j]["values"] = list(np.sum(tmp_layers[j], 0))
            if merge_type == 'avg':
              merged["features"][-1]["layers"][j]["values"] = list(np.mean(tmp_layers[j], 0))
            if merge_type == 'first':
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][0])
            if merge_type == 'last':
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][-1])
            if merge_type == 'mid':
              mid = int(len(tmp_layers[j]) / 2)
              merged["features"][-1]["layers"][j]["values"] = list(tmp_layers[j][mid])
          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        elif item["token"] == "[UNK]":
          n_unk += 1
          merged["features"].append(item)
          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        else:
          merged["features"].append(item)
        i += 1
      try:
        assert len(merged["features"]) == len(sents[n]) + 2
      except:
        orig = [m["token"] for m in merged["features"]]
        print ('\n',len(merged["features"]), len(sents[n]))
        print (sents[n], '\n', orig)
        print (zip(sents[n], orig[1:-1]))
        raise ValueError("Sentence-{}:{}".format(n, ' '.join(sents[n])))
      for i in range(len(sents[n])):
        try:
          assert sents[n][i].lower() == merged["features"][i+1]["token"]
        except:
          print ('wrong word id:{}, word:{}'.format(i, sents[n][i]))

      n_tok += len(sents[n])
      fo.write(json.dumps(merged)+"\n")
      line = fin.readline()
      n += 1
    print ('Total tokens:{}, UNK tokens:{}'.format(n_tok, n_unk))
    info_file=os.path.dirname(merge_file) + '/README.txt'
    print (info_file)
    with open(info_file, 'a') as info:
      info.write('File:{}\nTotal tokens:{}, UNK tokens:{}\n\n'.format(merge_file, n_tok, n_unk))

#if len(sys.argv) < 7:
#  print ("usage:%s [map_model] [bert_model] [layer(-1)] [conllu file] [output bert] [merged bert]" % sys.argv[0])
#  exit(1)

parser = argparse.ArgumentParser(description='CoNLLU to BERT')
parser.add_argument("--config", type=str, default=None, help="bert config")
parser.add_argument("--vocab", type=str, default=None, help="bert vocab")
parser.add_argument("--model", type=str, default=None, help="bert model")
parser.add_argument("conll_file", type=str, default=None, help="input conllu file")
parser.add_argument("bert_file", type=str, default=None, help="orig bert file")
parser.add_argument("merge_file", type=str, default=None, help="merged bert file")
parser.add_argument("--emb_file", type=str, default=None, help="use this embedding to replace bert input embedding")
parser.add_argument("--layer", type=int, default=-1, help="output bert layer")
parser.add_argument("--max_seq", type=int, default=256, help="output bert layer")
parser.add_argument("--batch", type=int, default=8, help="output bert layer")
parser.add_argument("--merge_type", type=str, default=None, help="merge type (sum|avg|first|last|mid)")
args = parser.parse_args()

n = 0
sents = []
for sent in load_conllu(args.conll_file):
  sents.append(sent)
print ("Total {} Sentences".format(len(sents)))

embeddings, tok2id = None, None
if args.emb_file:
  embeddings, tok2id = load_emb(args.emb_file)

list_to_bert(sents, args.config, args.vocab, args.model, args.bert_file, args.layer, 
              max_seq=args.max_seq, batch_size=args.batch, emb=embeddings, tok2id=tok2id)
merge(args.bert_file, args.merge_file, sents, merge_type=args.merge_type)
