from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import codecs
import re
import numpy as np
import json
import argparse
import torch
import collections

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

def load_conllu(file):
  with codecs.open(file, 'rb') as f:
    reader = codecs.getreader('utf-8')(f)
    buff = []
    for line in reader:
      line = line.strip().lower()
      if line and not line.startswith('#'):
        if not re.match('[0-9]+[-.][0-9]+', line):
          buff.append(line.split('\t')[1])
      elif buff:
        yield buff
        buff = []
    if buff:
      yield buff
    
def to_raw(sents, file):
  with codecs.open(file, 'w') as fo:
    for sent in sents:
      fo.write(" ".join(sent)+'\n')

def apply_bpe(sents, file, code_file):
  bpe_file = file + '.bpe'
  dict={'main':'/users2/yxwang/work/toolkit/XLM/tools/fastBPE/fast','output':bpe_file,
    'input':file,'codes':code_file}
  print (len(sents))
  to_raw(sents, file)
  
  cmd = '{main} applybpe {output} {input} {codes}'.format(**dict)
  print (cmd)
  os.system(cmd)
  bpe_sents = []
  with open(bpe_file, 'r') as fi:
    for line in fi.read().strip().split('\n'):
      bpe_sents.append(line.strip().split(' '))
  assert len(bpe_sents) == len(sents)
  return bpe_sents

def gen_embed(model_path, sentences, lang):
  reloaded = torch.load(model_path)
  params = AttrDict(reloaded['params'])
  print("Supported languages: %s" % ", ".join(params.lang2id.keys()))
  assert lang in params.lang2id.keys()

  # build dictionary / update parameters,
  dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
  params.n_words = len(dico)
  params.bos_index = dico.index(BOS_WORD)
  params.eos_index = dico.index(EOS_WORD)
  params.pad_index = dico.index(PAD_WORD)
  params.unk_index = dico.index(UNK_WORD)
  params.mask_index = dico.index(MASK_WORD)
    
  # build model / reload weights,
  model = TransformerModel(params, dico, True, True)
  model.load_state_dict(reloaded['model'])

  # list of (sentences, lang)
  #sentences = [
  #    ('the following secon@@ dary charac@@ ters also appear in the nov@@ el .', 'en'),
  #    ('les zones rurales offr@@ ent de petites routes , a deux voies .', 'fr'),
  #    ('luego del cri@@ quet , esta el futbol , el sur@@ f , entre otros .', 'es'),
  #    ('am 18. august 1997 wurde der astero@@ id ( 76@@ 55 ) adam@@ ries nach ihm benannt .', 'de'),
  #    ('此外 ， 松@@ 嫩 平原 上 还有 许多 小 湖泊 ， 当地 俗@@ 称 为 “ 泡@@ 子 ” 。', 'zh')]

  # add </s> sentence delimiters
  sentences = [['</s>']+sent+['</s>'] for sent in sentences]

  bs = len(sentences)
  slen = max([len(sent) for sent in sentences])

  word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
  for i in range(len(sentences)):
    sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
    word_ids[:len(sent), i] = sent

  lengths = torch.LongTensor([len(sent) for sent in sentences])
  langs = torch.LongTensor([params.lang2id[lang] for _ in sentences]).unsqueeze(0).expand(slen, bs)

  # [slen, bs, hidden_dim]
  tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()

  return tensor

def sent2bert(sents, model_path, lang, bert_file, batch_size=16):

  reloaded = torch.load(model_path)
  params = AttrDict(reloaded['params'])
  print("Supported languages: %s" % ", ".join(params.lang2id.keys()))
  assert lang in params.lang2id.keys()

  # build dictionary / update parameters,
  dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
  params.n_words = len(dico)
  params.bos_index = dico.index(BOS_WORD)
  params.eos_index = dico.index(EOS_WORD)
  params.pad_index = dico.index(PAD_WORD)
  params.unk_index = dico.index(UNK_WORD)
  params.mask_index = dico.index(MASK_WORD)
    
  # build model / reload weights,
  model = TransformerModel(params, dico, True, True)
  model.load_state_dict(reloaded['model'])

  model.eval()

  unique_id = 0
  with open(bert_file, "w", encoding='utf-8') as writer:
    for i in range(0, len(sents), batch_size):
      if i + batch_size < len(sents):
        sentences = sents[i:i+batch_size]
      else:
        sentences = sents[i:]
      # add </s> sentence delimiters
      sentences = [['</s>']+sent+['</s>'] for sent in sentences]

      bs = len(sentences)
      slen = max([len(sent) for sent in sentences])

      word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
      for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

      lengths = torch.LongTensor([len(sent) for sent in sentences])
      langs = torch.LongTensor([params.lang2id[lang] for _ in sentences]).unsqueeze(0).expand(slen, bs)

      # [slen, bs, hidden_dim]
      tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
      # [slen, bs, hidden_dim] => [bs, slen, hidden_dim]
      layer = tensor.transpose(0,1).detach().cpu().numpy()

      for b, sent in enumerate(sentences):
        output_json = collections.OrderedDict()
        output_json["linex_index"] = unique_id
        all_out_features = []
        for (t, token) in enumerate(sent):
          all_layers = []
          layer_output = layer[b]
          layers = collections.OrderedDict()
          layers["index"] = -1
          layers["values"] = [
              round(x.item(), 6) for x in layer_output[t]
          ]
          all_layers.append(layers)
          out_features = collections.OrderedDict()
          out_features["token"] = token
          out_features["layers"] = all_layers
          all_out_features.append(out_features)
        output_json["features"] = all_out_features
        writer.write(json.dumps(output_json) + "\n")
        unique_id += 1

def merge(bert_file, merge_file, sents, merge_type='sum'):
  merge_file = merge_file + '.' + merge_type
  n = 0
  n_unk = 0
  n_tok = 0
  fo = codecs.open(merge_file, 'w')
  assert merge_type is not None
  print ("Merge Type: {}".format(merge_type))
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
        if item["token"]=="</s>":
          merged["features"].append(item)
        elif item["token"].endswith("@@") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
          tmp_layers = []
          tmp_layers.append(np.array(item["layers"][0]["values"]))

          item = bert["features"][i+1]
          while item["token"].endswith("@@") and not (len(merged["features"])-1<len(sents[n]) and item["token"] == sents[n][len(merged["features"])-1]):
            tmp_layers.append(np.array(item["layers"][0]["values"]))
            i += 1
            item = bert["features"][i+1]
          # add the first token not end with '@@'
          i += 1
          item = bert["features"][i]
          tmp_layers.append(np.array(item["layers"][0]["values"]))

          # append the first token not end with '@@', then edit its values and token
          merged["features"].append(item)
          if merge_type == 'sum':
            merged["features"][-1]["layers"][0]["values"] = list(np.sum(tmp_layers, 0))
          elif merge_type == 'avg':
            merged["features"][-1]["layers"][0]["values"] = list(np.mean(tmp_layers, 0))
          elif merge_type == 'first':
            merged["features"][-1]["layers"][0]["values"] = list(tmp_layers[0])
          elif merge_type == 'last':
            merged["features"][-1]["layers"][0]["values"] = list(tmp_layers[-1])
          elif merge_type == 'mid':
            mid = int(len(tmp_layers) / 2)
            merged["features"][-1]["layers"][0]["values"] = list(tmp_layers[mid])

          if len(sents[n]) < len(merged["features"]) - 1:
            print (sents[n], len(merged["features"]))
          else:
            merged["features"][-1]["token"] = sents[n][len(merged["features"])-2].lower()
        elif item["token"] == "<unk>":
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

parser = argparse.ArgumentParser(description='CoNLLU to BERT')
parser.add_argument("--codes", type=str, default=None, help="bert vocab")
parser.add_argument("--model", type=str, default=None, help="bert model")
parser.add_argument("--lang", type=str, default=None, help="language")
parser.add_argument("--bpe_file", type=str, default='sent.tmp', help="temporary sent file")
parser.add_argument("conll_file", type=str, default=None, help="input conllu file")
parser.add_argument("bert_file", type=str, default=None, help="orig bert file")
parser.add_argument("merge_file", type=str, default=None, help="merged bert file")
parser.add_argument("--batch", type=int, default=8, help="output bert layer")
parser.add_argument("--merge_type", type=str, default=None, help="merge type (sum|avg|first|last|mid)")
args = parser.parse_args()

n = 0
sents = []
for sent in load_conllu(args.conll_file):
  sents.append(sent)
print ("Total {} Sentences".format(len(sents)))

bpe_sents = apply_bpe(sents, args.bpe_file, args.codes)

sent2bert(bpe_sents, args.model, args.lang, args.bert_file, batch_size=args.batch)

merge_types = args.merge_type.strip().split(',')
for merge_type in merge_types:
  merge(args.bert_file, args.merge_file, sents, merge_type=args.merge_type)
