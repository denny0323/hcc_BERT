### 0. settings ###
import tensorflow as tf
import pandas as pd
import numpy as np

import transformers
import torch
import functools
import os
import gc
import unicodedata

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_cores = 20
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)

os.environ["NCCL_DEBUG"] = "INFO"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:
    print(e)

    
    
path = '../WorkDir'
modelpath = path + '/model_checkpoint'

import sys
sys.path.append('..')

from hyspark import Hyspark

hs = Hyspark('{}_dataloading_{}'.format(employee_id, curr_time), mem_per_core=2)
hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc) ## custom function, result True

query = hc.sql(""" QUERY """)
df = df_as_pandas_with_pyspark(query) ## custom function
hs.stop()

#### 1. preprocessing ###

Last_N = 5
df.paragraph = df.paragraph.apply(lambda x: x.split(" ")[:-Last_N])
df.labels = df.paragraph.apply(lambda x: x.split(" ")[-Last_N:])

# df.to_parquet(df_for_LSTM.parquet)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretraining(path+'/tokenizers/tokenizer_path')

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token




from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.2, random_state=42)


def tokenize_fn(data):
  encoded_token_labels = []
  for token_label in data['labels']:
    encoded_token_labels += tokenizer.tokenize(token_label)
    
  data['paragraph'] += [tokenizer.mask_token] * len(encoded_token_labels)
  
  result = tokenizer(" ".join(data['paragraph']), max_length=256, padding='max_length', truncation=True)
  ## pad token : 1, sep token : 3, mask token : 4을 제외한 input길이 만큼 -100을 넣고(masking value) 나머지 길이를 채움
  result['labels'] = [-100] * (len([xx for xx in result['input_ids'] if xx not in [1, 3, 4]])) + tokenizer.convert_tokens_to_ids(encoded_token_labels) + [tokenizer.sep_token_id]  
  
  if len(result['labels']) > 256:
    result['labels'] = result['labels'][-256:]
  else:
    result['labels'] += [-100]*(256-len(result['labels']))
  
  return result
    
