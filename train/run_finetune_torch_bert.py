'''
    0. Settings
'''
import os, gc
import torch
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer

use_gpu_num = [0, 1, 2, 3]
n_cores = 20

os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NCCL_DEBUG"] = "INFO"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




'''
   1.  Make dataset
'''
import sys
sys.path.append('../utils/')

from spark_hive_utils import *
from hyspark import Hyspark
from datetime import datetime

import pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.window import *

from functools import reduce
from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.sql.functions import udf

import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_columns = 1e5
pd.options.display.max_rows = 1e5
pd.options.mode.chained_assignment = None
pd.options.display.float_format = '{:,.4f}'.format


now = datetime.now()
curr_time = now.strftime("%H:%M:%S")
def get_employee_id():
  from os import getcwd
  from re import search
  return str(search(r'.+(\d{6}).?', getcwd()).group(1))

employee_id = get_employee_id()
hs = HySpark(f'{employee_id}_seqdata_tokenizer_LLM_{curr_time}',
             mem_per_core=8, instance='general')

hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc)

ps_df = hc.sql('''
    select *
    from {db_name}{finetune_tbl_name}
    where date <= '2023-00-00'
''')

df = df_as_pandas_with_pyspark(ps_df, double_to_float=True, bigint_to_int=True)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df[['x', 'y']], test_size=.2, random_state=42)
train, valid = train_test_split(df[['x', 'y']], test_size=.1, random_state=42)
del df


train_dataset = Dataset.from_pandas(train)
valid_dataset = Dataset.from_pandas(valid)
del train, valid


TOKENIZER_PATH = '{tokenizer_path}/tokenizer'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, max_len=max_len)

def tokenize_fn(examples, max_len=256):    
    result = tokenizer(examples['evnt'], 
                       padding='max_length', truncation=True, 
                       max_length=max_len)
    result['label'] = examples['y']
    gc.collect()
    return result


'''
    .parquet file이 분석환경 /tmp에 쌓이지 않도록 우회하는 함수
'''
def set_tempdir():
    return '/mydir/tmp'

from datasets import table
table.tempfile._get_default_tempdir = set_tempdir
table.tempfile.tempdir = '/mydir/tmp'

import datasets
datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING /= 10


'''
    2. Make Tokenized Datasets
'''

tokenized_train_dataset = train_dataset.map(
    tokenize_fn, batched=True, num_proc=8, remove_columns=train_dataset.features
)

tokenized_valid_dataset = train_dataset.map(
    tokenize_fn, batched=True, num_proc=8, remove_columns=valid_dataset.features
)
del train_dataset, valid_dataset

tokenized_train_dataset.set_format('torch')
tokenized_valid_dataset.set_format('torch')

# save tokenized datasets
tokenized_train_dataset.save_to_disk('dataset_path/tokenized_train_dataset')
tokenized_valid_dataset.save_to_disk('dataset_path/tokenized_valid_dataset')

# load tokenized datasets
# from datasets import load_from_disk
# tokenized_train_dataset = load_from_disk('dataset_path/tokenized_train_dataset')
# tokenized_valid_dataset = load_from_disk('dataset_path/tokenized_train_dataset')


from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors='pt')


'''
    3. Build Model (torch ver.)
'''
from torch import nn
from torcn.nn import BCEWithLogitsLoss, Dropout, Linear
from ..modeling_bert_custom import SequenceClassifierOutput, BertModelWithCustomEmbeddings. BertConfig


class BertForMultiLabelClassification(nn.Moudle):
    def __init__(self, pretrained_model_root, pretrained_model_name, num_labels, update_config=None, *inputs, **kwargs):
        super(BertForMultiLabelClassification, self).__init__(*inputs, **kwargs)
        
        self.pretrained_model_root = pretrained_model_root
        self.pretrained_model_name = pretrained_model_name
        self.config = BertConfig.from_json_file(self.pretrained_model_root + self.pretrained_model_name + '/config.json')

        if update_config is not None:
            self.config.update(update_config)
            
        self.pretrained_model = BertModelWithCustomEmbeddings.from_pretrained(self.pretrained_model_root+self.pretrained_model_name,
                                                                             from_tf=True,
                                                                             config=self.config)

        self.num_labels = num_labels
        self.loss_fn = BCEWithLogitsLoss(reduction='mean')
        
        self.classifier = Linear(self.pretrained_model.config.hidden_size, self.num_labels))
        self.dropout = Dropout(p=self.pretrained_model.config.hidden_dropout_prob)
        

    def forward(self, inputs, predict=False):
        if not predict:
            labels = inputs["labels"]

        outputs = self.pretrained_model.bert(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            attention_mask=inputs['attention_mask'])

        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output)
        
        logits = self.classifier(pooled_output)

        if not predict:
            loss = self.loss_fn(labels, logits)
        
            return TFSequenceClassifierOutput(loss=loss,
                                              logits=logits,
                                              hidden_states=outputs.hidden_states,
                                              attentions=outputs.attentions)
        else:        
            return TFSequenceClassifierOutput(loss=loss,
                                              hidden_states=outputs.hidden_states,
                                              attentions=outputs.attentions)


update_config = {
    "architecture": "BertModel",
    "custom1" : 0,    # the range of custom embedding 1 value (int) : max of cust. embed. val.
    "custom2" : 0,
    "custom3" : 0,
    "custom4" : 0,
    "custom5" : 0,
    "custom6" : 0,
    ...
    "custom10" : 0,
}


COMMON_PATH = '{/path}'

PT_MODEL_PATH = COMMON_PATH + 'PT_ckpt/'
PT_MODEL_NAME = 'BERTMini'

def build_ft_model(pretrained_model_root, pretrained_model_root, num_labels, update_config):
    return BertForMultiLabelClassification(pretrained_model_root, pretrained_model_name, 
                                           num_labels=1, update_config=update_config)


model = nn.DataParallel(
    build_ft_model(pretrained_model_root, pretrained_model_name, 
                   num_labels=1, update_config=update_config, output_device=4)
)


'''
    5. Train
'''
MAX_LENGTH = 256
MLM_PROB = 0.15

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 64
MAX_EPOCH = 20
LEARNING_RATE = 3e-4


from transfromers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='/BERT_Small_FT',
    num_train_epochs=MAX_EPOCH,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=1e3,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir='./log',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    fp16=True,
    fp16_opt_level='O2',
    run_name='llm-small-all-ft',
    seed=42,
    save_steps=1e6
)


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

model = model.to(device)
model = model.train()


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

'''
TypeError: Caught TypeError in replica 0 on device 0.
Original Traceback (most recent call last):
    File "~/torch/nn/parallel/parallel_apply.py" line 61, in _worker
        output = module(*input, **kwargs)
    FIle "~/torch/nn/modules/module.py" line 1102, in _call_impl
        return forward_call(*output, **kwargs)
TypeError: forward() got an unexpected keyword argument 'labels'
'''
'''
