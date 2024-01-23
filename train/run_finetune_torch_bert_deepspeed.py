'''
        cmd usage
'''
import Dataset

# deepspeed --num_gpus=2 {this_script}.py --deepspeed_config ds_config_zero2.json
# deepspeed --include localhost:0,1, --master_port {} {this script}.py --deepspeed_config ds_config_zero2.json

# 백그라운드 실행
# nohup deepspeed --include localhost:0,1, --master_port {} {this script}.py --deepspeed_config ds_config_zero2.json

'''
        0. Settings
'''

import os, gc
import torch
import transformers
import deepspeed
import datasets

import numpy as np
import pandas as pd

from datasets import Dataset, table
from transformers import AutoTokenizer
from datetime import datetime
from hyspark import Hyspark

import sys
sys.path.append('../utils')
sys.path.append('../model')

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, roc_auc_score
from utils.spark_hive_utils import check_hive_available, df_as_pandas_with_pyspark

import pyspark.sql.functions as F

from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model.modeling_bert_custom import BertForSequenceClassificationWithCustomEmbeddings


'''
        0. 1   functions
'''
def tokenize_fn(examples, max_len=256):
    result = tokenizer(examples['evnt'],
                       padding='max_length', truncation=True,
                       max_length=max_len)
    result['labels'] = np.expand_dims(
        np.array(examples['label'], dtype=np.float), axis=1   # bert output = (batch_size, 1)
    )
    gc.collect()
    return result

def set_tempdir():
    return '/myhome/mydir/mytmp'


def get_employee_id():
  from os import getcwd
  from re import search
  return str(search(r'.+(\d{6}).?', getcwd()).group(1))


def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    predictions = torch.gt(
        torch.from_numpy(eval_preds.predictions), 0.5
    ).long()

    pre, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    return {
        'accuracy' : acc,
        'precision': auc,
        'recall' : rec,
        'f1-score': f1,
        'auroc' : auc
    }
'''    
        0. 2  sys settings
'''
n_cores = 20

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NCCL_DEBUG"] = "INFO"

# tokenized dataset(arrow_datasets) settings
table.tempfile._get_default_tempdir = set_tempdir
table.tempfile.tempdir = 'myhome/mydir/tmp'
datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING /= 10


''' 
        1.  Arguments 
'''
# training arguments
MAX_LENGTH = 256      # Maximum number of tokens in an input sample after padding
MAX_EPOCHS = 5        # Maximum number of epochs to train the model for
LEARNING_RATE = 3e-4  # Learning rate for training the model

TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

# dataset arguments
meta_columns = ['column1', 'column2', 'evnt', 'label']
remove_columns = ['column1', 'column2', 'evnt', 'label', '__index_level_0__']
df_columns = ['column1', 'column2', 'evnt', 'label',
              'cust_id1', 'cust_id2', 'cust_id3', 'cust_id4', 'cust_id5'
              'cust_id6', 'cust_id7', 'cust_id8', 'cust_id9', 'cust_id10']
mapping_column_nms = {
    'cust_bin1' : 'cust_id1',
    'cust_bin2' : 'cust_id2',
    ...
    'cust_bin10' : 'cust_id10',
}

# dataset query
SQL = '''
    SELECT *
    FROM {db_name}.{table_name}
'''

# pretrain root/name
pretrained_model_root = 'myhome/mydir/ckpt/'
pretrained_model_name = 'BERT_small/'

sample_frac = 0.100
sampled = (sample_frac < 1.000) # sample_frac = 1.0, 100% = all data --> sampled = False

'''
        2.  Build a fine-tune Model
'''

model = BertForSequenceClassificationWithCustomEmbeddings(
    config = PretrainedConfig(),
    pretrained_model_root=pretrained_model_root,
    pretrained_model_name=pretrained_model_name,
    num_labels=1
)


'''
        3. Make Dataset
'''
now = datetime.now()
curr_time = now.strftime("%H:%M:%S")
employee_id = get_employee_id()

hs = HySpark(f'{employee_id}_seqdata_tokenizer_LLM_{curr_time}',
             mem_per_core=8, instance='general')
hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc)

ps_df = hc.sql(SQL)
if sampled:
    ps_df = ps_df.sampleBy('label', fractions={0: sample_frac, 1: sample_frac})

# default값이 0인 column들을 1로 맞춰줌 (0 : dummy로 쓰기 위해)
selection = [
    F.when(F.col(column).isNull(), F.col(column))\
     .otherwise(F.col(column) + F.lit(1)).alias(column)
    for column in ps_df.columns if column not in meta_columns
]

ps_df = ps_df.select(meta_columns+selection).fillna(0)

df = df_as_pandas_with_pyspark(ps_df,
                               double_to_float=True,
                               bigint_to_int=True)
hs.stop()

df.rename(columns=mapping_column_nms, inplace=True)

for col in list(df.columns)[3:]:
    df[col] = df[col].astype(np.int32)

train, test = train_test_split(df, test_size=.2, random_state=42)
train, valid = train_test_split(train, test_size=.1, random_state=42)
del df

train_dataset = Dataset.from_pandas(train)
valid_dataset = Dataset.from_pandas(valid)
del train, valid

tokenizer = AutoTokenizer.from_pretrained('/myhome/mydir/tokenizer_path/tokenizer_name/',
                                          max_len=256)

tokenized_train_dataset = train_dataset.map(
    tokenize_fn, batched=True, num_proc=4, remove_columns=remove_columns
)

tokenized_valid_dataset = valid_dataset.map(
    tokenize_fn, batched=True, num_proc=4, remove_columns=remove_columns
)

tokenized_train_dataset.set_format('torch')
tokenized_valid_dataset.set_format('torch')


'''
            4.  Train
'''

training_args = TrainingArguments(
    output_dir='/BERT_Small_FT_ds',
    num_train_epochs=MAX_EPOCHS,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=1000,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    fp16=True,
    fp16_opt_level='O2',
    run_name='llm-small-all-ft',
    load_best_model_at_end=True,
    eval_accumulation_steps=10,
    seed=42,
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model('/myhome/mydir/trainer/llm-small-all-ft')