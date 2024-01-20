'''
    0. Settings
'''

import sys

import keras.losses

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


'''
   0.1  Load & Check dataset
'''
online_train = hc.sql('''
    select *
    from {db_name}{finetune_tbl_name}
    where date <= '2023-00-00'
''')

online_train.limit(10).toPandas() # check data
online_train.groupby(['y']).agg(F.count('col1').alias('cnt')).toPandas() # y label check
online_train.groupby(['y']).agg(F.countDistinct('col1').alias('cnt')).toPandas() # y label check

online_pred = hc.sql('''
    select *
    from {db_name}{predict_tbl_name}
''')
online_pred.groupby(['y']).agg(F.count('col1').alias('cnt')).toPandas()

import os, gc

import numpy as np
import tensorflow as tf

from datasets import Dataset
from transformers import AutoTokenizer, BertTokenizer

use_gpu_num = [0, 1, 2]
n_cores = 20

os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NCCL_DEBUG"] = "INFO"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MultiWorkerMirroredStrategy()




'''
   1.  Load & Check dataset
'''

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

def tokenize_fn(examples, max_len=256):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, max_len=max_len)
    result = tokenizer([
            " ".join(evnt.split(" ")[-max_len:])
            for evnt in examples['x']
        ], padding= 'max_length', truncation=True, max_length=max_len
    )

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
del train_dataset

tokenized_valid_dataset = train_dataset.map(
    tokenize_fn, batched=True, num_proc=8, remove_columns=valid_dataset.features
)
del valid_dataset


# save tokenized datasets
tokenized_train_dataset.save_to_disk('dataset_path/tokenized_train_dataset')
tokenized_valid_dataset.save_to_disk('dataset_path/tokenized_valid_dataset')

# load tokenized datasets
# from datasets import load_from_disk
# tokenized_train_dataset = load_from_disk('dataset_path/tokenized_train_dataset')
# tokenized_valid_dataset = load_from_disk('dataset_path/tokenized_train_dataset')


from transformers import DefaultDataCollator

BATCH_SIZE = 128
BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

data_collator = DefaultDataCollator(return_tensors='tf')

tokenized_train_dataset_tf = tokenized_train_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'token_type_ids', 'labels'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    drop_remainder=True,
    prefetch=1
)

tokenized_valid_dataset_tf = tokenized_valid_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'token_type_ids', 'labels'],
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    drop_remainder=True,
    prefetch=1
)


'''
    3. Build Model
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import Progbar

from transformers import TFBertModel
from transformers.modeling_tf_utils import get_initializer
from transformers.models.bert.modeling_tf_bert import TFSequenceClassifierOutput

class TFBertForMultiLabelClassification(Model):
    def __init__(self, pretrained_model_root, pretrained_model_name, num_labels, training=True, *inputs, **kwargs):
        super(TFBertForMultiLabelClassification, self).__init__(*inputs, **kwargs)
        self.training = training
        self.pretrained_model_root = pretrained_model_root
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = TFBertModel.from_pretrained(self.pretrained_model_root+self.pretrained_model_name)

        self.num_labels = num_labels
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.SUM)
        self.classifier = Dense(self.num_labels,
                                kernel_initializer=get_initializer(self.pretrained_model.config.initializer_range)
                                name='classifier')
        self.dropout = Dropout(self.pretrained_model.config.hidden_dropout_prob)

    def call(self, inputs, traininig):
        labels = inputs['labels']

        outputs = self.pretrained_model.bert(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            attention_mask=inputs['attention_mask'], training=self.training
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=self.training)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(y_true=label, y_pred=logits)

        return TFSequenceClassifierOutput(loss=loss,
                                          logits=logits,
                                          hidden_states=outputs.hidden_states,
                                          attentions=outputs.attentions)



COMMON_PATH = '{/path}'

PT_MODEL_PATH = COMMON_PATH + 'PT_ckpt/'
PT_MODEL_NAME = 'BERTMini'

def build_ft_model(pretrained_model_root, pretrained_model_name):
    return TFBertForMultiLabelClassification(pretrained_model_root, pretrained_model_name, num_labels=1)

with strategy.scope():
    model = build_ft_model(PT_MODEL_PATH, PT_MODEL_NAME)



'''
    4. Prepare Metrics
'''

import tensorflow_addons as tfa

train_loss_tracker = tf.keras.metrics.Mean()
val_loss_tracker = tf.keras.metrics.Mean()

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
train_microF1_metric = tfa.metrics.F1Score(num_classes=model.num_labels, average='micro')
train_macroF1_metric = tfa.metrics.F1Score(num_classes=model.num_labels, average='macro')

val_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_microF1_metric = tfa.metrics.F1Score(num_classes=model.num_labels, average='micro')
val_macroF1_metric = tfa.metrics.F1Score(num_classes=model.num_labels, average='macro')

with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath='finetune_model_root/model_name'
                                                      monitor='val_loss',
                                                      save_weights_only=True,
                                                      save_best_only=True,
                                                      mode="auto",
                                                      save_freq='epoch',
                                                      verbose=1)

    callbacks = tf.keras.callbacks.CallbackList(cp_callbacks, add_history=True, model=model)

    logs = {}
    callbacks.on_train_begin(logs=logs)



'''
    5. Train
'''

