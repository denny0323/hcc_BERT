import os, gc
import torch
import transformers

import numpy as np
import pandas as pd
from datasets import Dataset

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000

from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

use_gpu_num = [0, 1]

n_cores = 20

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(num) for num in use_gpu_num])
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NCCL_DEBUG"] = "INFO"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'''
- Device: {device}
- Current cuda devices: {torch.cuda.current_devices()}
- # of using devices: {torch.cuda.device_count()}
''')

import sys
sys.path.append('../../')

from datetime import datetime
from hyspark import HySpark
from spark_hive_utils import check_hive_available, df_as_pandas_with_pyspark

hs = Hyspark('{}_dataloading_{}'.format(employee_id, curr_time), mem_per_core=2)
hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc) ## custom function, result True

ps_df = hc.sql(""" QUERY """)
ps_df_grp = ps_df.groupby('column1').agg(
  F.concat_ws(' ', F.collect_list('evnt')).alias('evnt')
).orderBy('column1')

df = df_as_pandas_with_pyspark(ps_df)
hs.stop()

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.2, random_state=42)
train, valid = train_test_split(train, test_size=.1, random_state=42)
del df

train_dataset = Dataset.from_pandas(pd.DataFrame(train))
valid_dataset = Dataset.from_pandas(pd.DataFrame(valid))
del train, valid

MAX_LENGTH = 256
SHORT_SEQ_PROB = 0.1
MLM_PROB = 0.15

TRAN_BATCH_SIZE = 32
MAX_EPOCH = 100
LEARNING_RATE = 3e-4

CHUNK_SIZE = MAX_LENGTH-2

