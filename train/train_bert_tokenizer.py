from datetime import datetime
from hyspark import Hyspark

now= datetime.now()
curr_time = now.strftime('%H:%M:%S')

from spark_hive_utils import *

def get_employee_id():
  from os import getcwd
  from re import search
  return str(search(r'.+(\d{6}).?', getcwd()).group(1))


employee_id = get_employee_id()
hs = HySpark(f'{employee_id}_seqdata_tokenizer_LLM_{curr_time}',
             mem_per_core=8, instance='general')

hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc)


# load data
ps_df = hc.sql('SELECT * FROM db_name.llm_uniq_evnt_parsed_v1')

import pyspark.sql.functions as F
ps_df = ps_df.withColumn('evnt', F.regexp_replace('evnt', '_', ''))

df = ps_df.toPandas()
#df = df_as_pandas_with_pyspark(ps_df, double_to_float=True, bigint_to_int=True)
hs.stop()


from tokenizers import BertWordPieceTokenizer, pre_tokenizers

from tokenizers.normallizers import NFC
from tokenizers.pre_tokenizers import WhitespaceSplit

tokenizer = BertWordPieceTokenizer(strop_accents=False)
tokenizer.normalizer = normallizers.Sequence([NFC()])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit()])

special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']

tokenizer.train_from_iterator(iter(df.evnt), vocab_size = 1e6,
                              limit_alphabet = 1e6,
                              special_tokens=special_tokens, min_frequency=20)


file_path = './tokenizer/BERT_tokenizer_tmp'

import os
tokenizer.save_model(directory=file_path)

from transformers import BertTokenizer
vocab_path = file_path + '/vocab.txt'
file_path2 = './tokenizer/BERT_tokenizer'
os.mkdir(file_path2)

tokenizer = BertTokenizer(vocab_file=vocab_path, strip_accents=False)
tokenizer.save_pretrained(file_path2)
