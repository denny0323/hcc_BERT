'''
    0. Settings
'''
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

'''
    1. Make Datasets
'''

import sys
sys.path.append('../utils/')

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

TRAIN_BATCH_SIZS = 32
MAX_EPOCH = 100
LEARNING_RATE = 3e-4

CHUNK_SIZE = MAX_LENGTH-2

tokenizer = BertTokenizer.from_pretrained("tokenizer_path/vocab.txt",
                                           padding='max_length', truncation=True,
                                           max_len=MAX_LENGTH, do_lower_case=False, strip_accents=False)

def tokenizer_fn(examples):
    examples['input_ids'] = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(evnt))
        for evnt in examples['evnt']
    ]
    return examples

tokenized_train_dataset = train_dataset.map(tokenizer_fn, batched=True, num_proc=4,
                                            remove_columns=['col1', '__index_level_0__', 'evnt'])
tokenized_valid_dataset = valid_dataset.map(tokenizer_fn, batched=True, num_proc=4,
                                            remove_columns=['col1', '__index_level_0__', 'evnt'])

# 저장해두고 재사용
tokenized_train_dataset.save_to_disk('save_path/tokenized_train_dataset')
tokenized_valid_dataset.save_to_disk('save_path/tokenized_valid_dataset')


# load tokenized_dataset
from datasets import load_from_disk
tokenized_train_dataset = load_from_disk('save_path/tokenized_train_dataset')
tokenized_valid_dataset = load_from_disk('save_path/tokenized_valid_dataset')


def group_texts(examples):
    from collections import defaultdict

    concatenated_examples = {
        k: [example[1:-1] for example in exmaples[k]][0] # cls, sep token 제외
    }

    result = defaultdict(list)
    for k, item in concatenated_examples.items(): # item : [[], []]
        item = item[::-1] # 왼쪽부터 truncation할 수 있도록 뒤집기
        total_length = len(item)

        for i in range(0, total_length, CHUNK_SIZE):
            chunk = item[i:i+CHUNK_SIZE][::-1] # re-flip. 뒤집은 순서 다시 원복
            input_ids = tokenizer.build_inputs_with_special_tokens(chunk)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(chunk)
            paded = tokenizer.pad({'input_ids': input_ids, 'token_type_ids': token_type_ids},
                                  padding='max_length', max_length=MAX_LENGTH)

            result['input_ids'].append(paded['input_ids'])
            result['attention_mask'].append(paded['attention_mask'])
            result['token_type_ids'].append(paded['token_type_ids'])

    result['labels'] = result['input_ids'].copy()
    return result


lm_datasets = tokenized_train_dataset.map(group_texts, batched=True, num_proc=4,
                                          remove_columns=tokenized_train_dataset.features)
lm_datasets_valid = tokenized_valid_dataset.map(group_texts, batched=True, num_proc=4,
                                          remove_columns=tokenized_train_dataset.features)

from transformers import DataCollatorForWholeWordMask

data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                             mlm=True, mlm_probability=MLM_PROB, return_tensors='pt')




'''
    2. Build the Model
'''

Model_size = 'Small'

if Model_size == 'Tiny':
    NUM_HEADS = 4
    HIDDEN_SIZE = 128
    NUM_HIDDEN_LAYERS = 2
    INTERMEDIATE_SIZE = 512
elif Model_size == 'Mini':
    NUM_HEADS = 4
    HIDDEN_SIZE = 256
    NUM_HIDDEN_LAYERS = 4
    INTERMEDIATE_SIZE = 1024
elif Model_size == 'Small':
    NUM_HEADS = 8
    HIDDEN_SIZE = 512
    NUM_HIDDEN_LAYERS = 4
    INTERMEDIATE_SIZE = 2048
elif Model_size == 'Tiny':
    NUM_HEADS = 12
    HIDDEN_SIZE = 768
    NUM_HIDDEN_LAYERS = 12
    INTERMEDIATE_SIZE = 3072


config = BertConfig(
    architectures="BertForMaskedLM",
    model_type="bert",
    gradient_checkpointing_enable=False,
    vocab_size=tokenizer.vocab_size,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    num_attention_heads=NUM_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=256,
    type_vocab_size=2,
    pad_token_id=1,
    use_cache=True
)

model = BertForMaskedLM(config)



'''
    3. Train
'''
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='/BERT_Small_PT',
    num_train_epochs=MAX_EPOCH,
    per_device_train_batch_size=int(TRAIN_BATCH_SIZS/torch.cuda.device_count()),
    per_device_eval_batch_size=4,
    warmup_steps=10,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir='./log',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    fp16=True,
    fp16_opt_level='O2',
    run_name='llm-small-all',
    seed=42,
    eval_accumulation_steps=10, # 지정하지 않으면 main gpu에 eval data가 모두 올라가서 CUDA OOM발생
    load_best_model_at_end=True
)

model = model.to(device)
model = model.train()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels.flatten(). preds.flatten())
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
    eval_dataset=lm_datasets_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
