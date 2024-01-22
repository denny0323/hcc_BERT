'''
  0. Settings 
'''
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



'''
    1.  Pre-processing
'''


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



from datasets import Dataset

train_set = Dataset.from_pandas(train)
test_set  = Dataset.from_pandas(test)


import datasets
datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING /= 10

def set_tempdir():
  return '/myLocalWorkSpace'

from datasets import table
table.tempfile._get_default_tempdir = set_tempdir
table.tempfile.tempdir = '/myLocalWorkSpace'


tokenized_train_dataset = train_set.map(tokenized_fn, num_proc = 12, remove_columns=['paragraph', 'col1', 'col2']) # 우회된 경로에 .arrow 임시파일 생성, remove_column의 이름은 임시로 작성
tokenized_test_dataset  = test_set.map(tokenized_fn, num_proc = 12, remove_columns=['paragraph', 'col1', 'col2'])
del train_set, test_set

gc.collect()


from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors='tf')

tf_tokenized_train_dataset = tokenized_train_dataset.to_tf_dataset(columns=list(tokenized_train_dataset.features.keys()),
                                                                    shuffle=False,
                                                                    batch_size=32,
                                                                    collate_fn=data_collator,
                                                                    drop_remainder=False)

tf_tokenized_test_dataset = tokenized_test_dataset.to_tf_dataset(columns=list(tokenized_test_dataset.features.keys()),
                                                                    shuffle=False,
                                                                    batch_size=1,
                                                                    collate_fn=data_collator,
                                                                    drop_remainder=False)
del tokenized_train_dataset, tokenized_test_dataset
                                                                 
                                                                 
                                                                 
                                                                 
'''
    2.  Build model : BERT + LSTM 
'''

from transformers import TFBertModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM


class TFBertLSTM(Model):
  def __init__(self, pretrained_bert, training=False):
    super(TFBertLSTM, self).__init__()
    self.training = training
    
    self.bert = TFBertModel.from_pretrained(modelpath + pretrained_bert, name='bert')
    self.LSTM = LSTM(self.bert.config.hidden_size, return_sequences=True, name='lstm')
    self.dense = TimeDistributed(Dense(self.bert.config.vocab_size), name='dense')
    
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE 
    )
  
  def call(self, inputs):
    bert_output = self.bert(input_ids = inputs['input_ids'],
                            attention_mask = inputs['attention_mask'],
                            token_type_ids = inputs['token_type_ids'],
                            training=self.training)[0] # last_hidden_state
    lstm_output = self.LSTM(bert_output) # return_sequence=True : 512 sequence output
    logits = self.dense(lstm_output)
    
    if 'labels' in list(inputs.keys()):
      masked_lm_active_loss = tf.not_equal(inputs['labels'], -100)
      
      masked_lm_labels = tf.boolean_mask(inputs['labels'], mask=masked_lm_active_loss)
      masked_lm_reduced_logits = tf.boolean_mask(logits, mask=masked_lm_active_loss)
      
      masked_lm_loss = self.loss_fn(masked_lm_labels, masked_lm_reduced_logits)
      masked_lm_loss = tf.reduce_mean(masked_lm_loss)
      
      return logits, masked_lm_loss
    return logits
  

'''
    3.  Training
'''

model_name = '/pretrained_model/model.h5'
model = TFBertLSTM(model_name)


from transformers import AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=1e-5)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


import time
step_start = time.time()

epochs = 40
print('+---------+---------------------+----------------------+')
for epoch in range(epochs):
  step = 1
  for inputs in tf_tokenized_train_dataset:
    loss_total = 0
    
    with tf.GradientTape() as tape:
      logits, loss = model(inputs, training=True)
      
    trainable_variables = [var for va in model.trainable_variables if var.name not in ['bert/bert/pooler/dense/kernel:0', 'bert/bert/pooler/dense/bias:0']]
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    loss_total += loss.numpy().mean()
    
    if (step % 4000) == 0:
      step_end = time.time()
      print('| Epoch {} | Step {:>5}: {:.5f} | step_time: {:>8.3f} s|'.format(epoch+1, step, loss_total, step_end-step_start))
      step_start = step_end
      
    step += 1
  print('+---------+---------------------+----------------------+')
  
  
model.bert.save_pretrained(modelpath + '/TFBertLSTM_bert_ep{}'.format(epochs))
model.save_weights(modelpath + '/TFBertLSTM_ep{}'.format(epochs))




'''
    4. Evaluation 
'''
from IPython.display import display

log_interval = 5000
correct = p_correct = 0
for i, x in enumerate(tf_tokenized_test_dataset):
  logits, _ = model(x, training=False)
  masks = tf.equal(x['input_ids'], 4)
  
  labels = tf.boolean_mask(x['labels'], mask=masks)
  logits = tf.boolean_mask(logits, mask=masks)
  
  predicted_ids = tf.argmax(logits, 1).numpy()
  predicted_seq = [tokenizer.convert_ids_to_tokens(int(ids)) for ids in predicted_ids]
  labels_seq = [tokenizer.convert_ids_to_tokens(int(label)) for label in labels]
  
  correct   += sum([1/len(labels_seq) if labels_seq[k] == predicted_seq[k] else 0 for k in range(len(labels_seq))])
  p_correct += sum([1/len(labels_seq) if predicted_seq[k].split("_")[0] in labels_seq[k] else 0 for k in range(len(labels_seq))])
  
  
  if i % log_interval == 0:
    print('[{:>2}/{:>2}]'.format((i//log_interval)+1, (len(tf_tokenized_test_dataset)//log_interval)+1))
    print('test_seq', " ".join(tokenizer.convert_ids_to_tokens([int(ids) for ids in x['input_ids'][0] if int(ids) not in [tokenizer.cls_token_id,
                                                                                                                         tokenizer.pad_token_id,
                                                                                                                         tokenizer.sep_token_id,
                                                                                                                         tokenizer.mask_token_id]])))
    display(pd.concat([pd.DataFrame(labels_seq, columns=['answer']).transpose(),
                       pd.DataFrame(predicted_seq, columns=['answer']).transpose()]))
    
    
print('Accuracy :', round(correct)/len(tf_tokenized_test_datset), 4))
print('p_Accuracy :', round(p_correct)/len(tf_tokenized_test_datset), 4))
                                                            
