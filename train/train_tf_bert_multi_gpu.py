############ 0. settings ############


import os
import time
import functools

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import Dataset
from transformers import AutoTokenizer, PretrainedTokenizerFast
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

use_gpu_num = [2, 3] # [2, 3, 4, 5]

lr = 1e-4

n_cores = 20

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(num) for num in use_gpu_num])
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
    
strategy = tf.distribute.MultiWorkerMirroredStrategy()


############ 1. input & pre-processing ############


tokenizer = AutoTokenizer.from_pretrained("tokenizers/path...", max_len=256)

## skeleton code
df = sql_as_pandas_with_pyspark(""" sql query """)


train, test = train_test_split(df.paragraph, test_size=.2)
del df

train_set = Dataset.from_pandas(pd.DataFrame(train))
test_set  = Dataset.from_pandas(pd.DataFrame(test))
del train, test


def tokenize_fn(data):
  result = tokenizer(data['paragraph'], padding='max_length', truncation=True, max_length=256)
  result['labels'] = result['input_ids'].copy()
  return result
  
def set_tempdir():       # 다음 코드에서 huggingface의 arrowdataset에 map method를 사용할 때, tmp directory에 임시파일을 생성
  return '/myLocalWorkSpace'  # 해당 공간의 용량이 제한되어, local로 우회하기 위해 directory를 변경해주는 임시 함수 작성
  
from datasets import table
table.tempfile._get_default_tempdir = set_tempdir
table.tempfile.tempdir = '/myLocalWorkSpace'


tokenized_train_dataset = train_set.map(tokenized_fn, num_proc = 12, remove_columns=['paragraph', 'col1', 'col2']) # 우회된 경로에 .arrow 임시파일 생성, remove_column의 이름은 임시로 작성
tokenized_test_dataset  = test_set.map(tokenized_fn, num_proc = 12, remove_columns=['paragraph', 'col1', 'col2'])
del train_set, test_set

tokenizer = PretrainedTokenizerFast(
  tokenizer_file = 'tokenizers/path/.../pretrained_tokenizer.json',
  bos_token="[CLS]",
  eos_token="[SEP]",
  unk_token="[UNK]",
  cls_token="[CLS]",
  sep_token="[SEP]",
  pad_token="[PAD]",
  mask_token="[MASK]",
  model_max_length=256
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.1)

BATCH_SIZE = 32
GLOBAL_BATCH_SIZE = strategy.num_replicas_in_sync * BATCH_SIZE

tf_tokenized_train_dataset = tokenized_train_dataset.to_tf_dataset(columns=list(tokenized_train_dataset.features.keys()),
                                                                    shuffle=True,
                                                                    batch_size=GLOBAL_BATCH_SIZE,
                                                                    collate_fn=data_collator,
                                                                    drop_remainder=False,
                                                                    prefetch=1)
                                                                    
tf_tokenized_test_dataset = tokenized_test_dataset.to_tf_dataset(columns=list(tokenized_test_dataset.features.keys()),
                                                                    shuffle=False,
                                                                    batch_size=GLOBAL_BATCH_SIZE,
                                                                    collate_fn=data_collator,
                                                                    drop_remainder=False,
                                                                    prefetch=1)
del tokenized_train_dataset, tokenized_test_dataset


dist_mlm_train_dataset = strategy.experimental_distribute_dataset(tf_tokenized_train_dataset)
dist_mlm_test_dataset = strategy.experimental_distribute_dataset(tf_tokenized_test_dataset)


############ 2. Bulid the Model ############

from transformers import BertConfig, TFBertForMaskedLM

config = BertConfig(
      architectures = "BertForMaskedLM",
      model_type = "bert",
      gradient_checkpointing_enable = False,
      vocab_size = tokenizer.vocab_size,
      hidden_size = 512,
      num_hidden_layers = 4,
      num_attention_heads = 4,
      initializer_range = 0.02,
      intermediate_size = 512,
      hidden_act = "gelu",
      hidden_dropout_prob = 0.1
      attention_probs_dropout_prob = 0.1,
      max_position_embeddings = 256,
      type_vocab_size =2,
      pad_token_id = tokenizer.pad_token_id,
      use_cache = False
)

def create_model(config, lr):
  with strategy.scope():
    model = TFBertForMaskedLM.from_config(config)
    optimizer = transformers.AdamWeightDecay(learning_rate=lr, weight_decay_rate=0.01)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
  return model, optimizer
  
model, optimizer = create_model(config, lr)

with strategy.scope():
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
  )
  
  def compute_loss(labels, logits):  # from huggingface source code
    masked_lm_active_loss = tf.not_equal(labels, -100)
    masked_lm_reduced_logits = tf.boolean_mask(tensor=logits,
                                               mask=masked_lm_active_loss)
    masked_lm_labels = tf.boolean_mask(tensor=labels,
                                       mask=masked_lm_active_loss)
    masked_lm_loss = loss_fn(y_true=masked_lm_labels,
                             y_pred=masked_lm_reduced_logits)
                             
    return tf.nn.compute_average_loss(masked_lm_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    
    
    
with strategy.scope():
  train_metric = tf.keras.metrics.Mean(name='train_loss')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  
  
  
#### training loop (ref: Tensorflow Tutorial)

#@tf.function
#def distributed_train_step(data):
#  strategy.run(train_step, args=(data,))
  
@tf.function
def distributed_train_step(data):
  per_replica_losses = strategy.run(train_step, args=(data,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
  
def train_step(inputs):
  with tf.GradientTape() as tape:
    output = model.bert(inputs, training=True)
    sequence_output = output[0]
    logits = model.mlm(sequence_output=sequence_output, training=True)
    loss = compute_loss(inputs['labels'], logits)
    
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
  
  
@tf.function
def distributed_test_step(data):
  return strategy.run(test_step, args=(data,))
  
def test_step(inputs):
  output_test = model(inputs, training=False)
  loss_test = output_test.loss
  
  test_loss.update_state(loss_test)
  
  

NUM_EPOCHS = 5
STEPS_PER_EPOCH = len(tf_tokenized_train_dataset)
EVALUATE_EVERY = 500

start = time.time()
for epoch in range(1, epochs+1):
  total_loss = 0.0
  step = 0

  for tensor in dist_mlm_train_dataset:
    total_loss += distributed_train_step(tensor)
    step += 1
    train_loss = total_loss / step

    if (step % EVALUATE_EVERY == 0) or (step == STEPS_PER_EPOCH):
      metric = train_metric.result()
      end = time.time()
      print("Epoch [{:>2}/{:>2}] - [{:>6}/{:>6}] - {:.2f}s - train_loss : {:.4f} ".format(
                                              epoch, epochs, step, steps_per_epoch, end-start, train_loss))
      start = end

    gc.collect()

    for test_tensors in test_dist_dataset:
      distributed_test_step(test_tensors)

    template = ("Epoch {} | Loss : {:.4f} | Test Loss : {:.4f}")
    print(template.format(epoch, train_loss, test_loss.result()))

    test_loss.reset_states()
      
      

        



  
                                                                    
  
