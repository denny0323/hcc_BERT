'''
    0. Settings
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
@tf.function(experimental_relax_shapes=True)
def train_on_batch(model, tensors):

    with tf.GradientTape() as tape:
        output = model(tensors, training=True)
        loss = output.loss
        logits = output.logits

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    y_pred = tf.squeeze(tf.cast(tf.sigmoid(logits) > 0.5, dtype=tf.int32))

    train_loss_tracker.update_state(loss)
    train_acc_metric.update_state(tensors['labels'], y_pred)
    train_microf1_metric.update_state(tf.reshape(tensors['labels'], (-1, 1)), tf.sigmoid(logits))
    train_macrof1_metric.update_state(tf.reshape(tensors['labels'], (-1, 1)), tf.sigmoid(logits))

    return loss, logits, {'train_loss': train_loss_tracker.result(), 'train_acc': train_acc_metric.result(),
                          'train_microf1': train_microf1_metric.result(), 'train_macrof1': train_macrof1_metric.result()}


@tf.function
def distributed_train_step(model, tensors):
    per_replica_losses, logits, train_dict = strategy.run(train_on_batch, args=(model, tensors,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), logits, train_dict


@tf.function(experimental_relax_shapes=True)
def valid_on_batch(model, tensor_val):
    val_output = model(tensor_val, training=False)
    val_loss = val_output.loss
    val_logits = val_output.logits

    y_pred_val = tf.squeeze(tf.cast(tf.sigmoid(val_logits) > 0.5, dtype=tf.int32))

    val_loss_tracker.update_state(val_loss)
    val_acc_metric.update_state(tensor_val['labels'], y_pred_val)
    val_microf1_metric.update_state(tf.reshape(tensorf_val['labels'], (-1, 1)), tf.sigmoid(val_logits))
    val_macrof1_metric.update_state(tf.reshape(tensorf_val['labels'], (-1, 1)), tf.sigmoid(val_logits))

    return {'val_loss': val_loss_tracker.result(), 'val_acc': val_acc_metric.result(),
            'val_microf1': val_microf1_metric.result(), 'val_macrof1': val_macrof1_metric.result()}

@tf.function
def distributed_test_step(model, tensors):
    return strategy.run(valid_on_batch, args=(model, tensors))


epochs = 5
metric_names = ['train_loss', 'train_acc', 'val_loss']

for epoch in range(1, epochs+1):

    print("\nEpoch {}/{}".format(epoch, epochs))
    progBar = Progbar(len(tokenized_train_dataset_tf), stateful_metrics=metric_names, interval=0.3)

    for step, tensors in enumerate(tokenized_train_dataset_tf):

        callbacks.on_batch_begin(step, logs=logs)
        callbacks.on_train_batch_begin(step, logs=logs)

        ## Train Loss & Back Prop
        train_loss, logits, train_dict = distributed_train_step(model, tensors)

        ## Tracking Train metrics
        loss["train_loss"] = train_dict["train_loss"]

        callbacks.on_train_batch_end(step, logs=logs)
        callbacks.on_batch_end(step, logs=logs)

        values = [('train_loss', train_dict['train_loss']), ('train_acc', train_dict['train_acc']),
                 ('train_macrof1', train_dict['train_macrof1']), ('train_microf1', train_dict['train_micref1'])]
        
        progBar.update(step+1, values=values)

    train_acc_metric.reset_states()
    train_microf1_metric.reset_states()
    train_macrof1_metric.reset_states()


    for step_val, tensors_val in enumerate(tokenized_valid_datasets_tf):

        callbacks.on_batch_begin(step_val, logs=logs)
        callbacks.on_test_batch_begin(step_val, logs=logs)

        ## Valid Loss & Back Prop
        valid_dict = distributed_test_step(model, tensors_val)

        callbacks.on_test_batch_end(step_val, logs=logs)
        callbacks.on_batch_end(step_val, logs=logs)

    values = [('train_loss', train_loss), ('train_acc', train_dict['train_acc']), 
              ('val_loss', valid_dict['val_loss']), ('val_acc', valid_dict['val_acc']),
              ('val_macrof1', valid_dict['val_macrof1']), ('val_microf1', valid_dict['val_microf1'])]

    logs['val_loss'] = val_loss_tracker.result()

    val_acc_metric.reset_states()
    val_microf1_metric.reset_states()
    val_macrof1_metric.reset_states()

    callbacks.on_epoch_end(epoch, logs=logs)
    progBar.update(step+1, values=values, finalize=True)

    callbacks.on_train_end(logs=logs)


'''
    6. Evaluate
'''

test_set = Dataset.from_pandas(test)
tokenized_test_dataset = test_set.map(
    tokenize_fn, batched=True, num_proc=8, remove_columns=test_set.features
)

tokenized_test_dataset_tf = tokenized_test_dataset.to_tf_dataset(columns=list(tokenized_test_dataset.features.keys())
                                                                 shuffle=True,
                                                                 batch_size=1024,
                                                                 collate_fn=data_collator,
                                                                 drop_remainder=False,
                                                                 prefetch=1)

from tqdm import tqdm_notebook as tqdm
from tensorflow.keras.metrics import AUC

import warnings
warnings.filterwarnings('ignore')

AUROC = AUC(curve='ROC')
AUPRC = AUC(curve='PR')

test_labels = []
pred_probas = []
predicts = []
for test_tensors in tqdm(tokenized_test_dataset_tf, position=0, leave=True):
    test_output = model(test_tensors, training=False)
    test_loss, test_logits = test_output.loss, test_output.logits

    y_pred = tf.greater(tf.sigmoid(tesg_logits), 0.5)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_test = test_tensors['labels']

    AUROC.update_state(y_test, y_pred)
    AUPRC.update_state(y_test, y_pred)

    test_labels.append(y_test)
    pred_probas.append(tf.sigmoid(test_logits))
    predicts.append(y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/nhn-nanum/NanumGothic.ttf'
font = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font)

titile_classlabels = ['class_1']

def print_confusion_matrix(confusion_matrix, axes,
                           class_label, class_name, fontsize=10):
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    group_percentages = ["{0:.3f}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    labels = [f"{v1}\n\n({v2})" for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    heatmap = sns.heatmap(confusion_matrix, 
                          annot=labels, fmt="", cmap="Blues", linewidths=.2,
                          cbar=False, ax=axes, annot_kws={"size":12})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    axes.set_ylabel('True')
    axes.set_xlabel('Predicted')
    axes.set_title(class_label)


test_labels_no_last = np.hstack(test_labels[:-1])
test_labels_last = test_labels[-1]
test_labels = tf.concat([test_labels_no_last, test_labels_last], axis=0)

predicts = np.vstack(np.asarray(predicts)).squeeze()
pred_probas = np.vstack(np.asarray(pred_probas)).squeeze()


from sklearn.metrics import confusion_matrix

cfm = confusion_matrix(test_labels, predicts)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
print_confusion_matrix(cfm, ax, "Confusion Matrix for class_1", ["0", "1"])
