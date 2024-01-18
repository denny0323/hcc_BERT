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


# 예제로 대체 - 데이터 구조 및 스키마에 따라 다름
df_dict = {
  'evnt_cd1' : 'col1',
  'evnt_cd2' : ['col1', 'col2'],
  'evnt_cd3' : 'col9',
  'evnt_cd4' : 'col11',
  ...
  'evnt_cd14' : ['col12', 'col15'],
}


import pyspark.sql.functions as F

from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import udf





















