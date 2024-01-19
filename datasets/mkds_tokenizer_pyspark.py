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


# masking - 사내 데이터 구조 및 스키마
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


df_list = []

START_YRMN = '202201'
END_YRMN = '202306'

for cd, col in df_dict.items():
  print(cd, col)
  if cd in ['evnt_cd3', 'evnt_cd11', 'evnt_cd4', 'evnt_cd7']:
    df = hc.sql(f"""
        SELECT a.column1, a.column2, 
          CASE
            WHEN a.column3 IS NOT NULL THEN a.column3
            WHEN a.column3 IS NULL THEN '000000'
          END AS a.column3, a.{col} AS evnt,
          a.column5
        FROM {db_name}{tbl_name} AS a
        WHERE a.evnt_cd = '{cd}' and a.part_yrmn BETWEEN '{START_YRMN}' and '{END_YRMN}'
    """)


  elif cd == 'evnt_cd2':
    df = hc.sql(f"""
      SELECT a.column1, a.column2,
        CASE
          WHEN a.column3 IS NOT NULL THEN a.column3
          WHEN a.column3 IS NULL THEN '000000'
        END AS a.column3,
        CASE
          WHEN (a.column4 != 1111) OR ((a.column4 =  1111) AND (a.{col[1]} = "-")) THEN a.{col[0]}
          WHEN ((a.column4 = 1111) AND (a.{col[1]} != "-")) THEN a.{col[1]}
        END AS evnt,
        a.column5
      FROM {db_name}{tbl_name} AS a
        LEFT JOIN {db_name1}{tbl_name1} AS b ON a.{col[1]} = b.column1
          and b.column2 = '9999-12-31' and b.column2 = '2222'
        LEFT JOIN {db_name}{tbl_name2} t ON a.column5 = t.column1
      WHERE a.column5 ='{cd}' and a.part_yrmn BETWEEN '{START_YRMN}' and '{END_YRMN}'
    """)

  else:
    df = hc.sql(f"""
      SELECT a.column1, a.column2,
        CASE
          WHEN a.column3 IS NOT NULL THEN a.column3
          WHEN a.column3 IS NULL THEN '000000'
        END AS a.column3, concat(a.{col[0]}, '_', a.{col[1]}) as evnt, 
        a.column5
      FROM {db_name}{tbl_name} AS a
      WHERE a.column5 ='{cd}' and a.part_yrmn BETWEEN '{START_YRMN}' and '{END_YRMN}'
    """)


from functools import reduce
from pyspark.sql import DataFrame
import re
df_series = reduce(DataFrame.unionAll, df_list)

# check NULL event 
# df_series.filter(F.col('evnt').isNull()).limit(10).toPandas()
# print(df_series.filter(F.col('evnt').isNull()).select('column5').distinct().collect())
# print(df_series.filter(F.col('evnt').isNull()).count())

# space, 특수문자 처리
@udf(returnType=StrintType())
def parseEvent(evnt:str):
  evnt_parsed = evnt.replace(' ', '')
  evnt_parsed = re.sub(r'[\uC00-\uD7A30-9a-zA-Z]', '_', evnt_parsed)
  return evnt_parsed

# underbar 전처리(제거)
@udf(returnType=StirngType())
def parseEvent_underbar(evnt:str):
  evnt_parsed = evnt.replace('_', '')
  return evnt_parsed


df_series_ordered = df_series.orderBy(column1, column2, column3)

















