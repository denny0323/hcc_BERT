'''
    0. Settings
'''


from datetime import datetime
from hyspark import Hyspark

now= datetime.now()
curr_time = now.strftime('%H:%M:%S')

import sys
sys.path.append('../utils')
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




'''
    1. Make Datasets
'''


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
df_series_ordered = df_series.orderBy('column1', 'column2', 'column3')


from pyspark.sql.types import ArrayType, StringType, IntegerType
from pyspark.sql.functions import udf

@udf(returnType=IntegerType())
def ToLabel(evnt:str) -> int:
    evnt = str(evnt)
    MccnDict = {'mccn1': 1,
                'mccn2': 2,
                'mccn3': 3,
                'mccn4': 4,
                'mccn5-1': 5, 'mccn5-2':5, 'mccn5-3':5,
                'mccn6-1': 6, 'mccn6-2': 6,
                'mccn7-1': 7, 'mccn7-2': 7, 'mccn7-3':7, 'mccn7-4':7}

    for Mccn in MccnDict.keys():
        if evnt == 'mccn4':
            return MccnDict[evnt]
        elif Mccn in evnt:
            return MccnDict[Mccn]
    return 0

df_labeled = df_series_ordered.withColumn('label', ToLabel(df_series_ordered.evnt))


df_agg = df_series_ordered.groupby('column1', 'column2')\
                          .agg(F.collect_list('evnt').alias('evnt'),
                               F.collect_list('label').alias('label'))\
                          .orderBy('column1', 'column2')

from pyspark.sql.window import Window
days = lambda d: d * 86400

y_timespan = 7
x_timespan = 30

windowSpec_y = Window.partitionBy('column1').orderBy(F.col('column2').cast('timestamp').cast('long'))\
                                            .rangeBetween(1, days(y_timespan))

df_y = df_agg.withColumn('y'. F.concat_ws(' ', F.collect_list('label')\
                                                .over(windowSpec_y)))\
             .orderBy('column1', 'column2')

windowSpec_x = Window.partitionBy('column1').orderBy(F.col('column2').cast('timestamp').cast('long'))\
                                            .rangeBetween(-days(x_timespan), Window.currentRow)

df_xy = df_y.withColumn('x', F.concat_ws(' ', F.collect_list('evnt').over(windowSpec_x)))\
            .orderBy('column1', 'column2')


from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import udf

@udf(returnType=ArrayType(IntegerType()))
def multi_labelize(col):
    week_y = [0] * 7
    distinct_labels = list(map(int, set(col.split())))
    distinct_labels = [label for label in distinct_labels if label] ## 0이 아닌 label들만 select
    if not sum(distinct_labels): ## all_zeros -> 모두 0으로 return
        return week_y
    else:
        for label in distinct_labels:
            if label == 0:
                week_y[label] = 1
            else:
                week_y[label-1] = 1
        return week_y

df = df_xy.withColumn('y', multi_labelize(df_xy.y))

@udf(returnType=IntegerType())
def sum_list(col):
    return sum(col)

# non-zero (안 그러면 label에 0이 너무 많아짐: imbalanced)
df_nz = df.select('column1', 'column2', 'x', 'y')\
          .withColumn('sum', sum_list(F.col('y')))\
          .filter(F.col('sum') > 0)\
          .orderBy('column1', 'column2')\
          .select('column1', 'column2', 'x', 'y')



db_name = 'my_db_nm'
table_name = 'my_tbl_nm'
with elapsed_time():
    save_pyspark_df_as_table(hc, df_nz, db_name, table_name)