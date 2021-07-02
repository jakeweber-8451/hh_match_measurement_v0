# Databricks notebook source
# MAGIC %md ## HH Match Process

# COMMAND ----------

# MAGIC %md ### Parameters & Libraries

# COMMAND ----------

## Project Parameters
shortname = 'knox_wag'

#############################
## Dynamic Parameters
start_date = '20200830'
num_wks_meas = 20

day_of_week_start = 'SUNDAY'

model_period_length = 20



# COMMAND ----------

### Import Libraries
import pandas as pd
import numpy as np
from effodata import ACDS, golden_rules, Joiner, Sifter, Equality
from kpi_metrics import KPI, AliasMetric, CustomMetric, AliasGroupby, Rollup, Cube, available_metrics, get_metrics
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window
import re
from pyspark.sql.types import StringType 
from shutil import copyfile
from IPython.display import FileLink 
import os
import sys
import time
import datetime
import seg

from sklearn import linear_model, metrics

import statsmodels.api as sm

# COMMAND ----------

acds = ACDS(use_sample_mart=False) 
kpi = KPI(use_sample_mart=False)

# COMMAND ----------

# MAGIC %md ### Data Pull & Aggregation

# COMMAND ----------

## Some of the relevant input files for Walgreens measurement and Hometown pickup measurement:

# read zip code latitude/longitude reference file:
zipcode_lat_long = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/zipcode_lat_long.csv")

# read Hometown zip code mapping reference file:
hometown_zip_codes = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/hometown_zip_codes.csv")

# read Hometown zip code mapping reference file:
knox_kr_stores = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_stores.csv")\
  .withColumnRenamed('knox_stores','store_code').cache()

# COMMAND ----------

##Building the custom stores table

knox_wag_stores = acds.stores\
  .filter((f.col("mgt_div_no") == '105') & (f.col("sto_sta_cd") == "TN")).cache()

store_list = acds.stores\
  .withColumn("knox_wag_store_flag", f.when((f.col("mgt_div_no") == '105') & (f.col("sto_sta_cd") == "TN"), 'Y').otherwise('N'))\
  .select("store_code","knox_wag_store_flag").cache()

# COMMAND ----------

## Building the custom dates table
dates_acds = acds.dates

dates_custom_setup = dates_acds\
  .filter(f.col("fiscal_year") >= 2019)\
  .select('date','day_of_week_name','fiscal_week','trn_dt')\
  .sort('date')

# COMMAND ----------

## Dataframe to identify the start DOW of the weekly period

dow_array = np.array(["SUNDAY","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY"])

dow_start = np.repeat(dow_array, 7, axis=0)
dow_date = np.tile(dow_array, 7)

dow_adj = np.array([1,2,3,4,5,6,7])

dow_adj_setup1 = np.repeat(dow_adj, 7, axis=0)
dow_adj_setup2 = np.tile(dow_adj, 7)

df_dow = {"dow_start": dow_start, 
          "dow_date": dow_date, 
          "dow_adj_setup1": dow_adj_setup1, 
          "dow_adj_setup2": dow_adj_setup2}

df_dow = pd.DataFrame(df_dow)
df_dow = spark.createDataFrame(df_dow)

df_dow = df_dow.withColumn("dow_adj_setup3", f.col("dow_adj_setup1") - f.col("dow_adj_setup2"))\
  .withColumn("dow_adj", f.when(f.col("dow_adj_setup3") > 0, f.col("dow_adj_setup3") - 7).otherwise(f.col("dow_adj_setup3")))\
  .select("dow_start","dow_date","dow_adj")

df_dow_filtered = df_dow.filter(f.col("dow_start") == day_of_week_start)

# COMMAND ----------

## Finalizing the dates table:
dates_custom = dates_custom_setup\
  .join(df_dow_filtered,(dates_custom_setup.day_of_week_name == df_dow_filtered.dow_date), how = 'inner')\
  .withColumn("dow_adj", df_dow_filtered["dow_adj"].cast("integer"))\
  .withColumn("week_start_date", f.col("date") + f.col("dow_adj"))\
  .sort('date')

start_date_value = datetime.datetime.strptime(start_date,"%Y%m%d")

week_order = dates_custom\
  .select("week_start_date").distinct()\
  .withColumn("week_start_id", dates_custom["week_start_date"].cast(StringType()))\
  .withColumn("week_start_id", f.translate(f.col("week_start_id"), "-", ""))\
  .distinct()\
  .withColumn("period_flag", f.when(f.col("week_start_date") < start_date_value, "PRE").otherwise("POST"))\
  .withColumn("week_id", f.row_number().over(Window.partitionBy("period_flag").orderBy("week_start_date")))\
  .withColumn("week_count", f.count("period_flag").over(Window.partitionBy("period_flag")))\
  .withColumn("week_descending", f.col("week_count") - f.col("week_id") + 1)\
  .withColumn("model_period_0", f.when((f.col("period_flag") == 'PRE') & (f.col("week_descending") >=  1) & (f.col("week_descending") <= 13), "PRE 13 WKS")\
                                 .when((f.col("period_flag") == 'PRE') & (f.col("week_descending") >= 14) & (f.col("week_descending") <= 26), "PRE 26 WKS")\
                                 .when((f.col("period_flag") == 'PRE') & (f.col("week_descending") >= 27) & (f.col("week_descending") <= 52), "PRE 52 WKS")\
                                 .otherwise(f.col("period_flag")))\
  .filter(~((f.col("period_flag") == "PRE") & (f.col("week_descending") > 52)) & ~((f.col("period_flag") == "POST") & (f.col("week_id") > num_wks_meas)))\
  .withColumn("yoy_period_setup", f.lead(f.col("period_flag"),52).over(Window.orderBy("week_start_id")))\
  .withColumn("yoy_period", f.when((f.col("period_flag") == "PRE") & (f.col("yoy_period_setup") == "POST"), "PRE YOY").otherwise(f.col("period_flag")))\
  .sort("week_start_date")

dates_custom_joiner = dates_custom\
  .join(week_order,(dates_custom.week_start_date == week_order.week_start_date), how = 'inner').drop(week_order.week_start_date)\
  .select("date","week_start_date","week_start_id")\
  .sort("date").cache()

week_order_joiner = week_order\
  .select("week_start_date", "week_start_id", "period_flag", "model_period_0", "yoy_period")\
  .sort("week_start_date").cache()



# COMMAND ----------

## Isolating the end of the pre and post periods:

first_pre_date_setup1 = week_order\
  .filter(f.col("period_flag") == "PRE")\
  .withColumn("week_end_date", f.col("week_start_date") + 6)

first_pre_date_setup2 = first_pre_date_setup1\
  .withColumn("week_start_date", first_pre_date_setup1["week_start_date"].cast(StringType()))\
  .withColumn("week_start_date", f.translate(f.col("week_start_date"), "-", ""))
  
first_pre_date = first_pre_date_setup2.agg({"week_start_date": "min"}).collect()[0][0]


last_pre_date_setup1 = week_order\
  .filter(f.col("period_flag") == "PRE")\
  .withColumn("week_end_date", f.col("week_start_date") + 6)

last_pre_date_setup2 = last_pre_date_setup1\
  .withColumn("week_end_date", last_pre_date_setup1["week_end_date"].cast(StringType()))\
  .withColumn("week_end_date", f.translate(f.col("week_end_date"), "-", ""))
  
last_pre_date = last_pre_date_setup2.agg({"week_end_date": "max"}).collect()[0][0]


last_post_date_setup1 = week_order\
  .filter(f.col("period_flag") == "POST")\
  .withColumn("week_end_date", f.col("week_start_date") + 6)

last_post_date_setup2 = last_post_date_setup1\
  .withColumn("week_end_date", last_post_date_setup1["week_end_date"].cast(StringType()))\
  .withColumn("week_end_date", f.translate(f.col("week_end_date"), "-", ""))
  
last_post_date = last_post_date_setup2.agg({"week_end_date": "max"}).collect()[0][0]

print(first_pre_date)
print(last_pre_date)
print(last_post_date)


# COMMAND ----------

# Get Jasmine Pickup Households
wag_hshds = kpi.get_aggregate(start_date = last_pre_date, 
                              end_date = last_post_date,
                              join_with = [Joiner(df = f.broadcast(dates_custom_joiner),join_cond = Equality(['date']),method='inner'),
                                           'dates'],
                              filter_by = [Sifter(df = f.broadcast(knox_wag_stores),join_cond = Equality(['store_code']),method='include')],
                              group_by = ['ehhn'],
                              metrics = ["sales"])\
  .filter(f.col("sales") > 0).select("ehhn").cache()

# COMMAND ----------

# Get Control Households
knox_kr_hshds = kpi.get_aggregate(start_date = last_pre_date, 
                                  end_date = last_post_date,
                                  filter_by = [Sifter(df = f.broadcast(knox_kr_stores),join_cond = Equality(['store_code']),method="include"),
                                               Sifter(df = f.broadcast(wag_hshds), join_cond=Equality("ehhn"), method="exclude")],
                                  group_by = ['ehhn'],
                                  metrics = ["sales"])\
  .filter(f.col("sales") > 0).select("ehhn").cache()


# COMMAND ----------

## Defining the test and control households

wag_hsdhs_setup = wag_hshds.withColumn('jas_hshd_flag',f.lit('1'))

knox_kr_hshds_setup = knox_kr_hshds.withColumn('jas_hshd_flag',f.lit('0'))

tot_hshds =  wag_hsdhs_setup.union(knox_kr_hshds_setup).cache()

# COMMAND ----------

## Get KPIs by household
hshd_wkly_kpis_temp = kpi.get_aggregate(start_date = first_pre_date, 
                                        end_date = last_post_date,
                                        join_with = [Joiner(df = f.broadcast(dates_custom_joiner),join_cond = Equality(['date']),method='inner'),
                                                     Joiner(df = f.broadcast(store_list),join_cond = Equality(['store_code']),method='inner'),
                                                     Joiner(df = f.broadcast(tot_hshds),join_cond = Equality(['ehhn']),method='inner'),
                                                     'dates'],
                                        group_by = ['ehhn','week_start_date','week_start_id','jas_hshd_flag'],
                                        metrics = ["visits","sales","units",
                                                  CustomMetric("pickup_visits", "count(distinct case when knox_wag_store_flag = 'N' and order_type = '1' then transaction_code else null end)"),
                                                  CustomMetric("pickup_sales", "sum(case when knox_wag_store_flag = 'N' and order_type = '1' then net_spend_amt else 0 end)"),
                                                  CustomMetric("pickup_units", "sum(case when knox_wag_store_flag = 'N' and order_type = '1' then scn_unt_qy else 0 end)"),
                                                  CustomMetric("delivery_visits", "count(distinct case when knox_wag_store_flag = 'N' and order_type = '3' then transaction_code else null end)"),
                                                  CustomMetric("delivery_sales", "sum(case when knox_wag_store_flag = 'N' and order_type = '3' then net_spend_amt else 0 end)"),
                                                  CustomMetric("delivery_units", "sum(case when knox_wag_store_flag = 'N' and order_type = '3' then scn_unt_qy else 0 end)"),
                                                  CustomMetric("wag_visits", "count(distinct case when knox_wag_store_flag = 'Y' then transaction_code else null end)"),
                                                  CustomMetric("wag_sales", "sum(case when knox_wag_store_flag = 'Y' then net_spend_amt else 0 end)"),
                                                  CustomMetric("wag_units", "sum(case when knox_wag_store_flag = 'Y' then scn_unt_qy else 0 end)")],
                                        apply_golden_rules = golden_rules()).cache()


# COMMAND ----------

## Identifying the first jasmine visit by household
hshd_wkly_kpis = hshd_wkly_kpis_temp\
  .withColumn("first_jas_visit_setup", f.when((f.col("wag_visits") > 0 ), (f.col("week_start_id"))).otherwise(None))\
  .withColumn("first_jas_visit", f.min("first_jas_visit_setup").over(Window.partitionBy('ehhn')))\
  .drop("first_jas_visit_setup").cache()

hshd_first_visit = hshd_wkly_kpis\
  .select("ehhn","first_jas_visit").distinct().cache()



# COMMAND ----------

# MAGIC %md ### Post-Period Iterations

# COMMAND ----------

i = 1

##  Identifying the proper pre/post period weeks for the current group of Jasmine engagement households in the modeling/pre window:
week_order_i = week_order\
  .withColumn("model_period_setup1", f.lag(f.col("model_period_0"),model_period_length*(i-1)).over(Window.orderBy("week_start_id")))\
  .withColumn("model_period_setup2", f.lag(f.col("period_flag"),model_period_length*(i)).over(Window.orderBy("week_start_id")))\
  .withColumn("model_period_i", f.when((f.col("model_period_setup1") == "POST") & (f.col("model_period_setup2") == "PRE"), "MODELING")\
              .otherwise(f.col("model_period_setup1"))).cache()

# COMMAND ----------

## Defining the post-period dates for the ith run:
first_last_post_date_i_setup1 = week_order_i\
  .filter(f.col("model_period_i") == "MODELING")\
  .withColumn("week_end_date", f.col("week_start_date") + 6)

first_last_post_date_i_setup2 = first_last_post_date_i_setup1\
  .withColumn("week_start_date", first_last_post_date_i_setup1["week_start_date"].cast(StringType()))\
  .withColumn("week_start_date", f.translate(f.col("week_start_date"), "-", ""))\
  .withColumn("week_end_date", first_last_post_date_i_setup1["week_end_date"].cast(StringType()))\
  .withColumn("week_end_date", f.translate(f.col("week_end_date"), "-", ""))
  
first_post_date_i = first_last_post_date_i_setup2.agg({"week_start_date": "min"}).collect()[0][0]
last_post_date_i = first_last_post_date_i_setup2.agg({"week_end_date": "max"}).collect()[0][0]

print(first_post_date_i)
print(last_post_date_i)

# COMMAND ----------

## Aggregating KPIs for the modeling/pre/post periods:

week_order_i_joiner = week_order_i\
  .select("week_start_id","model_period_i")

hshd_wkly_kpis_modeling = hshd_wkly_kpis\
  .join(week_order_i_joiner,(hshd_wkly_kpis.week_start_id == week_order_i_joiner.week_start_id), how = 'inner').drop(week_order_i_joiner.week_start_id)\
  .withColumn("tot_sales_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), (f.col("sales"))).otherwise(f.lit(0)))\
  .withColumn("tot_sales_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), (f.col("sales"))).otherwise(f.lit(0)))\
  .withColumn("tot_sales_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), (f.col("sales"))).otherwise(f.lit(0)))\
  .withColumn("tot_sales_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), (f.col("sales"))).otherwise(f.lit(0)))\
  .withColumn("tot_sales_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), (f.col("sales"))).otherwise(f.lit(0)))\
  .withColumn("tot_units_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), (f.col("units"))).otherwise(f.lit(0)))\
  .withColumn("tot_units_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), (f.col("units"))).otherwise(f.lit(0)))\
  .withColumn("tot_units_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), (f.col("units"))).otherwise(f.lit(0)))\
  .withColumn("tot_units_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), (f.col("units"))).otherwise(f.lit(0)))\
  .withColumn("tot_units_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), (f.col("units"))).otherwise(f.lit(0)))\
  .withColumn("tot_visits_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), (f.col("visits"))).otherwise(f.lit(0)))\
  .withColumn("tot_visits_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), (f.col("visits"))).otherwise(f.lit(0)))\
  .withColumn("tot_visits_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), (f.col("visits"))).otherwise(f.lit(0)))\
  .withColumn("tot_visits_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), (f.col("visits"))).otherwise(f.lit(0)))\
  .withColumn("tot_visits_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), (f.col("visits"))).otherwise(f.lit(0)))\
  .withColumn("kr_sales_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("kr_sales_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("kr_sales_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), 
                                     (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("kr_sales_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), 
                                          (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("kr_sales_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), 
                                      (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("kr_units_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("kr_units_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("kr_units_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), 
                                     (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("kr_units_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), 
                                          (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("kr_units_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), 
                                      (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("kr_visits_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("kr_visits_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})), 
                                     (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("kr_visits_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})), 
                                     (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("kr_visits_modeling", f.when((f.col("model_period_i").isin({"MODELING"})), 
                                           (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("kr_visits_post", f.when((f.col("model_period_i").isin({"MODELING","POST"})), 
                                       (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("pickup_sales_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_sales"))).otherwise(f.lit(0)))\
  .withColumn("pickup_sales_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_sales"))).otherwise(f.lit(0)))\
  .withColumn("pickup_sales_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("pickup_sales"))).otherwise(f.lit(0)))\
  .withColumn("pickup_sales_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("pickup_sales"))).otherwise(f.lit(0)))\
  .withColumn("pickup_sales_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("pickup_sales"))).otherwise(f.lit(0)))\
  .withColumn("pickup_units_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_units"))).otherwise(f.lit(0)))\
  .withColumn("pickup_units_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_units"))).otherwise(f.lit(0)))\
  .withColumn("pickup_units_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("pickup_units"))).otherwise(f.lit(0)))\
  .withColumn("pickup_units_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("pickup_units"))).otherwise(f.lit(0)))\
  .withColumn("pickup_units_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("pickup_units"))).otherwise(f.lit(0)))\
  .withColumn("pickup_visits_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_visits"))).otherwise(f.lit(0)))\
  .withColumn("pickup_visits_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("pickup_visits"))).otherwise(f.lit(0)))\
  .withColumn("pickup_visits_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("pickup_visits"))).otherwise(f.lit(0)))\
  .withColumn("pickup_visits_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("pickup_visits"))).otherwise(f.lit(0)))\
  .withColumn("pickup_visits_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("pickup_visits"))).otherwise(f.lit(0)))\
  .withColumn("delivery_sales_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_sales"))).otherwise(f.lit(0)))\
  .withColumn("delivery_sales_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_sales"))).otherwise(f.lit(0)))\
  .withColumn("delivery_sales_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("delivery_sales"))).otherwise(f.lit(0)))\
  .withColumn("delivery_sales_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("delivery_sales"))).otherwise(f.lit(0)))\
  .withColumn("delivery_sales_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("delivery_sales"))).otherwise(f.lit(0)))\
  .withColumn("delivery_units_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_units"))).otherwise(f.lit(0)))\
  .withColumn("delivery_units_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_units"))).otherwise(f.lit(0)))\
  .withColumn("delivery_units_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("delivery_units"))).otherwise(f.lit(0)))\
  .withColumn("delivery_units_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("delivery_units"))).otherwise(f.lit(0)))\
  .withColumn("delivery_units_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("delivery_units"))).otherwise(f.lit(0)))\
  .withColumn("delivery_visits_52", f.when((f.col("model_period_i").isin({"PRE 52 WKS", "PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_visits"))).otherwise(f.lit(0)))\
  .withColumn("delivery_visits_26", f.when((f.col("model_period_i").isin({"PRE 26 WKS", "PRE 13 WKS"})),(f.col("delivery_visits"))).otherwise(f.lit(0)))\
  .withColumn("delivery_visits_13", f.when((f.col("model_period_i").isin({"PRE 13 WKS"})),(f.col("delivery_visits"))).otherwise(f.lit(0)))\
  .withColumn("delivery_visits_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("delivery_visits"))).otherwise(f.lit(0)))\
  .withColumn("delivery_visits_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("delivery_visits"))).otherwise(f.lit(0)))\
  .withColumn("wag_sales_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("wag_sales_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("wag_sales"))).otherwise(f.lit(0)))\
  .withColumn("wag_units_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("wag_units_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("wag_units"))).otherwise(f.lit(0)))\
  .withColumn("wag_visits_modeling", f.when((f.col("model_period_i").isin({"MODELING"})),(f.col("wag_visits"))).otherwise(f.lit(0)))\
  .withColumn("wag_visits_post", f.when((f.col("model_period_i").isin({"MODELING", "POST"})),(f.col("wag_visits"))).otherwise(f.lit(0)))\
  .groupBy("ehhn","jas_hshd_flag","first_jas_visit")\
  .agg(f.sum("tot_sales_52").alias("tot_sales_52"),
       f.sum("tot_sales_26").alias("tot_sales_26"),
       f.sum("tot_sales_13").alias("tot_sales_13"),
       f.sum("tot_sales_modeling").alias("tot_sales_modeling"),
       f.sum("tot_sales_post").alias("tot_sales_post"),
       f.sum("tot_units_52").alias("tot_units_52"),
       f.sum("tot_units_26").alias("tot_units_26"),
       f.sum("tot_units_13").alias("tot_units_13"),
       f.sum("tot_units_modeling").alias("tot_units_modeling"),
       f.sum("tot_units_post").alias("tot_units_post"),
       f.sum("tot_visits_52").alias("tot_visits_52"),
       f.sum("tot_visits_26").alias("tot_visits_26"),
       f.sum("tot_visits_13").alias("tot_visits_13"),
       f.sum("tot_visits_modeling").alias("tot_visits_modeling"),
       f.sum("tot_visits_post").alias("tot_visits_post"),
       f.sum("kr_sales_52").alias("kr_sales_52"),
       f.sum("kr_sales_26").alias("kr_sales_26"),
       f.sum("kr_sales_13").alias("kr_sales_13"),
       f.sum("kr_sales_modeling").alias("kr_sales_modeling"),
       f.sum("kr_sales_post").alias("kr_sales_post"),
       f.sum("kr_units_52").alias("kr_units_52"),
       f.sum("kr_units_26").alias("kr_units_26"),
       f.sum("kr_units_13").alias("kr_units_13"),
       f.sum("kr_units_modeling").alias("kr_units_modeling"),
       f.sum("kr_units_post").alias("kr_units_post"),
       f.sum("kr_visits_52").alias("kr_visits_52"),
       f.sum("kr_visits_26").alias("kr_visits_26"),
       f.sum("kr_visits_13").alias("kr_visits_13"),
       f.sum("kr_visits_modeling").alias("kr_visits_modeling"),
       f.sum("kr_visits_post").alias("kr_visits_post"),
       f.sum("pickup_sales_52").alias("pickup_sales_52"),
       f.sum("pickup_sales_26").alias("pickup_sales_26"),
       f.sum("pickup_sales_13").alias("pickup_sales_13"),
       f.sum("pickup_sales_modeling").alias("pickup_sales_modeling"),
       f.sum("pickup_sales_post").alias("pickup_sales_post"),
       f.sum("pickup_units_52").alias("pickup_units_52"),
       f.sum("pickup_units_26").alias("pickup_units_26"),
       f.sum("pickup_units_13").alias("pickup_units_13"),
       f.sum("pickup_units_modeling").alias("pickup_units_modeling"),
       f.sum("pickup_units_post").alias("pickup_units_post"),
       f.sum("pickup_visits_52").alias("pickup_visits_52"),
       f.sum("pickup_visits_26").alias("pickup_visits_26"),
       f.sum("pickup_visits_13").alias("pickup_visits_13"),
       f.sum("pickup_visits_modeling").alias("pickup_visits_modeling"),
       f.sum("pickup_visits_post").alias("pickup_visits_post"),
       f.sum("delivery_sales_52").alias("delivery_sales_52"),
       f.sum("delivery_sales_26").alias("delivery_sales_26"),
       f.sum("delivery_sales_13").alias("delivery_sales_13"),
       f.sum("delivery_sales_modeling").alias("delivery_sales_modeling"),
       f.sum("delivery_sales_post").alias("delivery_sales_post"),
       f.sum("delivery_units_52").alias("delivery_units_52"),
       f.sum("delivery_units_26").alias("delivery_units_26"),
       f.sum("delivery_units_13").alias("delivery_units_13"),
       f.sum("delivery_units_modeling").alias("delivery_units_modeling"),
       f.sum("delivery_units_post").alias("delivery_units_post"),
       f.sum("delivery_visits_52").alias("delivery_visits_52"),
       f.sum("delivery_visits_26").alias("delivery_visits_26"),
       f.sum("delivery_visits_13").alias("delivery_visits_13"),
       f.sum("delivery_visits_modeling").alias("delivery_visits_modeling"),
       f.sum("delivery_visits_post").alias("delivery_visits_post"),
       f.sum("wag_sales_modeling").alias("wag_sales_modeling"),
       f.sum("wag_sales_post").alias("wag_sales_post"),
       f.sum("wag_units_modeling").alias("wag_units_modeling"),
       f.sum("wag_units_post").alias("wag_units_post"),
       f.sum("wag_visits_modeling").alias("wag_visits_modeling"),
       f.sum("wag_visits_post").alias("wag_visits_post"))\
  .withColumnRenamed("ehhn","ehhn_kpi").cache()




# COMMAND ----------

## Pulling in segmentations right before the modeling period:

hshd_funlo = seg.get_segs_for_date(segs=["funlo"], date=last_pre_date)[0]\
  .select("ehhn","funlo_seg_desc","funlo_rollup_desc").cache()

hshd_cust_seg = seg.get_segs_for_date(segs=["cds_hh"], date=last_pre_date)[0]\
  .select("ehhn","convenience_dim_seg","quality_dim_seg","health_dim_seg","inspiration_dim_seg","price_dim_seg").cache()

hshd_sow = seg.get_segs_for_date(segs=["sow"], date=last_pre_date)[0]\
  .select("ehhn","sow_seg","sow_perc").cache()

hshd_comp_eng = seg.get_segs_for_date(segs=["comp_eng"], date=last_pre_date)[0]\
  .select("ehhn","club_prediction","dd_prediction","mass_prediction").cache()

hshd_pref_store = seg.get_segs_for_date(segs=["pref_store"], date=last_pre_date)[0]

hshd_pref_store = hshd_pref_store\
  .select("ehhn","pref_store_code_1")\
  .withColumnRenamed("pref_store_code_1","pref_store")\
  .withColumn("pref_div",f.substring(f.col("pref_store"), 1,3))

# COMMAND ----------

## Final KPI/segmentation data

hshd_wkly_kpis_modeling_full = hshd_wkly_kpis_modeling\
  .join(hshd_funlo,(hshd_wkly_kpis_modeling.ehhn_kpi == hshd_funlo.ehhn), how = 'left').drop(hshd_funlo.ehhn)\
  .join(hshd_cust_seg,(hshd_wkly_kpis_modeling.ehhn_kpi == hshd_cust_seg.ehhn), how = 'left').drop(hshd_cust_seg.ehhn)\
  .join(hshd_sow,(hshd_wkly_kpis_modeling.ehhn_kpi == hshd_sow.ehhn), how = 'left').drop(hshd_sow.ehhn)\
  .join(hshd_comp_eng,(hshd_wkly_kpis_modeling.ehhn_kpi == hshd_comp_eng.ehhn), how = 'left').drop(hshd_comp_eng.ehhn)\
  .join(hshd_pref_store,(hshd_wkly_kpis_modeling.ehhn_kpi == hshd_pref_store.ehhn), how = 'left').drop(hshd_pref_store.ehhn)\
  .withColumnRenamed("ehhn_kpi","ehhn")\
  .filter(((f.col("first_jas_visit") >= first_post_date_i) & (f.col("first_jas_visit") <= last_post_date_i)) | ((f.col("first_jas_visit").isNull()))).cache()
  

# COMMAND ----------

## Write Final KPI/segmentation data
hshd_wkly_kpis_modeling_full.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_kpi_seg_final.csv")

# COMMAND ----------

# Reading the previously-saved data:
hshd_wkly_kpis_modeling_full = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_kpi_seg_final.csv").cache()

# COMMAND ----------

# Downsampling
hshd_wkly_kpis_modeling_N_downsamp = hshd_wkly_kpis_modeling_full\
  .filter((f.col("jas_hshd_flag") == '0'))\
  .sample(0.1)

hshd_wkly_kpis_modeling_Y = hshd_wkly_kpis_modeling_full\
  .filter((f.col("jas_hshd_flag") == '1'))

hshd_wkly_kpis_modeling_final = hshd_wkly_kpis_modeling_Y\
  .union(hshd_wkly_kpis_modeling_N_downsamp).cache()


# COMMAND ----------

## Write Final Modeling data
hshd_wkly_kpis_modeling_final.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_modeling.csv")

# COMMAND ----------

# Reading the previously-saved modeling data:
hshd_wkly_kpis_modeling_final = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_modeling.csv").cache()

# COMMAND ----------

# Check counts of test/control households
hshd_wkly_kpis_modeling_final\
  .groupBy("jas_hshd_flag")\
  .agg(f.countDistinct("ehhn").alias("countem")).display()

# COMMAND ----------

# MAGIC %md ### Modeling

# COMMAND ----------

# Removing post-period fields for the modeling dataframe
hshd_wkly_kpis_modeling_final_pd = hshd_wkly_kpis_modeling_final\
  .drop("tot_visits_post",
        "kr_sales_modeling","kr_sales_post",
        "kr_units_modeling","kr_units_post",
        "kr_visits_modeling","kr_visits_post",
        "pickup_sales_modeling","pickup_sales_post",
        "pickup_units_modeling","pickup_units_post",
        "pickup_visits_modeling","pickup_visits_post",
        "delivery_sales_modeling","delivery_sales_post",
        "delivery_units_modeling","delivery_units_post",
        "delivery_visits_modeling","delivery_visits_post",
        "wag_sales_modeling","wag_sales_post",
        "wag_units_modeling","wag_units_post",
        "wag_visits_modeling","wag_visits_post").toPandas()

# COMMAND ----------

# Install drpyspark
dbutils.library.installPyPI('drpyspark')

# Install shap 
dbutils.library.installPyPI('shap', '0.39.0')

import datarobot as dr  # official DataRobot interface 
#import drex             # internal (i.e., unofficial) DataRobot EXtension package
import shap             # for nice SHAP visualizations
import urllib3 as urll


# Needed so that DataRobot doesn't spam you with warning messages :/
urll.disable_warnings(urll.exceptions.InsecureRequestWarning)


# Connect to DataRobot; note that it's possible to store your token in a Databricks-backed secret scope
dr.Client(
    endpoint="https://datarobot.8451.cloud/api/v2",
    #token=dr.dbutils.secrets.get(scope="b780620-dbxadhoc", key="datarobot")
    token="NjA5YzEzNWNhZDJmMzU0YjI2ODMxNTUzOi9BNkdjR0ZUT1FXR0VmSnRwUGRnSGVMalN5YTFXT2VzMS82NFZoVk92VEk9",
    ssl_verify=False
)

# COMMAND ----------

# Start a new project
project = dr.Project.create(
    sourcedata = hshd_wkly_kpis_modeling_final_pd,  # has to be a Pandas DataFrame or valid path  to a CSV file
    project_name = "wag_knox_model_testing"  # make this unique!
)

# COMMAND ----------

# Set target variable
project.set_target(
  target="jas_hshd_flag",
  metric="LogLoss",
  mode=dr.AUTOPILOT_MODE.FULL_AUTO,  # or MANUAL or QUICK
  target_type="Binary",
  advanced_options=dr.AdvancedOptions(
    seed=8451,
    prepare_model_for_deployment=False,
    blend_best_models=False))

# Periodically check whether AutoPilot is finished
project.wait_for_autopilot()

# COMMAND ----------

# Find the model with the smallest log loss (on the validation set) trained on at least 60% of the data
models = project.get_models()
res = [m for m in models
       if m.sample_pct > 60]
df = pd.DataFrame(res)

# COMMAND ----------

# Set Project ID
pid = project.id

# COMMAND ----------

# Identify best model by default criteria
best = res[0]

# COMMAND ----------

#unlock holdout
project.unlock_holdout()

# COMMAND ----------

# Train a model on a different sample size
model_job_id = best.train(sample_pct=100)

# COMMAND ----------

# Upload scoring data to DataRobot
data_to_score = project.upload_dataset(hshd_wkly_kpis_modeling_final_pd)

# COMMAND ----------

# Request predictions for uploaded data set
predict_job = best.request_predictions(data_to_score.id)

# Wait until the previous job is done
predict_job.wait_for_completion()

# If you happen to have only the job ID and not the prediction job
# object, you can use the PredictJob.get function:
# predict_job = dr.PredictJob.get(project_id=project.id,
#                                 predict_job_id=predict_job.id)

# COMMAND ----------

# Fetch all available predictions associated with this project ID
prediction_list = dr.Predictions.list(project.id)

# COMMAND ----------

# Retrieve generated model predictions
predictions = (dr.PredictJob.get_predictions(project_id = project.id, predict_job_id = predict_job.id))
predictions = predict_job.get_result_when_complete()

# COMMAND ----------

# Deleting scoring data
data_to_score.delete()

# COMMAND ----------

# Isolate predictions
predictions_for_binding = predictions[["positive_probability"]]

# COMMAND ----------

#Dataframe w/ predictions
hshd_wkly_kpis_modeling_w_preds = pd.concat([hshd_wkly_kpis_modeling_final_pd, predictions_for_binding], axis = 1)

hshd_data_w_preds = spark.createDataFrame(hshd_wkly_kpis_modeling_w_preds)\
  .withColumnRenamed("positive_probability","jas_hshd_pred")

# COMMAND ----------

## Write Final Modeling data
hshd_data_w_preds.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_preds.csv")

# COMMAND ----------

# Reading the previously-saved modeling data:
hshd_data_w_preds = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_preds.csv").cache()

# COMMAND ----------

# MAGIC %md ### Matching

# COMMAND ----------

## Isolating test HHs
target_hshds = hshd_data_w_preds.filter(f.col("jas_hshd_flag")==1).sort(f.col("jas_hshd_pred").desc())
## Isolating control HHS
non_target_hshds = hshd_data_w_preds.filter(f.col("jas_hshd_flag")==0).sort(f.col("jas_hshd_pred").desc())

# COMMAND ----------

## Loop through every test HH
rows_to_loop = target_hshds.count()

## Unless I convert the above to an array (which will make calculations a little trickier), I'll add an "index" created by the row number
w=Window.orderBy(f.lit(1))
target_hshds = target_hshds.withColumn("index",f.row_number().over(w) - 1)

# COMMAND ----------

# Breaking out target vs. non-target households
target_hshds_pd = target_hshds.toPandas()
non_target_hshds_pd = non_target_hshds.toPandas()

# COMMAND ----------

# Giving each test household their top 200 closest households by propensity score:
for j in range(rows_to_loop):

  c_hshd = target_hshds_pd[(target_hshds_pd.index == j)][["ehhn"]].values.tolist()[0][0]
  c_hshd_div = target_hshds_pd[(target_hshds_pd.index == j)][["pref_div"]].values.tolist()[0][0]
  c_hshd_pred = target_hshds_pd[(target_hshds_pd.index == j)][["jas_hshd_pred"]].values.tolist()[0][0]

  ##Filter to only control HHs with the same pref div as the test HH
  ## WELLP. THIS MIGHT NOT WORK TO FILTER ON NULLS
  if c_hshd_div == None:
    non_target_hshds_pref_div = non_target_hshds_pd[(non_target_hshds_pd.pref_div.isnull())]
  else:
    non_target_hshds_pref_div = non_target_hshds_pd[(non_target_hshds_pd.pref_div == c_hshd_div)]

  ## ID controls for each test HH
  find_match_multiplier = 1
  match_counter = 0

  rows_counter = non_target_hshds_pref_div.shape[0]
  ## Ask Luke why it's 200 here
  while (match_counter < min(200,rows_counter)):
    
    target_hshd_matching_setup = non_target_hshds_pref_div.assign(jas_test_id = c_hshd, 
                                                                  jas_test_pred = c_hshd_pred)

    target_hshd_matching_setup = target_hshd_matching_setup[["jas_test_id", "ehhn", "jas_test_pred", "jas_hshd_pred"]]\
      .rename(columns = {"ehhn":"jas_control_id", "jas_hshd_pred":"jas_control_pred"})

    target_hshd_matching_setup["jas_test_pred"] = target_hshd_matching_setup["jas_test_pred"].astype(float)
    target_hshd_matching_setup["jas_control_pred"] = target_hshd_matching_setup["jas_control_pred"].astype(float)
    target_hshd_matching_setup["matching_metric"] = abs(target_hshd_matching_setup["jas_test_pred"] - target_hshd_matching_setup["jas_control_pred"])

    target_hshd_matching_setup = target_hshd_matching_setup[(target_hshd_matching_setup.matching_metric <= (0.01*find_match_multiplier))].sort_values(by=['matching_metric'], ascending = False)

    match_counter = target_hshd_matching_setup.shape[0]
    find_match_multiplier = find_match_multiplier + 1

  if j == 0:
    target_hshd_matching = target_hshd_matching_setup
  else:
    target_hshd_matching = pd.concat([target_hshd_matching, target_hshd_matching_setup], ignore_index=True)

# COMMAND ----------

## Beginning the matching process:

## Identifying the best match for each test household, idependent of whether or not a control household
##  is selected for multiple test households:
target_hshd_matching_top_rank = target_hshd_matching
target_hshd_matching_top_rank["match_rank"] = target_hshd_matching.groupby("jas_test_id")["matching_metric"].rank(method="first", ascending=False)
target_hshd_matching_top_rank = target_hshd_matching_top_rank[(target_hshd_matching_top_rank.match_rank == 1)]

## ID all control HHs selected for multiple test HHs
target_hshd_multi_matches = target_hshd_matching_top_rank.reset_index(drop=True)
target_hshd_multi_matches = target_hshd_multi_matches.groupby("jas_control_id").agg(control_count=pd.NamedAgg(column="jas_control_id", aggfunc="count")).reset_index(drop=False)\
  .sort_values(by=['control_count'], ascending = False)
target_hshd_multi_matches = target_hshd_multi_matches[(target_hshd_multi_matches.control_count > 1)]

## How many HHs are multi-matched:
target_hshd_multi_match_count = target_hshd_multi_matches.shape[0]




# COMMAND ----------

## Control for HHs needing the match "the most"

## If no controls selected for multiple test stores, we're done
if target_hshd_multi_match_count == 0:
  target_hshd_matches_final = target_hshd_matching_top_rank
## Else, give the control HH to the test household that needs in the most 
## i.e. has the worst match among test households also matched to that control household
else:
  ## Flag to start building the appended multi-match HH dataframe:
  target_hshd_multi_match_counter = 0
  
  while target_hshd_multi_match_count > 0:
    
    ## Giving the control household to the test household that needs it most
    target_hshd_top_multi_match = target_hshd_matching_top_rank.merge(target_hshd_multi_matches[["jas_control_id"]], on = "jas_control_id", how = "inner")\
      .sort_values(by=['matching_metric'], ascending = False)
    target_hshd_top_multi_match["secondary_match_rank"] = target_hshd_top_multi_match.groupby("jas_control_id")["matching_metric"].rank(method="first", ascending=False)
    target_hshd_top_multi_match = target_hshd_top_multi_match[(target_hshd_top_multi_match.secondary_match_rank == 1)]
    
    ## Building the appended multi-match HH dataframe:
    if target_hshd_multi_match_counter == 0:
      target_hshd_matches_final = target_hshd_top_multi_match
    else:
      target_hshd_matches_final = pd.concat([target_hshd_matches_final, target_hshd_top_multi_match], ignore_index=True)
    
    ## The test and control households that were multi-matched but now are matched:
    target_test_hshd_matches_final = target_hshd_matches_final[["jas_test_id"]].drop_duplicates()
    target_control_hshd_matches_final = target_hshd_matches_final[["jas_control_id"]].drop_duplicates()
    
    ## ID best match for each test HH (not including the test households that were multi-matched earlier)
    target_hshd_matching_top_rank = target_hshd_matching.merge(target_test_hshd_matches_final, on = "jas_test_id", how = 'outer', indicator = True)
    target_hshd_matching_top_rank = target_hshd_matching_top_rank[~(target_hshd_matching_top_rank._merge == 'both')].drop('_merge', axis = 1)
    target_hshd_matching_top_rank = target_hshd_matching_top_rank.merge(target_control_hshd_matches_final, on = "jas_control_id", how = 'outer', indicator = True)
    target_hshd_matching_top_rank = target_hshd_matching_top_rank[~(target_hshd_matching_top_rank._merge == 'both')].drop('_merge', axis = 1)
    target_hshd_matching_top_rank["match_rank"] = target_hshd_matching_top_rank.groupby("jas_test_id")["matching_metric"].rank(method="first", ascending=False)
    target_hshd_matching_top_rank = target_hshd_matching_top_rank[(target_hshd_matching_top_rank.match_rank == 1)]
                                                                 
    ## ID all control HHs selected for multiple test stores if any
    target_hshd_multi_matches = target_hshd_matching_top_rank.reset_index(drop=True)
    target_hshd_multi_matches = target_hshd_multi_matches.groupby("jas_control_id").agg(control_count=pd.NamedAgg(column="jas_control_id", aggfunc="count")).reset_index(drop=False)\
      .sort_values(by=['control_count'], ascending = False)
    target_hshd_multi_matches = target_hshd_multi_matches[(target_hshd_multi_matches.control_count > 1)]
    
    ## Count control HHs selected for multiple test HHs if any - process again if there are multi-matched HHs still
    target_hshd_multi_match_count = target_hshd_multi_matches.shape[0]
    target_hshd_multi_match_counter = target_hshd_multi_match_counter + 1
    
  ## Once all test/control HHs are 1:1, output final match
  target_hshd_matches_final = pd.concat([target_hshd_matches_final.drop("secondary_match_rank", axis = 1), target_hshd_matching_top_rank], ignore_index=True)
  
target_hshd_matches_final = target_hshd_matches_final.sort_values(by=['matching_metric'], ascending = False)

# COMMAND ----------


## Write final matches to CSV

target_hshd_matches_final_pyspark = spark.createDataFrame(target_hshd_matches_final)

target_hshd_matches_final_pyspark.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_matches_final.csv")

# COMMAND ----------

# MAGIC %md ### Matching Data Aggregation

# COMMAND ----------

# Reading the previously-saved KPI data:
#hshd_wkly_kpis_modeling_full = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_kpi_seg_final.csv").cache()

# COMMAND ----------

# Reading the previously-saved matching data:
#target_hshd_matches_final_pyspark = spark.read.format("csv").options(header = 'true').load("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/knox_hshd_matches_final.csv").cache()


# COMMAND ----------

## Organizing the test/control households into a matched household dataframe
target_hshds_final = target_hshd_matches_final_pyspark\
  .drop("matching_metric","match_rank")\
  .withColumnRenamed("jas_test_id","ehhn")\
  .withColumnRenamed("jas_control_id","matched_ehhn")\
  .withColumn("test_control_group", f.lit("T"))\
  .select("ehhn","matched_ehhn","test_control_group","jas_test_pred","jas_control_pred")

control_hshds_final = target_hshd_matches_final_pyspark\
  .drop("matching_metric","match_rank")\
  .withColumnRenamed("jas_control_id","ehhn")\
  .withColumnRenamed("jas_test_id","matched_ehhn")\
  .withColumn("test_control_group", f.lit("C"))\
  .select("ehhn","matched_ehhn","test_control_group","jas_test_pred","jas_control_pred")

matched_hshds_final = target_hshds_final.union(control_hshds_final).cache()



# COMMAND ----------

## Blowing out the distinct HH/week fields into a new dataframe:
weeks_distinct = week_order_i.select('week_start_date').withColumnRenamed("week_start_date","week_start_date_temp").distinct().sort("week_start_date_temp")
hshds_distinct = matched_hshds_final.select('ehhn').withColumnRenamed("ehhn","ehhn_temp").distinct()

full_hshds_weeks = hshds_distinct.crossJoin(weeks_distinct)

# COMMAND ----------

## Organizing weekly KPI fields for the final weekly household output:
hshds_wkly_setup = hshd_wkly_kpis\
  .select("ehhn", "week_start_date",
          "sales","units","visits",
          "pickup_sales", "pickup_units", "pickup_visits",
          "delivery_sales", "delivery_units", "delivery_visits",
          "wag_sales", "wag_units", "wag_visits")\
  .withColumn("kr_sales", (f.col("sales") - f.col("pickup_sales") - f.col("delivery_sales") - f.col("wag_sales")))\
  .withColumn("kr_units", (f.col("units") - f.col("pickup_units") - f.col("delivery_units") - f.col("wag_units")))\
  .withColumn("kr_visits", (f.col("visits") - f.col("pickup_visits") - f.col("delivery_visits") - f.col("wag_visits")))


## Organizing segmentation fields for the final weekly household output:
hshd_segs = hshd_wkly_kpis_modeling_full\
  .select("ehhn", "jas_hshd_flag", "first_jas_visit",
          "funlo_seg_desc","funlo_rollup_desc",
          "convenience_dim_seg","quality_dim_seg","health_dim_seg","inspiration_dim_seg","price_dim_seg",
          "sow_seg","sow_perc",
          "club_prediction","dd_prediction","mass_prediction",
          "pref_store","pref_div")

## Organizing week order fields for the final weekly household output:
week_order_setup = week_order_i\
  .select("week_start_date", "week_start_id", "period_flag", "yoy_period", "model_period_i")

## Joining KPI/segmentation/week order fields:
hshds_wkly_setup_final = full_hshds_weeks\
  .join(week_order_setup, (full_hshds_weeks.week_start_date_temp == week_order_setup.week_start_date), how = 'inner').drop(week_order_setup.week_start_date)\
  .join(hshds_wkly_setup,(full_hshds_weeks.ehhn_temp == hshds_wkly_setup.ehhn) & 
                         (full_hshds_weeks.week_start_date_temp == hshds_wkly_setup.week_start_date), how = 'left').drop(hshds_wkly_setup.ehhn).drop(hshds_wkly_setup.week_start_date)\
  .join(hshd_segs, (full_hshds_weeks.ehhn_temp == hshd_segs.ehhn), how = 'inner').drop(hshd_segs.ehhn)\
  .fillna({'sales': 0}).fillna({'units': 0}).fillna({'visits': 0})\
  .fillna({'kr_sales': 0}).fillna({'kr_units': 0}).fillna({'kr_visits': 0})\
  .fillna({'pickup_sales': 0}).fillna({'pickup_units': 0}).fillna({'pickup_visits': 0})\
  .fillna({'delivery_sales': 0}).fillna({'delivery_units': 0}).fillna({'delivery_visits': 0})\
  .fillna({'wag_sales': 0}).fillna({'wag_units': 0}).fillna({'wag_visits': 0})



# COMMAND ----------


## Breaking out test/control KPIs so they are wide on the dataframe, using test household as the key field:
test_hshds_wkly_final = hshds_wkly_setup_final\
  .join(target_hshd_matches_final_pyspark, (hshds_wkly_setup_final.ehhn_temp == target_hshd_matches_final_pyspark.jas_test_id), how = 'inner')\
  .drop("ehhn_temp","jas_control_pred","matching_metric","match_rank")\
  .withColumnRenamed("sales", "test_sales")\
  .withColumnRenamed("units", "test_units")\
  .withColumnRenamed("visits", "test_visits")\
  .withColumnRenamed("kr_sales", "test_kr_sales")\
  .withColumnRenamed("kr_units", "test_kr_units")\
  .withColumnRenamed("kr_visits", "test_kr_visits")\
  .withColumnRenamed("pickup_sales", "test_pickup_sales")\
  .withColumnRenamed("pickup_units", "test_pickup_units")\
  .withColumnRenamed("pickup_visits", "test_pickup_visits")\
  .withColumnRenamed("delivery_sales", "test_delivery_sales")\
  .withColumnRenamed("delivery_units", "test_delivery_units")\
  .withColumnRenamed("delivery_visits", "test_delivery_visits")\
  .withColumnRenamed("wag_sales", "test_wag_sales")\
  .withColumnRenamed("wag_units", "test_wag_units")\
  .withColumnRenamed("wag_visits", "test_wag_visits")


ctrl_hshds_wkly_final = hshds_wkly_setup_final\
  .join(target_hshd_matches_final_pyspark, (hshds_wkly_setup_final.ehhn_temp == target_hshd_matches_final_pyspark.jas_control_id), how = 'inner')\
  .select("jas_test_id","jas_control_id","jas_control_pred","week_start_date_temp",
          "sales","units","visits",
          "kr_sales","kr_units","kr_visits",
          "pickup_sales","pickup_units","pickup_visits",
          "delivery_sales","delivery_units","delivery_visits",
          "wag_sales","wag_units","wag_visits")\
  .withColumnRenamed("sales", "ctrl_sales")\
  .withColumnRenamed("units", "ctrl_units")\
  .withColumnRenamed("visits", "ctrl_visits")\
  .withColumnRenamed("kr_sales", "ctrl_kr_sales")\
  .withColumnRenamed("kr_units", "ctrl_kr_units")\
  .withColumnRenamed("kr_visits", "ctrl_kr_visits")\
  .withColumnRenamed("pickup_sales", "ctrl_pickup_sales")\
  .withColumnRenamed("pickup_units", "ctrl_pickup_units")\
  .withColumnRenamed("pickup_visits", "ctrl_pickup_visits")\
  .withColumnRenamed("delivery_sales", "ctrl_delivery_sales")\
  .withColumnRenamed("delivery_units", "ctrl_delivery_units")\
  .withColumnRenamed("delivery_visits", "ctrl_delivery_visits")\
  .withColumnRenamed("wag_sales", "ctrl_wag_sales")\
  .withColumnRenamed("wag_units", "ctrl_wag_units")\
  .withColumnRenamed("wag_visits", "ctrl_wag_visits")
                     



# COMMAND ----------

## Final wkly matched household dataframe:
matched_hshds_wkly_final = test_hshds_wkly_final\
  .join(ctrl_hshds_wkly_final, (test_hshds_wkly_final.jas_test_id == ctrl_hshds_wkly_final.jas_test_id) &
                               (test_hshds_wkly_final.jas_control_id == ctrl_hshds_wkly_final.jas_control_id) &
                               (test_hshds_wkly_final.week_start_date_temp == ctrl_hshds_wkly_final.week_start_date_temp), how = 'inner')\
  .drop(ctrl_hshds_wkly_final.jas_test_id).drop(ctrl_hshds_wkly_final.jas_control_id)\
  .drop(ctrl_hshds_wkly_final.week_start_date_temp)\
  .withColumn("match_run", f.lit(i)).cache()

# COMMAND ----------

## Writing the final wkly matched household dataframe:
matched_hshds_wkly_final.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/jas_matched_hshds_wkly_final.csv")

# COMMAND ----------


## Creating the ANCOVA dataset
hshd_match_ancova_data = matched_hshds_final\
  .join(hshd_wkly_kpis_modeling_full,  (matched_hshds_final.ehhn == hshd_wkly_kpis_modeling_full.ehhn)).drop(hshd_wkly_kpis_modeling_full.ehhn)\
  .select("ehhn","matched_ehhn","test_control_group",
          "jas_test_pred","jas_control_pred","first_jas_visit",
          "tot_sales_52","tot_sales_post",
          "tot_units_52","tot_units_post",
          "tot_visits_52","tot_visits_post",
          "kr_sales_52","kr_sales_post",
          "kr_units_52","kr_units_post",
          "kr_visits_52","kr_visits_post",
          "pickup_sales_52","pickup_sales_post",
          "pickup_units_52","pickup_units_post",
          "pickup_visits_52","pickup_visits_post",
          "delivery_sales_52","delivery_sales_post",
          "delivery_units_52","delivery_units_post",
          "delivery_visits_52","delivery_visits_post",
          "wag_sales_post",
          "wag_units_post",
          "wag_visits_post")\
  .withColumn("match_run", f.lit(i))

# COMMAND ----------

## Write Final Modeling data
hshd_match_ancova_data.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/jas_hshd_ancova_data.csv")

# COMMAND ----------

# MAGIC %md ### ANCOVA

# COMMAND ----------

## Convert to pandas dataframe
hshd_match_ancova_data_pd = hshd_match_ancova_data.toPandas()
## Convert test/control field to binary (for OLS in statmodel package)
hshd_match_ancova_data_pd.loc[hshd_match_ancova_data_pd['test_control_group'] == 'T', 'ancova_group'] = 1
hshd_match_ancova_data_pd.loc[hshd_match_ancova_data_pd['test_control_group'] == 'C', 'ancova_group'] = 0

## List of metrics for ANCOVA:
metrics_list = ["kr_visits","kr_units","kr_sales","pickup_visits","pickup_units","pickup_sales"]

metric_count = len(metrics_list)

##ANCOVA calculation loop:
for m in range(metric_count):

  # mth metric, converting metric names:
  metric = metrics_list[m]
  metric_pre_name = (metric + "_52")
  metric_post_name = (metric + "_post")
  
  # Renaming metric fields for ANCOVA
  hshd_match_ancova_loop_data = hshd_match_ancova_data_pd.rename(columns = {metric_pre_name: "pre_metric",metric_post_name: "post_metric"}, inplace=False)
  hshd_match_ancova_loop_data["pre_metric"] = hshd_match_ancova_loop_data["pre_metric"].astype(float)
  hshd_match_ancova_loop_data["post_metric"] = hshd_match_ancova_loop_data["post_metric"].astype(float)
  
  ## Linear Model predicting post-period sales with pre-period sales, with test/control group as an additional variable
  ancova_y = hshd_match_ancova_loop_data["post_metric"].astype(float)
  ancova_x = np.array(hshd_match_ancova_loop_data[["pre_metric","ancova_group"]]).astype(float)
  # adding the constant term
  ancova_x = sm.add_constant(ancova_x)
  post_pre_model = sm.OLS(ancova_y, ancova_x).fit()
  
  ## Averages across test/control household groups:
  avg_data = hshd_match_ancova_loop_data.groupby("ancova_group").agg(pre_metric=pd.NamedAgg(column="pre_metric", aggfunc="mean"),
                                                                     post_metric=pd.NamedAgg(column="post_metric", aggfunc="mean")).reset_index(drop=False)
  
  ## Average of ALL households in the set, with false test/control flags, to calculate LS means              
  lsmeans_pred_data_pd = pd.DataFrame({"pre_metric":[hshd_match_ancova_loop_data["pre_metric"].mean(),hshd_match_ancova_loop_data["pre_metric"].mean()],
                                       "ancova_group":[0.0,1.0]})
  lsmeans_pred_data = np.array(lsmeans_pred_data_pd[["pre_metric","ancova_group"]]).astype(float)
  lsmeans_pred_data = sm.add_constant(lsmeans_pred_data, has_constant='add')
                                 
  ## Calculating LS means for control/test household groups
  lsmeans_preds = post_pre_model.predict(lsmeans_pred_data)
  
  ## Building a summary table
  ancova_results = pd.DataFrame({"metric": metric,
                                 "control_pre_avg": avg_data["pre_metric"][0],
                                 "test_pre_avg": avg_data["pre_metric"][1],
                                 "control_post_avg": avg_data["post_metric"][0],
                                 "test_post_avg": avg_data["post_metric"][1],
                                 "control_post_pred_avg": lsmeans_preds[[0]],
                                 "test_post_pred_avg": lsmeans_preds[[1]],
                                 "test_impact": lsmeans_preds[[1]] - lsmeans_preds[[0]],
                                 "percent_impact": lsmeans_preds[[1]]/lsmeans_preds[[0]] - 1,
                                 "p_value": post_pre_model.pvalues[2]})
  
  ancova_results['significance'] = np.where(ancova_results['p_value'].between(0, 0.1, inclusive=True), 'Significant', 
                                   np.where(ancova_results['p_value'].between(0.1, 0.2, inclusive=True), 'Directional', 'Not Significant'))
                
  if m == 0:
    ancova_results_aggr = ancova_results
  else:
    ancova_results_aggr = pd.concat([ancova_results_aggr, ancova_results], ignore_index=True)

# COMMAND ----------

## Outputing the ANCOVA results:
ancova_results_aggr["match_run"] = i

ancova_results_aggr_pyspark = spark.createDataFrame(ancova_results_aggr)

## Write ANCOVA results
ancova_results_aggr_pyspark.repartition(1)\
  .write.format("csv").mode('overwrite').options(header = 'true').save("abfss://users@sa8451sasdev.dfs.core.windows.net/l236076/jas_hshd_ancova_results.csv")
