from pyspark.sql import SparkSession
from pyspark.sql.functions import count, when, col, hour, dayofweek, weekofyear, to_timestamp
from pyspark.sql import functions as F
from pyspark.sql.functions import radians, sin, cos, sqrt, atan2, round, to_timestamp
from pyspark.ml import Pipeline
import pandas as pd 

from functools import reduce
import matplotlib.pyplot as plt 


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Fraud Detection") \
    .getOrCreate()
# Load data without schema to inspect
df_raw = spark.read.csv("fraud_test.csv", header=True)

# Drop the index column if necessary
df1 = df_raw.drop(df_raw.columns[0])  # Adjust this if needed

row_count = (df1.count())
print('Total # of Rows:', row_count)

# count of columns
cols_count = (len(df1.columns))
print('Total # of Columns:', cols_count)

# shape
shape = (row_count, cols_count)
print('Total Shape of Dataframe:', shape)
print('Total Data:', (row_count * cols_count))

# column names
names = df1.columns
print('Column Names:', names)

# schema
# df1.printSchema()

# df1.describe().show()

# Convert "dob" to Age
df_reduced = df1.withColumn(
    "Age",
    F.year(F.current_date()) - F.year(F.to_date(df1["dob"], "dd/MM/yyyy"))
)
# Dropping columns
columns_to_drop = ['category', 'first', 'last', 'gender', 'street', 'state',
                   'zip', 'city_pop', 'job', 'trans_num', 'unix_time', 'dob']

df_dropped = df_reduced.drop(*columns_to_drop)

df_dropped.show(1)

# checking for total null values after dropping columns
print("Null Value Count:")
null_counts = df_dropped.select([count(
    when(col(c).isNull(), c)).alias(c) for c in df_dropped.columns])
null_counts.show()

# Rename columns for clarity
# # Rename columns to more descriptive names
print("Renamed columns for readablity")
rename_columns = {
    "cc_num": "credit_card_number",
    "amt": "total_amount",
    "lat": "latitude",
    "long": "longitude",
    "merch_lat": "merchant_latitude",
    "merch_long": "merchant_longitude",
    "Age": "age"
}

df = reduce(lambda df_dropped, col: df_dropped.withColumnRenamed(
    col, rename_columns[col]), rename_columns.keys(), df_dropped)
df.show(1)


# Change date and time to individual columns
print("Transform trans_date_trans_time to their own individual columns(trans timestamp, hour, day of the week, week of the year)")
df = df.withColumn("trans_timestamp", to_timestamp("trans_date_trans_time", "dd/MM/yyyy HH:mm"))
df = df.withColumn("trans_hour", hour("trans_timestamp"))
df = df.withColumn("trans_day_of_week", dayofweek("trans_timestamp"))
df = df.withColumn("trans_week_of_year", weekofyear("trans_timestamp"))

print("this is the schema")
df.printSchema()

# Filter fraud data 
fraudulent_transaction = df.filter(col('is_fraud') == 1)

# Distribution of Transactions 
transaction_analysis_amount = df.select("total_amount").describe()
transaction_analysis_amount.show()

amounts = df.select("total_amount").toPandas()
plt.hist(amounts['total_amount'], bins = 50)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amounts')
plt.ylabel('Frequency') 
# plt.show()

# Change age to age group
df = df.withColumn(
    "age_group",
    F.when(F.col("age") <= 18, "Teenager")
    .when((F.col("age") > 18) & (F.col("age") <= 25), "Young Adult") 
    .when((F.col("age") > 25) & (F.col("age") <= 64), "Adult")
    .otherwise("Elder")
)

# Removed the "fraud_" in merchant 
df = df.withColumn("merchant", F.regexp_replace("merchant", "^fraud_", ""))

#  This function will convert both user and merchant latitude/longitude to total distance of that transaction

# Define the earth's radius in km
earth_radius_km = 6371.0

df = df.withColumn("lat_rad", radians(df["latitude"]))
df = df.withColumn("long_rad", radians(df["longitude"]))
df = df.withColumn("merch_lat_rad", radians(df["merchant_latitude"]))
df = df.withColumn("merch_long_rad", radians(df["merchant_longitude"]))

# computing deltas 
df = df.withColumn("delta_lat", df["merch_lat_rad"] - df["lat_rad"])
df = df.withColumn("delta_long", df["merch_long_rad"] - df["long_rad"])

# Applying Haversine formula 
df = df.withColumn("a",
                   sin(df["delta_lat"] / 2) ** 2 + 
                   cos(df["lat_rad"]) * cos(df["merch_lat_rad"]) * sin(df["delta_long"] / 2) ** 2 
                   )

df = df.withColumn("c", 2 * atan2(sqrt(df["a"]), sqrt(1 - df["a"])))
df = df.withColumn("distance_km", earth_radius_km * df["c"])

# dropping intermediate columns 
df = df.drop("lat_rad", "long_rad", "merch_lat_rad", "merch_long_rad", "delta_lat", "delta_long", "a", "c")
df = df.withColumn("distance_km", round(col("distance_km"), 2)) # change rounded value after if it makes difference on xgboost 
df.select("distance_km").show()

df.select("distance_km").describe().show()



df.show(1)

# df = df.withColumn("trans_timestamp", col("trans_timestamp").cast("timestamp"))
# df_pandas = df.toPandas()
df.describe().show()

updated_names = df.columns
print('Column Names:', updated_names)


extra_columns_drop = ("trans_date_trans_time", "credit_card_number", "age", "merchant_latitude", "merchant_longitude", "latitude", "longitude") 
df = df.drop(*extra_columns_drop)


# Cast 'trans_timestamp' as string in Spark
df_string_timestamp = df.withColumn('trans_timestamp', df['trans_timestamp'].cast('string'))

# Convert the entire DataFrame (including string timestamp) to Pandas
df_pandas = df_string_timestamp.toPandas()

# Recast 'trans_timestamp' as datetime64[ns] in Pandas
df_pandas['trans_timestamp'] = pd.to_datetime(df_pandas['trans_timestamp'], errors='coerce')

# Verify the first few rows of the Pandas DataFrame
print(df_pandas.head())
df_pandas.to_csv('fraud_data_pandas.csv', index=False)

# Why did we do this?
# TypeError: Casting to unit-less dtype 'datetime64' is not supported.
# PySpark 3.4.0 and Pandas 2.0.2 "timestamp" was causing errors during the conversion
# Why not use XGBoost with spark?
# I'm more comfortable with pandas library especially when we are working with a medium dataset, 
# it is more efficent and has faster processing.
# Why did I do my preproccessing in Spark? It's very fast and efficient 




