from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
import great_expectations as ge
import sys

# Initialize Glue context
sc    = SparkContext()
glue  = GlueContext(sc)
spark = glue.spark_session

# Base S3 prefix
raw_prefix = "s3://convictionai-data/conviction-ai/raw/"

# Define source paths
paths = {
    'minute':  ('parquet', raw_prefix + 'minute/'),
    'daily':   ('parquet', raw_prefix + 'daily/'),
    'options': ('parquet', raw_prefix + 'options/'),
    'news':    ('parquet', raw_prefix + 'news/'),
    'DXY':     ('csv', raw_prefix + 'macro/DXY_raw.csv'),
    'fed':     ('csv', raw_prefix + 'macro/fed_funds_rate.csv'),
    'FRED':    ('csv', raw_prefix + 'macro/FRED.csv'),
    'VIX':     ('csv', raw_prefix + 'macro/VIX_daily.csv'),
    'vix_json':('json', raw_prefix + 'macro/vix_data.json')
}

# Read into DataFrames
dfs = {}
for name, (fmt, uri) in paths.items():
    try:
        if fmt == 'parquet':
            dfs[name] = spark.read.parquet(uri)
        elif fmt == 'csv':
            dfs[name] = spark.read.option("header", True).csv(uri)
        elif fmt == 'json':
            dfs[name] = spark.read.json(uri)
        print(f"✅ Loaded {name} from {uri}")
    except Exception as e:
        print(f"⚠️ Failed to load {name} from {uri}: {e}")
        sys.exit(1)

# Cast timestamp columns
for name, df in dfs.items():
    if 'timestamp' in df.columns:
        dfs[name] = df.withColumn('timestamp', F.to_timestamp('timestamp'))

# Join order: minute as base
df = dfs['minute']
for key in ['daily', 'options', 'news', 'DXY', 'fed', 'FRED', 'VIX', 'vix_json']:
    df = df.join(dfs[key], on='timestamp', how='left')

# Add day-of-week
df = df.withColumn('dayofweek', F.dayofweek('timestamp'))

# Encode categorical columns to numeric indices
categorical_cols = ['symbol']  # add other categorical columns here as needed
for col in categorical_cols:
    idx = StringIndexer(inputCol=col, outputCol=f"{col}_idx").setHandleInvalid("keep")
    df = idx.fit(df).transform(df)

# -----------------
# Data Quality Validation with Great Expectations
# -----------------
# Convert a sample to Pandas for Great Expectations
sample_pd = df.limit(1000).toPandas()

gdf = ge.from_pandas(sample_pd)
# Define expectations
expectations = [
    gdf.expect_column_values_to_not_be_null("timestamp"),
    gdf.expect_column_min_to_be_between("dayofweek", 1, 7)
]

# Validate expectations
validation = gdf.validate()
if not validation["success"]:
    print("❌ Data quality checks failed:")
    for r in validation["results"]:
        print(r)
    sys.exit(1)
print("✅ Data quality checks passed")

# Write clean dataset
out_path = "s3://convictionai-data/conviction-ai/clean/train_dataset/"
df.repartition(1).write.mode('overwrite').parquet(out_path)
print(f"✅ Cleaned training data written to {out_path}")