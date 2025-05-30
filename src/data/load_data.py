# %%
import pandas as pd
from pyspark.sql import SparkSession
import os

# Initialize Spark session
def start_spark(app_name="DynamicPricingSparkApp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

# Load raw CSVs as pandas DataFrames
def load_data_pandas(data_dir="data/raw"):
    product_catalog = pd.read_csv(os.path.join(data_dir, "product_catalog.csv"))
    sales_data = pd.read_csv(os.path.join(data_dir, "sales_data.csv"), parse_dates=["date"])
    return product_catalog, sales_data

# Convert pandas DataFrames to Spark DataFrames
def load_data_spark(spark, data_dir="data/raw"):
    product_catalog = spark.read.csv(os.path.join(data_dir, "product_catalog.csv"), header=True, inferSchema=True)
    sales_data = spark.read.csv(os.path.join(data_dir, "sales_data.csv"), header=True, inferSchema=True)
    return product_catalog, sales_data

# Load both formats if needed
def load_all(data_dir="data/raw"):
    spark = start_spark()
    pandas_catalog, pandas_sales = load_data_pandas(data_dir)
    spark_catalog, spark_sales = load_data_spark(spark, data_dir)
    return {
        "pandas": {
            "catalog": pandas_catalog,
            "sales": pandas_sales
        },
        "spark": {
            "catalog": spark_catalog,
            "sales": spark_sales
        }
    }

if __name__ == "__main__":
    # Test loading
    all_data = load_all()
    print("Pandas Sales Data Sample:")
    print(all_data["pandas"]["sales"].head())

    print("\nSpark Sales Data Schema:")
    all_data["spark"]["sales"].printSchema()


