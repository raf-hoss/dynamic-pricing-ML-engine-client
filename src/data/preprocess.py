import pandas as pd
from pyspark.sql import DataFrame as SparkDF
from pyspark.sql.functions import col, when
import os

# Basic preprocessing for pandas DataFrame
def preprocess_pandas(sales_df: pd.DataFrame) -> pd.DataFrame:
    df = sales_df.copy()

    # Drop rows with nulls in key columns
    df.dropna(subset=["price", "cost", "units_sold", "competitor_price"], inplace=True)

    # Feature: price margin
    df["price_margin"] = df["price"] - df["cost"]

    # Feature: price to competitor ratio
    df["price_vs_competitor"] = df["price"] / df["competitor_price"]

    # Flag unreasonable prices
    df = df[df["price"] > 0]
    df = df[df["cost"] > 0]
    df = df[df["units_sold"] >= 0]

    return df

# Basic preprocessing for Spark DataFrame
def preprocess_spark(sales_df: SparkDF) -> SparkDF:
    df = sales_df

    # Remove nulls in important columns
    df = df.dropna(subset=["price", "cost", "units_sold", "competitor_price"])

    # Feature: price margin
    df = df.withColumn("price_margin", col("price") - col("cost"))

    # Feature: price to competitor ratio
    df = df.withColumn("price_vs_competitor", col("price") / col("competitor_price"))

    # Filter: valid prices and units sold
    df = df.filter((col("price") > 0) & (col("cost") > 0) & (col("units_sold") >= 0))

    return df

# Save cleaned pandas DataFrame to CSV
def save_processed_pandas(df: pd.DataFrame, output_path: str = "data/processed/clean_sales_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    from load_data import load_all

    # Load data
    data = load_all()
    pandas_sales = data["pandas"]["sales"]
    spark_sales = data["spark"]["sales"]

    # Process
    cleaned_pandas = preprocess_pandas(pandas_sales)
    cleaned_spark = preprocess_spark(spark_sales)

    # Save
    save_processed_pandas(cleaned_pandas)
    print("✅ Preprocessing complete. Saved cleaned pandas dataset to data/processed/")