import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import random
import os

# Set random seed for reproducibility
random.seed(22)
np.random.seed(22)

# Parameters
NUM_PRODUCTS = 1000
DAYS = 365
START_DATE = datetime(2024, 1, 1)

CATEGORY_SPLIT = {
    "Women_Clothing": 300,
    "Men_Clothing": 300,
    "Men_Shoes": 80,
    "Women_Shoes": 150,
    "Men_Accessories": 70,
    "Women_Accessories": 100
}

HOLIDAYS = [
    {"name": "Eid_ul_Fitr", "start": "2024-04-10", "length": 7},
    {"name": "Eid_ul_Adha", "start": "2024-06-15", "length": 8},
    {"name": "Other_Holiday", "start": "2024-12-20", "length": 5}
]

# Helper to create date range
def create_date_range():
    return pd.date_range(start=START_DATE, periods=DAYS, freq="D")

# Generate product catalog with SKU-level attributes
def generate_product_catalog():
    skus = []
    for cat, count in CATEGORY_SPLIT.items():
        for i in range(count):
            skus.append({
                "sku_id": f"{cat[:2].upper()}{i:03d}",
                "category": cat,
                "base_price": round(random.uniform(20, 300), 2),
                "cost": round(random.uniform(10, 100), 2)
            })
    return pd.DataFrame(skus)

# Generate daily sales with promotions, competitor pricing, and weather features
def generate_sales_data(products, dates):
    rows = []
    for _, row in products.iterrows():
        for date in dates:
            demand = max(0, int(np.random.normal(5, 3)))  # daily demand
            promo = 0
            for h in HOLIDAYS:
                holiday_dates = pd.date_range(start=h["start"], periods=h["length"])
                if date in holiday_dates:
                    promo = 0.3 if "Eid" in h["name"] else 0.25
                    demand += np.random.poisson(3)
                    break

            competitor_price = round(row["base_price"] * np.random.uniform(0.85, 1.15), 2)
            temperature = np.random.normal(25, 5)  # simple weather feature

            rows.append({
                "date": date,
                "sku_id": row["sku_id"],
                "category": row["category"],
                "price": round(row["base_price"] * (1 - promo), 2),
                "cost": row["cost"],
                "units_sold": demand,
                "promo_discount": promo,
                "competitor_price": competitor_price,
                "temperature": temperature
            })
    return pd.DataFrame(rows)

# Generate and save the datasets
def main():
    print("Generating synthetic product catalog and sales data...")

    dates = create_date_range()
    products = generate_product_catalog()
    sales = generate_sales_data(products, dates)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    products.to_csv(f"{output_dir}/product_catalog.csv", index=False)
    sales.to_csv(f"{output_dir}/sales_data.csv", index=False)

    print(f"Saved product catalog and sales data to {output_dir}")

if __name__ == "__main__":
    main()