from src.data.load_data import load_data_pandas
from src.data.preprocess import preprocess_pandas

def test_preprocess_returns_clean_df():
    _, sales_df = load_data_pandas()
    clean_df = preprocess_pandas(sales_df)
    assert clean_df.isnull().sum().sum() == 0, "Null values remain in cleaned data"
    assert "price_margin" in clean_df.columns