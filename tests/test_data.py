from src.data.load_data import load_data_pandas

def test_load_data_shapes():
    catalog, sales = load_data_pandas()
    assert not catalog.empty, "Product catalog is empty"
    assert not sales.empty, "Sales data is empty"
    assert "sku_id" in catalog.columns
    assert "price" in sales.columns