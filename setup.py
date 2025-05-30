from setuptools import setup, find_packages

setup(
    name="dynamic_pricing_engine",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "scikit-learn", "lightgbm", "fastapi", "uvicorn", "nannyml", "mlflow"
    ],
)