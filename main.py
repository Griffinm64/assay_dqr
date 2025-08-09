import pandas as pd 
from files import load_dataframe
from metrics import count_total_nulls


def main():
    print("Starting Assay DQR...")

    data_filepath = "./data/sample/product_sales.csv"
    df = load_dataframe(data_filepath)
    print(f"dataframe loaded successfully from {data_filepath}")

    print(df.head())


    print("=" * 7, "Summary Report", "=" * 7)
    print(f"Total Null values: {count_total_nulls(df)}")



if __name__ == "__main__":
    main()