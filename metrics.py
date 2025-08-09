
def count_total_nulls(df):
    return df.isnull().sum().sum()