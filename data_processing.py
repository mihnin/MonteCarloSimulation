import pandas as pd
import numpy as np

def load_data(file):
    try:
        if file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def preprocess_data(df):
    # Remove any rows with missing values
    df.dropna(inplace=True)

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_columns
