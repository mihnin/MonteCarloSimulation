import pandas as pd
import numpy as np

def load_data(file):
    try:
        if file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:
            raise ValueError("Unsupported file format. Please upload an Excel (.xlsx, .xls) or CSV file.")
        
        if df.empty:
            raise ValueError("The uploaded file is empty.")
        
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or contains no valid data.")
    except pd.errors.ParserError:
        raise ValueError("Unable to parse the file. Please check if it's a valid Excel or CSV file.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading the file: {str(e)}")

def preprocess_data(df):
    # Remove any rows with missing values
    df.dropna(inplace=True)

    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_columns
