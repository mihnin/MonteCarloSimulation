import pandas as pd
import numpy as np

def load_data(file):
    try:
        if file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:
            raise ValueError("Неподдерживаемый формат файла. Пожалуйста, загрузите файл Excel (.xlsx, .xls) или CSV.")
        
        if df.empty:
            raise ValueError("Загруженный файл пуст.")
        
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("Загруженный файл пуст или не содержит допустимых данных.")
    except pd.errors.ParserError:
        raise ValueError("Не удалось разобрать файл. Пожалуйста, проверьте, является ли он допустимым файлом Excel или CSV.")
    except Exception as e:
        raise ValueError(f"Произошла ошибка при загрузке файла: {str(e)}")

def preprocess_data(df):
    # Удаление строк с отсутствующими значениями
    df.dropna(inplace=True)

    # Получение числовых столбцов
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_columns
