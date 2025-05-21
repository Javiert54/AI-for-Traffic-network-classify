def read_csv(file):
    import pandas as pd
    from io import StringIO

    # Read the CSV file into a DataFrame
    try:
        data = pd.read_csv(StringIO(file.read().decode('utf-8')))
        return data
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def extract_relevant_data(data):
    # Assuming we want to extract specific columns from the DataFrame
    # Modify the column names as per your CSV structure
    relevant_columns = ['column1', 'column2']  # Replace with actual column names
    if not all(col in data.columns for col in relevant_columns):
        raise ValueError(f"CSV is missing required columns: {relevant_columns}")
    
    return data[relevant_columns]