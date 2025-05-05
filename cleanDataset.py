import pandas as pd
import numpy as np

def clean_dataset(df, threshold=0.8):
# Cargar el dataset (reemplaza 'dataset.csv' con el nombre de tu archivo)
    df = pd.read_csv("dataset.csv")

    # Función para determinar si una columna es mayormente numérica
    def is_mostly_numeric(series, threshold):
        numeric_values = pd.to_numeric(series, errors='coerce')
        numeric_ratio = numeric_values.notna().sum() / len(series)
        return numeric_ratio >= threshold

    # Identificar columnas mayormente numéricas
    numeric_columns = [col for col in df.columns if is_mostly_numeric(df[col], threshold)]

    # Convertir solo las columnas mayormente numéricas a valores numéricos y eliminar filas con valores no numéricos
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_cleaned = df.dropna(subset=numeric_columns)

    # Guardar el dataset limpio
    df_cleaned.to_csv("dataset_limpio.csv", index=False)

    print("Dataset limpio guardado como 'dataset_limpio.csv'")

BASE_DIR = os.getcwd() # Or specify a fixed base path if needed
DATASET_DIR = os.path.join(BASE_DIR, "datasets") # Input directory
csv_files = glob(os.path.join(DATASET_DIR, "*.csv")) # Find all CSV files
for file in csv_files:
    df = pd.read_csv(file)
    clean_dataset(df, threshold=0.8) # Call the function for each file
