import pandas as pd
import numpy as np
import os
from glob import glob

def clean_and_convert_by_majority_type_chunked(input_filepath, output_filepath, chunksize=10000):
    """
    Limpia un archivo CSV en bloques, intentando convertir valores al tipo de dato mayoritario
    en cada columna antes de eliminarlos, y guarda el resultado sobrescribiendo un archivo.

    Args:
        input_filepath (str): Ruta del archivo CSV original.
        output_filepath (str): Ruta del archivo donde guardar el CSV limpio.
        chunksize (int): Tamaño del bloque para leer el archivo en partes.
    """
    def get_majority_dtype(series):
        dtype_counts = series.map(type).value_counts()
        if not dtype_counts.empty:
            return dtype_counts.idxmax()
        return None

    def convert_value(value, target_type):
        try:
            return target_type(value)
        except (ValueError, TypeError):
            return np.nan

    cleaned_chunks = []

    for chunk in pd.read_csv(input_filepath, chunksize=chunksize):
        for col in chunk.columns:
            majority_dtype = get_majority_dtype(chunk[col])
            if majority_dtype:
                chunk[col] = chunk[col].apply(lambda x: convert_value(x, majority_dtype))

        chunk_cleaned = chunk.dropna()
        cleaned_chunks.append(chunk_cleaned)

    df_final = pd.concat(cleaned_chunks, ignore_index=True)

    # Guardar el dataset limpio
    df_final.to_csv(output_filepath, index=False)

    print(f"Dataset limpio guardado y sobrescrito en: '{output_filepath}'")

BASE_DIR = os.getcwd()  # Or specify a fixed base path if needed
DATASET_DIR = os.path.join(BASE_DIR, "datasets")  # Input directory
csv_files = glob(os.path.join(DATASET_DIR, "*.csv"))  # Find all CSV files

for file in csv_files:
    clean_and_convert_by_majority_type_chunked(file, file)  # Procesar archivo en chunks y sobrescribirlo
