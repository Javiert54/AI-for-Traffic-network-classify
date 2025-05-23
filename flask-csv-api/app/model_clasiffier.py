# predict_flow.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json

# Definir la clase del modelo (debe ser idéntica a la usada para el entrenamiento)
class NetFlowLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

def load_artifacts(model_params_path='model_params.json',
                   model_state_dict_path='netflow_lstm_model_state_dict.pth',
                   scaler_path='scaler.joblib',
                   label_mapping_path='label_mapping.json'):
    """Carga todos los artefactos necesarios para la predicción."""
    try:
        # Cargar parámetros del modelo
        with open(model_params_path, 'r') as f:
            params = json.load(f)

        INPUT_DIM = params['num_features']
        SEQUENCE_LENGTH = params['sequence_length']
        HIDDEN_DIM = params['hidden_dim']
        OUTPUT_DIM = params['output_dim']
        FEATURE_COLUMNS = params['feature_columns']

        # Cargar mapeo de etiquetas
        with open(label_mapping_path, 'r') as f:
            label_mapping_str_keys = json.load(f)
        # Convertir claves de cadena de JSON de nuevo a enteros
        label_mapping = {int(k): v for k, v in label_mapping_str_keys.items()}

        # Inicializar y cargar el modelo
        model = NetFlowLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        model.load_state_dict(torch.load(model_state_dict_path))
        model.eval() # Poner el modelo en modo de evaluación

        # Cargar el escalador
        scaler = joblib.load(scaler_path)
        
        return model, scaler, label_mapping, SEQUENCE_LENGTH, INPUT_DIM, FEATURE_COLUMNS
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error al cargar artefactos: {e}. Asegúrate de que todos los archivos (.pth, .joblib, .json) estén presentes.")
    except Exception as e:
        raise Exception(f"Un error ocurrió al cargar los artefactos: {e}")


def preprocess_input_csv(csv_path, sequence_length, num_features, feature_columns, scaler_obj):
    """
    Lee un CSV, lo preprocesa para el modelo LSTM.
    Asume que el CSV representa una única secuencia de paquetes.
    El CSV debe contener las características usadas durante el entrenamiento, en el mismo orden.
    """
    try:
        df_input = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV {csv_path}: {e}")

    # Asegurar que todas las columnas de características requeridas estén presentes
    missing_cols = [col for col in feature_columns if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"El CSV de entrada no contiene las columnas requeridas: {missing_cols}.\nColumnas esperadas: {feature_columns}")

    # Seleccionar y reordenar columnas para que coincidan con los datos de entrenamiento
    df_features = df_input[feature_columns]

    # Convertir a array numpy del tipo apropiado
    try:
        sequence_X = df_features.values.astype(np.float32)
    except ValueError as e:
        raise ValueError(f"Error al convertir datos CSV a tipos numéricos. Asegúrate de que todas las columnas de características sean numéricas. Detalles: {e}")

    # Padding o truncamiento de la secuencia
    if len(sequence_X) < sequence_length:
        pad_width = sequence_length - len(sequence_X)
        padding = np.zeros((pad_width, num_features), dtype=np.float32)
        sequence_X = np.vstack((sequence_X, padding))
    elif len(sequence_X) > sequence_length:
        sequence_X = sequence_X[:sequence_length, :]

    # Escalar las características
    try:
        scaled_X = scaler_obj.transform(sequence_X)
    except Exception as e:
        raise ValueError(f"Error al aplicar la transformación del escalador: {e}. Asegúrate de que los datos de entrada coincidan con las características esperadas por el escalador.")
    
    # Remodelar para LSTM: (batch_size, sequence_length, num_features)
    # Para una predicción de un solo CSV, batch_size es 1
    scaled_X_reshaped = scaled_X.reshape(1, sequence_length, num_features)
    return torch.tensor(scaled_X_reshaped, dtype=torch.float32)

def predict_from_tensor(model, input_tensor, label_mapping_dict):
    """
    Realiza una predicción a partir de un tensor de entrada preprocesado.
    """
    with torch.no_grad(): # Desactivar cálculo de gradientes para inferencia
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()

    predicted_label_code = predicted_idx
    predicted_label_name = label_mapping_dict.get(predicted_label_code, f"Código desconocido: {predicted_label_code}")
    
    return predicted_label_name, probabilities.numpy().flatten(), predicted_label_code

def predict_from_csv_path(input_csv_file_path: str):
    """
    Carga artefactos, preprocesa un archivo CSV y devuelve la predicción.

    Args:
        input_csv_file_path (str): Ruta al archivo CSV de entrada.

    Returns:
        tuple: (nombre_clase_predicha, probabilidades_clase, codigo_clase_predicha)
               Retorna None, None, None en caso de error.
    """
    try:
        # Cargar todos los artefactos
        loaded_model, loaded_scaler, loaded_label_mapping, loaded_seq_length, loaded_input_dim, loaded_feature_columns = load_artifacts()

        # Preprocesar el CSV de entrada
        input_tensor_data = preprocess_input_csv(input_csv_file_path, loaded_seq_length, loaded_input_dim, loaded_feature_columns, loaded_scaler)

        # Realizar la predicción
        pred_name, pred_probs, pred_code = predict_from_tensor(loaded_model, input_tensor_data, loaded_label_mapping)

        return pred_name, pred_probs, pred_code, loaded_label_mapping

    except ValueError as ve_error:
        print(f"Error de Valor durante la predicción: {ve_error}")
    except FileNotFoundError as fnfe_error:
        print(f"Error de Archivo No Encontrado: {fnfe_error}. Asegúrate de que todos los archivos de artefactos estén en el directorio actual o en las rutas especificadas.")
    except Exception as general_error:
        print(f"Ocurrió un error inesperado: {general_error}")
    return None, None, None, None
