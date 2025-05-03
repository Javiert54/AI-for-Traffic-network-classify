import pandas as pd
from pathlib import Path # Usaremos pathlib para manejar rutas y buscar archivos

def verificar_encabezados_iguales(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
  """
  Comprueba si dos DataFrames de pandas tienen los mismos encabezados (columnas y orden).

  Args:
    df1: El primer DataFrame.
    df2: El segundo DataFrame.

  Returns:
    True si los encabezados son idénticos, False en caso contrario.
  """
  # Obtener las listas de nombres de columnas
  encabezados1 = list(df1.columns)
  encabezados2 = list(df2.columns)

  # Comparar las listas
  return encabezados1 == encabezados2

def verificar_encabezados_directorio(directorio: str) -> bool:
    """
    Verifica si todos los archivos CSV en un directorio tienen el mismo encabezado.

    Args:
        directorio: La ruta al directorio que contiene los archivos CSV.

    Returns:
        True si todos los archivos CSV tienen el mismo encabezado, False en caso contrario.
        También imprime mensajes informativos.
    """
    ruta_directorio = Path(directorio)
    archivos_csv = list(ruta_directorio.glob('*.csv'))

    if not archivos_csv:
        print(f"No se encontraron archivos CSV en el directorio: {directorio}")
        return False # O True si consideras que un directorio vacío cumple la condición

    print(f"Encontrados {len(archivos_csv)} archivos CSV en '{directorio}'. Verificando encabezados...")

    primer_archivo = archivos_csv[0]
    encabezado_referencia = None

    # Leer el encabezado del primer archivo como referencia
    try:
        # Usamos nrows=0 para leer solo el encabezado, es más eficiente
        df_primero = pd.read_csv(primer_archivo, nrows=0)
        encabezado_referencia = list(df_primero.columns)
        print(f"Encabezado de referencia tomado de: {primer_archivo.name}")
        # Opcional: Imprimir el encabezado de referencia si es útil
        # print(f"Encabezado: {encabezado_referencia}")
    except Exception as e:
        print(f"Error al leer el encabezado del primer archivo ({primer_archivo.name}): {e}")
        return False

    # Comparar con los encabezados de los demás archivos
    for archivo_actual in archivos_csv[1:]:
        try:
            df_actual = pd.read_csv(archivo_actual, nrows=0)
            encabezado_actual = list(df_actual.columns)

            if encabezado_actual != encabezado_referencia:
                print(f"\n¡Error! El encabezado de '{archivo_actual.name}' es diferente.")
                print(f"Encabezado esperado ({primer_archivo.name}): {encabezado_referencia}")
                print(f"Encabezado encontrado ({archivo_actual.name}): {encabezado_actual}")
                return False
        except pd.errors.EmptyDataError:
             print(f"Advertencia: El archivo '{archivo_actual.name}' está vacío o no tiene encabezado. Se omite.")
             # Decide cómo manejar archivos vacíos. Aquí los omitimos.
             # Si un archivo vacío significa que los encabezados no son iguales, cambia esto a:
             # print(f"Error: El archivo '{archivo_actual.name}' está vacío.")
             # return False
        except Exception as e:
            print(f"Error al leer el encabezado del archivo ({archivo_actual.name}): {e}")
            # Decide si un error de lectura debe detener la verificación
            return False # Opcional: podrías continuar con otros archivos si prefieres

    print("\n¡Éxito! Todos los archivos CSV en el directorio tienen el mismo encabezado.")
    return True

# --- Ruta al directorio de datasets ---
directorio_datasets = "datasets/"

# --- Ejecutar la verificación ---
verificar_encabezados_directorio(directorio_datasets)

