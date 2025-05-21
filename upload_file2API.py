import requests

url = "http://localhost:5000/upload"  # Cambia el puerto si tu Flask corre en otro
csv_file_path = input("Ingresa la ruta del archivo CSV: ")         # Cambia por la ruta de tu archivo CSV

with open(csv_file_path, 'rb') as f:
    files = {'file': (csv_file_path, f, 'text/csv')}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response:", response.json())