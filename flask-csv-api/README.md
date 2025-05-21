# Flask CSV API

This project is a simple Flask API that allows users to upload CSV files for processing. The API provides endpoints to handle file uploads and extract relevant data from the CSV files.

## Project Structure

```
flask-csv-api
├── app
│   ├── __init__.py
│   ├── routes.py
│   └── utils.py
├── tests
│   └── test_routes.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-csv-api
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   flask run
   ```

2. Use a tool like Postman or cURL to send a POST request to the `/upload` endpoint with a CSV file.

   Example using cURL:
   ```
   curl -X POST -F "file=@path_to_your_file.csv" http://127.0.0.1:5000/upload
   ```

## Testing

To run the tests, ensure that your virtual environment is activated and run:
```
pytest
```

## License

This project is licensed under the MIT License.