from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predictor', methods=['GET'])
def predictor():
    return render_template('predictor.html')

@app.route('/predictions', methods=['GET'])
def predictions():
    return render_template('predictions.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        return {"message": "Predicción realizada con éxito."}
    except:
        return {"error": "Error inesperado."}
    
@app.route('/show_predictions', methods=['GET'])
def show_predictions():
    try:
        return {"predictions": "Predicciones."}
    except:
        return {"error": "Error inesperado."}

if __name__ == '__main__':
    app.run(debug= 'True')