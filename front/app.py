from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging
import socket
import json
from logging.handlers import SocketHandler

app = Flask(__name__, template_folder='../templates')

# Load model and columns
model = joblib.load('score_predictor.pkl')
model_columns = joblib.load('model_columns.pkl')

# ✅ Logstash logger config (host must be 'localhost' unless you're in Docker)
tcp_handler = SocketHandler('localhost', 5000)
tcp_handler.setLevel(logging.INFO)
tcp_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger('logstash')
logger.setLevel(logging.INFO)
logger.addHandler(tcp_handler)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_last_5 = int(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])

        input_dict = {
            'overs': overs,
            'runs': runs,
            'wickets': wickets,
            'runs_last_5': runs_last_5,
            'wickets_last_5': wickets_last_5,
            **{col: 0 for col in model_columns if 'bat_team_' in col},
            **{col: 0 for col in model_columns if 'bowl_team_' in col},
        }
        input_dict[f'bat_team_{bat_team}'] = 1
        input_dict[f'bowl_team_{bowl_team}'] = 1

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        # ✅ Send data to Logstash
        logger.info(json.dumps({
            'event': 'prediction',
            'input': input_dict,
            'prediction': round(prediction),
            'host': socket.gethostname()
        }))

        return render_template("index.html", prediction_text=f"Predicted Score: {round(prediction)}")

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

