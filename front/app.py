from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__, template_folder='../templates')

# Load model and feature columns
model = joblib.load('score_predictor.pkl')
model_columns = joblib.load('model_columns.pkl')

# Route: Home
@app.route("/")
def home():
    return render_template("index.html")

# Route: Predict
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

        # Create input dict
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

        # Format input as DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        return render_template("index.html", prediction_text=f"Predicted Score: {round(prediction)}")

    except Exception as e:
        return f"Error occurred: {e}"

# ✅ Updated for Docker container access
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
