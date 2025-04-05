from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

model = joblib.load("model.pkl")       
encoder = joblib.load("encoder.pkl")   
scaler = joblib.load("scaler.pkl")     

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/result", methods=["GET"])
def result():
    try:
        city = request.args.get("city")
        distance = float(request.args.get("distance"))
        time_taken = float(request.args.get("time_taken"))
        transport_mode = request.args.get("transport_mode")

        co2_saved = float(predict_co2(distance, time_taken, transport_mode))
        co2_saved = round(co2_saved, 2)

        # Pass transport_mode to be saved
        save_to_excel(city, distance, time_taken, transport_mode, co2_saved)

        return render_template("result.html", city=city, transport_mode=transport_mode, co2_saved=co2_saved)

    except Exception as e:
        print("Error:", e)
        return f"Error: {str(e)}", 400

def predict_co2(distance, time_taken, transport_mode):
    all_modes = encoder.categories_[0].tolist()
    encoded_columns = [f"Mode_of_Transport_{mode}" for mode in all_modes]
    encoded_values = [1 if transport_mode == mode else 0 for mode in all_modes]
    encoded_df = pd.DataFrame([encoded_values], columns=encoded_columns)

    input_df = pd.concat([
        encoded_df,
        pd.DataFrame([[distance, time_taken]], columns=["Distance_km", "Time_Taken_min"])
    ], axis=1)

    input_df = input_df[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

def save_to_excel(city, distance, time_taken, transport_mode, co2_saved):
    excel_path = "co2_savings_log.xlsx"
    new_data = pd.DataFrame([{
        "City": city,
        "Distance (km)": distance,
        "Time Taken (min)": time_taken,
        "Mode of Transport": transport_mode,
        "Predicted CO₂ Saved (kg)": co2_saved
    }])

    if os.path.exists(excel_path):
        existing = pd.read_excel(excel_path)
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        combined = new_data

    combined.to_excel(excel_path, index=False)

if __name__ == "__main__":
    app.run(debug=True)
