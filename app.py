from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
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
        co2_saved = round(co2_saved, 2)  # <-- round it here

        return render_template("result.html", city=city, transport_mode=transport_mode, co2_saved=co2_saved)
    
    except Exception as e:
        print("Error:", e)
        return f"Error: {str(e)}", 400
        
def predict_co2(distance, time_taken, transport_mode):
    # Get transport modes from encoder
    all_modes = encoder.categories_[0].tolist()
    
    # Create a DataFrame for one-hot encoding
    encoded_columns = [f"Mode_of_Transport_{mode}" for mode in all_modes]
    encoded_values = [1 if transport_mode == mode else 0 for mode in all_modes]
    encoded_df = pd.DataFrame([encoded_values], columns=encoded_columns)

    # Create DataFrame for numerical inputs
    input_df = pd.concat([encoded_df, pd.DataFrame([[distance, time_taken]], columns=["Distance_km", "Time_Taken_min"])], axis=1)

    # Ensure feature order matches training
    input_df = input_df[scaler.feature_names_in_]  # Align feature names with training

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

if __name__ == "__main__":
    app.run(debug=True)
